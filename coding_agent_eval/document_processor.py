import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import hashlib

from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.schema import Document

from models import RawContent, ProcessedDocument, VectorStoreMetadata, DocumentationSource

class DocumentProcessor:
    """
    Processes raw documentation content into structured chunks for vector storage
    """
    
    def __init__(self, 
                 embedding_model: str = "text-embedding-3-small",
                 chunk_size: int = 5000,
                 chunk_overlap: int = 250):
        """
        Initialize document processor
        
        Args:
            embedding_model: OpenAI embedding model to use
            chunk_size: Maximum characters per chunk
            chunk_overlap: Character overlap between chunks
        """
        self.embedding_model = embedding_model
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        
        # Initialize text splitters
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
        )
        
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Content type detection patterns
        self.content_patterns = {
            'code_example': [
                r'```[\s\S]*?```',
                r'`[^`]+`',
                r'^\s*def\s+\w+',
                r'^\s*class\s+\w+',
                r'^\s*import\s+\w+',
                r'^\s*from\s+\w+'
            ],
            'api_reference': [
                r'GET\s+/',
                r'POST\s+/',
                r'PUT\s+/',
                r'DELETE\s+/',
                r'@app\.',
                r'@router\.',
                r'endpoint',
                r'parameter',
                r'response'
            ],
            'tutorial': [
                r'step\s+\d+',
                r'tutorial',
                r'getting started',
                r'quick start',
                r'walkthrough',
                r'guide'
            ],
            'installation': [
                r'pip install',
                r'npm install',
                r'yarn add',
                r'installation',
                r'setup',
                r'requirements'
            ],
            'authentication': [
                r'authentication',
                r'auth',
                r'token',
                r'api key',
                r'login',
                r'credentials'
            ],
            'error_handling': [
                r'error',
                r'exception',
                r'troubleshooting',
                r'debugging',
                r'common issues',
                r'faq'
            ]
        }
    
    def process_raw_content(self, raw_content: RawContent) -> Tuple[List[ProcessedDocument], VectorStoreMetadata]:
        """
        Process raw content into structured documents
        
        Args:
            raw_content: Raw content from fetchers
            
        Returns:
            Tuple of (processed_documents, metadata)
        """
        processed_docs = []
        content_stats = {
            'total_files': 0,
            'total_chunks': 0,
            'content_type_distribution': {},
            'source_distribution': {},
            'code_example_count': 0,
            'api_endpoint_count': 0,
            'tutorial_section_count': 0
        }
        
        if raw_content.source_type == "github":
            processed_docs = self._process_github_content(raw_content, content_stats)
        elif raw_content.source_type == "website":
            processed_docs = self._process_website_content(raw_content, content_stats)
        else:
            raise ValueError(f"Unsupported source type: {raw_content.source_type}")
        
        # Create metadata
        metadata = VectorStoreMetadata(
            total_chunks=len(processed_docs),
            embedding_model=self.embedding_model,
            chunk_strategy="markdown_aware_recursive",
            average_chunk_size=sum(len(doc.content) for doc in processed_docs) // max(len(processed_docs), 1),
            content_type_distribution=content_stats['content_type_distribution'],
            source_distribution=content_stats['source_distribution'],
            code_example_count=content_stats['code_example_count'],
            api_endpoint_count=content_stats['api_endpoint_count'],
            tutorial_section_count=content_stats['tutorial_section_count']
        )
        
        if raw_content.source_type == "github":
            # Add GitHub-specific metadata
            metadata.repo_info = {
                'owner': raw_content.metadata.get('repo_owner', ''),
                'name': raw_content.metadata.get('repo_name', ''),
                'full_name': raw_content.metadata.get('repo_full_name', ''),
                'latest_commit': raw_content.metadata.get('latest_commit_hash', ''),
                'commit_date': raw_content.metadata.get('latest_commit_date', ''),
                'has_readme': raw_content.metadata.get('has_readme', 'false') == 'true'
            }
            
            # Extract file types processed
            file_types = set()
            for file_path in raw_content.content.keys():
                if not file_path.startswith('code:'):
                    file_types.add(Path(file_path).suffix.lower())
            metadata.file_types_processed = list(file_types)
        
        return processed_docs, metadata
    
    def _process_github_content(self, raw_content: RawContent, stats: Dict) -> List[ProcessedDocument]:
        """Process GitHub repository content"""
        processed_docs = []
        
        for file_path, content in raw_content.content.items():
            stats['total_files'] += 1
            stats['source_distribution'][file_path] = stats['source_distribution'].get(file_path, 0) + 1
            
            if file_path.startswith('code:'):
                # Process code files differently
                chunks = self._process_code_file_content(content, file_path[5:])  # Remove 'code:' prefix
            else:
                # Process markdown files
                chunks = self._process_markdown_content(content, file_path)
            
            for chunk in chunks:
                # Detect content type
                content_type = self._detect_content_type(chunk.content)
                chunk.content_type = content_type
                chunk.source_url = raw_content.source_url
                chunk.file_path = file_path
                
                # Update statistics
                stats['content_type_distribution'][content_type] = stats['content_type_distribution'].get(content_type, 0) + 1
                stats['total_chunks'] += 1
                
                if content_type == 'code_example':
                    stats['code_example_count'] += 1
                elif content_type == 'api_reference':
                    stats['api_endpoint_count'] += 1
                elif content_type == 'tutorial':
                    stats['tutorial_section_count'] += 1
                
                processed_docs.append(chunk)
        
        return processed_docs
    
    def _process_website_content(self, raw_content: RawContent, stats: Dict) -> List[ProcessedDocument]:
        """Process website content"""
        processed_docs = []
        
        for url, content in raw_content.content.items():
            if url in ['pages', 'page_count', 'all_urls']:
                # Skip metadata entries
                continue
            
            stats['total_files'] += 1
            stats['source_distribution'][url] = stats['source_distribution'].get(url, 0) + 1
            
            # For website content, we assume it's HTML that has been converted to markdown
            chunks = self._process_markdown_content(content, url)
            
            for chunk in chunks:
                content_type = self._detect_content_type(chunk.content)
                chunk.content_type = content_type
                chunk.source_url = url
                chunk.file_path = None  # Not applicable for websites
                
                # Update statistics
                stats['content_type_distribution'][content_type] = stats['content_type_distribution'].get(content_type, 0) + 1
                stats['total_chunks'] += 1
                
                if content_type == 'code_example':
                    stats['code_example_count'] += 1
                elif content_type == 'api_reference':
                    stats['api_endpoint_count'] += 1
                elif content_type == 'tutorial':
                    stats['tutorial_section_count'] += 1
                
                processed_docs.append(chunk)
        
        return processed_docs
    
    def _process_markdown_content(self, content: str, source_path: str) -> List[ProcessedDocument]:
        """Process markdown content into chunks"""
        chunks = []
        
        # First, try to split by markdown headers
        try:
            header_splits = self.markdown_splitter.split_text(content)
        except Exception:
            # Fallback to treating as plain text
            header_splits = [content]
        
        # Further split large sections
        for i, section in enumerate(header_splits):
            if isinstance(section, str):
                text = section
                metadata = {}
            else:
                # LangChain Document object
                text = section.page_content
                metadata = section.metadata
            
            # Split large sections recursively
            if len(text) > self.recursive_splitter._chunk_size:
                sub_chunks = self.recursive_splitter.split_text(text)
            else:
                sub_chunks = [text]
            
            for j, chunk_text in enumerate(sub_chunks):
                if not chunk_text.strip():
                    continue
                
                # Generate unique chunk ID
                chunk_id = self._generate_chunk_id(source_path, i, j, chunk_text)
                
                # Extract section title from metadata or content
                section_title = self._extract_section_title(chunk_text, metadata)
                
                chunk = ProcessedDocument(
                    chunk_id=chunk_id,
                    source_url="",  # Will be set by caller
                    content=chunk_text.strip(),
                    content_type="",  # Will be detected by caller
                    section_title=section_title,
                    file_path=None,  # Will be set by caller
                    metadata={
                        'chunk_index': len(chunks),
                        'section_metadata': metadata,
                        'word_count': len(chunk_text.split()),
                        'char_count': len(chunk_text)
                    }
                )
                chunks.append(chunk)
        
        return chunks
    
    def _process_code_file_content(self, content: str, source_path: str) -> List[ProcessedDocument]:
        """Process code file content"""
        # For code files, we'll create fewer, larger chunks to preserve context
        chunks = []
        
        # Extract the actual code from markdown code block if present
        code_match = re.search(r'```[\w]*\n([\s\S]*?)\n```', content)
        if code_match:
            actual_code = code_match.group(1)
        else:
            actual_code = content
        
        # Split code into logical sections (by classes, functions, etc.)
        code_chunks = self._split_code_content(actual_code)
        
        for i, chunk_text in enumerate(code_chunks):
            if not chunk_text.strip():
                continue
            
            chunk_id = self._generate_chunk_id(f"code:{source_path}", 0, i, chunk_text)
            
            chunk = ProcessedDocument(
                chunk_id=chunk_id,
                source_url="",  # Will be set by caller
                content=chunk_text.strip(),
                content_type="code_example",
                section_title=f"Code: {Path(source_path).name}",
                file_path=source_path,
                metadata={
                    'chunk_index': i,
                    'is_code_file': True,
                    'word_count': len(chunk_text.split()),
                    'char_count': len(chunk_text)
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_code_content(self, code: str) -> List[str]:
        """Split code into logical chunks"""
        # Simple approach: split by classes and functions
        chunks = []
        current_chunk = ""
        
        lines = code.split('\n')
        for line in lines:
            # Check if line starts a new class or function
            if re.match(r'^\s*(class|def|function|async def)\s+\w+', line):
                if current_chunk.strip():
                    chunks.append(current_chunk)
                current_chunk = line + '\n'
            else:
                current_chunk += line + '\n'
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk)
        
        # If no logical splits found, just return the whole content
        if not chunks:
            chunks = [code]
        
        return chunks
    
    def _detect_content_type(self, content: str) -> str:
        """Detect the type of content based on patterns"""
        content_lower = content.lower()
        
        # Calculate scores for each content type
        type_scores = {}
        for content_type, patterns in self.content_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, content_lower, re.MULTILINE | re.IGNORECASE))
                score += matches
            type_scores[content_type] = score
        
        # Return the type with highest score, or 'explanation' as default
        if max(type_scores.values()) > 0:
            return max(type_scores, key=type_scores.get)
        else:
            return 'explanation'
    
    def _extract_section_title(self, content: str, metadata: Dict) -> str:
        """Extract section title from content or metadata"""
        # First try metadata
        for header_key in ['Header 1', 'Header 2', 'Header 3']:
            if header_key in metadata:
                return metadata[header_key]
        
        # Try to extract from content
        lines = content.split('\n')
        for line in lines[:3]:  # Check first 3 lines
            # Look for markdown headers
            header_match = re.match(r'^#{1,6}\s+(.+)', line.strip())
            if header_match:
                return header_match.group(1).strip()
            
            # Look for title-like content
            if len(line.strip()) > 0 and len(line.strip()) < 100:
                return line.strip()
        
        return "Untitled Section"
    
    def _generate_chunk_id(self, source_path: str, section_idx: int, chunk_idx: int, content: str) -> str:
        """Generate unique chunk ID"""
        # Create a hash of the content for uniqueness
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{source_path}:s{section_idx}:c{chunk_idx}:{content_hash}"

class VectorStoreManager:
    """
    Manages creation and querying of vector stores
    """
    
    def __init__(self, 
                 embeddings: Optional[OpenAIEmbeddings] = None,
                 persist_directory: Optional[str] = None,
                 collection_name: Optional[str] = None):
        """
        Initialize vector store manager
        
        Args:
            embeddings: Embedding model to use
            persist_directory: Directory to persist vector store
            collection_name: Name for the collection
        """
        self.embeddings = embeddings or OpenAIEmbeddings(model="text-embedding-3-small")
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.vector_store = None
    
    def create_vector_store(self, processed_docs: List[ProcessedDocument]) -> None:
        """Create vector store from processed documents"""
        # Convert to LangChain Document format
        langchain_docs = []
        for doc in processed_docs:
            metadata = {
                'chunk_id': doc.chunk_id,
                'source_url': doc.source_url,
                'content_type': doc.content_type,
                'section_title': doc.section_title,
                'file_path': doc.file_path or '',
                **doc.metadata
            }
            
            langchain_doc = Document(
                page_content=doc.content,
                metadata=metadata
            )
            # Filter out complex metadata (dicts, lists) manually
            filtered_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    filtered_metadata[key] = value
            
            filtered_doc = Document(
                page_content=doc.content,
                metadata=filtered_metadata
            )
            langchain_docs.append(filtered_doc)
        
        # Create Chroma vector store
        chroma_kwargs = {
            'documents': langchain_docs,
            'embedding': self.embeddings,
            'persist_directory': self.persist_directory
        }
        if self.collection_name:
            chroma_kwargs['collection_name'] = self.collection_name
            
        self.vector_store = Chroma.from_documents(**chroma_kwargs)
        # Note: Chroma 0.4.x automatically persists documents when persist_directory is provided
    
    def similarity_search(self, 
                         query: str, 
                         k: int = 5,
                         filter_dict: Optional[Dict] = None) -> List[ProcessedDocument]:
        """
        Search vector store for relevant documents
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of relevant ProcessedDocument objects
        """
        if not self.vector_store:
            raise ValueError("Vector store not created yet")
        
        # Search with optional filtering
        results = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter_dict
        )
        
        # Convert back to ProcessedDocument format
        processed_results = []
        for doc in results:
            metadata = doc.metadata.copy()
            
            # Extract core fields from metadata
            chunk_id = metadata.pop('chunk_id', '')
            source_url = metadata.pop('source_url', '')
            content_type = metadata.pop('content_type', '')
            section_title = metadata.pop('section_title', '')
            file_path = metadata.pop('file_path', None)
            
            processed_doc = ProcessedDocument(
                chunk_id=chunk_id,
                source_url=source_url,
                content=doc.page_content,
                content_type=content_type,
                section_title=section_title,
                file_path=file_path if file_path else None,
                metadata=metadata
            )
            processed_results.append(processed_doc)
        
        return processed_results

# Complete pipeline function
def create_documentation_vector_store(documentation_source: str, 
                                    source_type: Optional[str] = None,
                                    persist_directory: Optional[str] = "./chroma_db",
                                    collection_name: Optional[str] = None,
                                    **fetcher_kwargs) -> Tuple[VectorStoreManager, VectorStoreMetadata]:
    """
    Complete pipeline to create vector store from documentation source
    
    Args:
        documentation_source: URL or path to documentation
        source_type: Type of source (auto-detected if None)
        persist_directory: Directory to persist vector store (default: ./chroma_db)
        collection_name: Name for the collection (auto-generated if None)
        **fetcher_kwargs: Additional arguments for fetcher
        
    Returns:
        Tuple of (VectorStoreManager, VectorStoreMetadata)
    """
    from fetcher import ContentFetcherFactory
    
    # Step 1: Fetch raw content
    print("Step 1: Fetching documentation...")
    raw_content = ContentFetcherFactory.fetch_documentation(
        documentation_source, 
        **fetcher_kwargs
    )
    
    # Step 2: Process into structured documents
    print("Step 2: Processing documents...")
    processor = DocumentProcessor()
    processed_docs, metadata = processor.process_raw_content(raw_content)
    
    print(f"Processed {len(processed_docs)} document chunks")
    print(f"Content type distribution: {metadata.content_type_distribution}")
    
    # Generate default collection name if not provided
    if not collection_name:
        if raw_content.source_type == "github":
            # Extract repo name from URL (e.g., "https://github.com/tiangolo/fastapi" -> "tiangolo-fastapi")
            repo_path = documentation_source.rstrip('/').split('/')[-2:]
            collection_name = f"{repo_path[0]}-{repo_path[1]}" if len(repo_path) >= 2 else "github-docs"
        else:
            # For other sources, use a simple name
            collection_name = "documentation"
    
    # Step 3: Create vector store
    print(f"Step 3: Creating vector store with collection '{collection_name}'...")
    vector_manager = VectorStoreManager(
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    vector_manager.create_vector_store(processed_docs)
    
    print("Vector store created successfully!")
    return vector_manager, metadata

# Example usage
if __name__ == "__main__":
    # Example: Create vector store from FastAPI GitHub repo
    try:
        vector_manager, metadata = create_documentation_vector_store(
            "https://github.com/tiangolo/fastapi",
            github_token=None,  # Add token for private repos
            include_code_files=False  # Set to True to include code examples
        )
        
        # Test similarity search
        results = vector_manager.similarity_search(
            "how to create a FastAPI application",
            k=3
        )
        
        print(f"\nFound {len(results)} relevant chunks:")
        for result in results:
            print(f"- {result.section_title} ({result.content_type})")
            print(f"  {result.content[:100]}...")
            
    except Exception as e:
        print(f"Error: {e}")