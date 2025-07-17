import hashlib
import tiktoken
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from openai import OpenAI

from models import RawContent, ProcessedDocument


class DocumentProcessor:
    """Simple document processor with token-based chunking and contextual summaries"""
    
    def __init__(self, 
                 chunk_size_tokens: int = 5000,
                 chunk_overlap_tokens: int = 500,
                 summary_length_words: int = 20000,
                 model_name: str = "gpt-4",
                 summarization_model: str = "gpt-4o",
                 max_workers: int = 4):
        """
        Initialize document processor
        
        Args:
            chunk_size_tokens: Maximum tokens per chunk
            chunk_overlap_tokens: Token overlap between chunks  
            summary_length_words: Words to use for creating initial content for summarization
            model_name: Model name for tokenizer
            summarization_model: Model to use for LLM summarization
            max_workers: Maximum number of threads for parallel summarization
        """
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.summary_length_words = summary_length_words
        self.summarization_model = summarization_model
        self.max_workers = max_workers
        
        # Initialize OpenAI client for summarization
        self.openai_client = OpenAI()
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fallback to a common encoding
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def process_raw_content(self, raw_content: RawContent) -> List[ProcessedDocument]:
        """
        Process raw content into token-based chunks with contextual summaries (parallel processing)
        
        Args:
            raw_content: Raw content from fetchers
            
        Returns:
            List of processed documents
        """
        # Filter out metadata entries
        content_items = [
            (file_path, content) for file_path, content in raw_content.content.items()
            if file_path not in ['pages', 'page_count', 'all_urls']
        ]
        
        if not content_items:
            return []
        
        print(f"Generating summaries for {len(content_items)} files in parallel (max_workers={self.max_workers})...")
        
        # Generate summaries in parallel
        file_summaries = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all summarization tasks
            future_to_file = {
                executor.submit(self._create_page_summary, content): file_path
                for file_path, content in content_items
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    summary = future.result()
                    file_summaries[file_path] = summary
                    print(f"✓ Generated summary for {file_path}")
                except Exception as e:
                    print(f"✗ Failed to generate summary for {file_path}: {e}")
                    file_summaries[file_path] = f"CONTEXT (Summary generation failed): {str(e)}"
        
        print("Summary generation complete. Processing chunks...")
        
        # Now process all files with their summaries
        processed_docs = []
        total_files = len(content_items)
        
        for file_idx, (file_path, content) in enumerate(content_items, 1):
            print(f"Processing file {file_idx}/{total_files}: {file_path}")
            
            page_summary = file_summaries.get(file_path, "CONTEXT: No summary available")
            
            try:
                # Split content into token-based chunks
                print(f"  Splitting into chunks (content length: {len(content)} chars)...")
                chunks = self._split_into_token_chunks(content)
                print(f"  Created {len(chunks)} chunks")
                
                # Create processed documents with contextual summaries
                for i, chunk_text in enumerate(chunks):
                    # Prepend page summary to each chunk
                    contextualized_content = f"{page_summary}\n\n---\n\n{chunk_text}"
                    
                    # Check if contextualized content is too long for embeddings (limit ~8000 tokens)
                    try:
                        total_tokens = len(self.tokenizer.encode(contextualized_content))
                        if total_tokens > 8000:
                            print(f"    Warning: Chunk {i} too long ({total_tokens} tokens), truncating...")
                            # Truncate the chunk text to fit within limits
                            max_chunk_tokens = 8000 - len(self.tokenizer.encode(page_summary)) - 50  # Leave buffer
                            if max_chunk_tokens > 0:
                                chunk_tokens = self.tokenizer.encode(chunk_text)[:max_chunk_tokens]
                                chunk_text = self.tokenizer.decode(chunk_tokens)
                                contextualized_content = f"{page_summary}\n\n---\n\n{chunk_text}"
                            else:
                                print(f"    Warning: Page summary too long, skipping chunk {i}")
                                continue
                    except Exception as e:
                        print(f"    Warning: Token length check failed for chunk {i}: {e}")
                    
                    chunk_id = self._generate_chunk_id(file_path, i, chunk_text)
                    
                    # Use try-catch for tokenization which might be slow/problematic
                    try:
                        original_tokens = len(self.tokenizer.encode(chunk_text))
                        total_tokens = len(self.tokenizer.encode(contextualized_content))
                    except Exception as e:
                        print(f"    Warning: Token counting failed for chunk {i}: {e}")
                        original_tokens = len(chunk_text) // 4  # Rough estimate
                        total_tokens = len(contextualized_content) // 4
                    
                    processed_doc = ProcessedDocument(
                        chunk_id=chunk_id,
                        source_url=raw_content.source_url,
                        content=contextualized_content,
                        content_type="contextualized_chunk",
                        section_title=f"{file_path} - Chunk {i+1}",
                        file_path=file_path,
                        metadata={
                            'chunk_index': i,
                            'original_chunk_tokens': original_tokens,
                            'total_tokens_with_context': total_tokens,
                            'has_page_summary': True,
                            'page_summary': page_summary,  # Store the full page summary
                            'page_summary_words': len(page_summary.split())
                        }
                    )
                    processed_docs.append(processed_doc)
                
                print(f"  ✓ Completed {file_path} ({len(chunks)} chunks)")
                
            except Exception as e:
                print(f"  ✗ Error processing {file_path}: {e}")
                continue
        
        return processed_docs
    
    def _create_page_summary(self, content: str) -> str:
        """
        Create an LLM-generated summary from page content
        
        Args:
            content: Full page content
            
        Returns:
            LLM-generated summary text to prepend to chunks
        """
        words = content.split()
        
        # Take first 20k words for summarization (to fit within context limits)
        summary_words = words[:self.summary_length_words]
        content_for_summary = ' '.join(summary_words)
        
        # Check token count and truncate if needed (leave room for prompt)
        tokens = self.tokenizer.encode(content_for_summary)
        max_content_tokens = 100000  # Leave room for prompt
        
        if len(tokens) > max_content_tokens:
            truncated_tokens = tokens[:max_content_tokens]
            content_for_summary = self.tokenizer.decode(truncated_tokens)
        
        try:
            # Create summarization prompt
            prompt = f"""Please create a comprehensive summary of the following documentation content. 
The summary should capture the main topics, key concepts, and important details that would help understand chunks from this document.
Focus on technical concepts, APIs, features, and how things work. Keep the summary detailed but concise.

Content to summarize:
{content_for_summary}

Summary:"""

            response = self.openai_client.chat.completions.create(
                model=self.summarization_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.,
                max_tokens=2000
            )
            
            llm_summary = response.choices[0].message.content.strip()
            
            # Add metadata about the summarization
            word_count = len(summary_words)
            total_words = len(words)
            
            if word_count < total_words:
                context_header = f"CONTEXT SUMMARY (Based on first {word_count:,} of {total_words:,} words):\n"
            else:
                context_header = f"CONTEXT SUMMARY (Complete document - {word_count:,} words):\n"
            
            return context_header + llm_summary
            
        except Exception as e:
            print(f"Warning: Failed to generate LLM summary, falling back to truncated content: {e}")
            # Fallback to truncated content if LLM summarization fails
            fallback_words = words[:1000]  # Much shorter fallback
            fallback_text = ' '.join(fallback_words)
            return f"CONTEXT (First 1,000 words - LLM summarization failed):\n{fallback_text}"
    
    def _split_into_token_chunks(self, text: str) -> List[str]:
        """
        Split text into chunks based on token count
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        # Safety check for extremely long content
        if len(text) > 5_000_000:  # 5MB limit
            print(f"    Warning: Content very long ({len(text)} chars), truncating...")
            text = text[:5_000_000]
        
        try:
            # Encode the full text
            tokens = self.tokenizer.encode(text)
        except Exception as e:
            print(f"    Error encoding text: {e}")
            # Fallback to character-based chunking
            return self._fallback_character_chunks(text)
        
        # Safety check for token count
        if len(tokens) > 500_000:  # Very large token limit
            print(f"    Warning: Too many tokens ({len(tokens)}), truncating...")
            tokens = tokens[:500_000]
        
        chunks = []
        start_idx = 0
        max_iterations = 1000  # Prevent infinite loops
        iteration = 0
        
        while start_idx < len(tokens) and iteration < max_iterations:
            # Calculate end index for this chunk
            end_idx = min(start_idx + self.chunk_size_tokens, len(tokens))
            
            # Extract chunk tokens
            chunk_tokens = tokens[start_idx:end_idx]
            
            try:
                # Decode back to text
                chunk_text = self.tokenizer.decode(chunk_tokens)
                chunks.append(chunk_text)
            except Exception as e:
                print(f"    Warning: Failed to decode chunk {iteration}: {e}")
                # Skip this chunk
                pass
            
            # Move start index forward, accounting for overlap
            next_start = end_idx - self.chunk_overlap_tokens
            
            # Prevent infinite loop - ensure we're making progress
            if next_start <= start_idx or next_start >= len(tokens):
                break
            
            start_idx = next_start
            
            iteration += 1
        
        if iteration >= max_iterations:
            print(f"    Warning: Hit max iterations limit during chunking")
        
        return chunks
    
    def _fallback_character_chunks(self, text: str) -> List[str]:
        """Fallback character-based chunking when tokenization fails"""
        print("    Using fallback character-based chunking...")
        chunk_size_chars = self.chunk_size_tokens * 4  # Rough estimate
        overlap_chars = self.chunk_overlap_tokens * 4
        
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size_chars, len(text))
            chunks.append(text[start:end])
            start = end - overlap_chars
            if start <= 0 or start >= end:
                break
        
        return chunks
    
    def _generate_chunk_id(self, file_path: str, chunk_index: int, content: str) -> str:
        """Generate unique chunk ID"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{file_path}:chunk_{chunk_index}:{content_hash}"


# Simple convenience function
def process_documentation(raw_content: RawContent, 
                         chunk_size_tokens: int = 5000,
                         summary_length_words: int = 20000,
                         summarization_model: str = "gpt-4o",
                         max_workers: int = 4) -> List[ProcessedDocument]:
    """
    Process documentation with contextual chunking and parallel LLM summarization
    
    Args:
        raw_content: Raw content from fetchers
        chunk_size_tokens: Maximum tokens per chunk
        summary_length_words: Words to use for creating content for LLM summarization
        summarization_model: Model to use for LLM summarization
        max_workers: Maximum number of threads for parallel summarization
        
    Returns:
        List of processed documents with LLM-generated contextual summaries
    """
    processor = DocumentProcessor(
        chunk_size_tokens=chunk_size_tokens,
        summary_length_words=summary_length_words,
        summarization_model=summarization_model,
        max_workers=max_workers
    )
    return processor.process_raw_content(raw_content)


class VectorStoreManager:
    """Manages creation and querying of vector stores"""
    
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
    
    def create_vector_store(self, processed_docs: List[ProcessedDocument], batch_size: int = 100) -> None:
        """
        Create vector store from processed documents with batching to avoid token limits
        
        Args:
            processed_docs: List of processed documents
            batch_size: Number of documents to process per batch (default: 100)
        """
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
            
            # Filter out complex metadata (keep only simple types)
            filtered_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    filtered_metadata[key] = value
            
            langchain_doc = Document(
                page_content=doc.content,
                metadata=filtered_metadata
            )
            langchain_docs.append(langchain_doc)
        
        print(f"Creating vector store with {len(langchain_docs)} documents in batches of {batch_size}...")
        
        # Process documents in batches to stay within embedding API limits
        for i in range(0, len(langchain_docs), batch_size):
            batch_docs = langchain_docs[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            if i == 0:
                # Create initial vector store with first batch
                chroma_kwargs = {
                    'documents': batch_docs,
                    'embedding': self.embeddings,
                    'persist_directory': self.persist_directory
                }
                if self.collection_name:
                    chroma_kwargs['collection_name'] = self.collection_name
                self.vector_store = Chroma.from_documents(**chroma_kwargs)
            else:
                # Add to existing vector store
                self.vector_store.add_documents(batch_docs)
            
            print(f"  ✓ Batch {batch_num}: Processed {len(batch_docs)} documents")
        
        print(f"Vector store creation complete! Total documents: {len(langchain_docs)}")
    
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
                                    persist_directory: Optional[str] = "./chroma_db",
                                    collection_name: Optional[str] = None,
                                    chunk_size_tokens: int = 5000,
                                    summary_length_words: int = 20000,
                                    summarization_model: str = "gpt-4o",
                                    max_workers: int = 4,
                                    **fetcher_kwargs) -> VectorStoreManager:
    """
    Complete pipeline to create vector store from documentation source
    
    Args:
        documentation_source: URL or path to documentation
        persist_directory: Directory to persist vector store
        collection_name: Name for the collection
        chunk_size_tokens: Maximum tokens per chunk
        summary_length_words: Words to use for creating content for LLM summarization
        summarization_model: Model to use for LLM summarization
        max_workers: Maximum number of threads for parallel summarization
        **fetcher_kwargs: Additional arguments for fetcher
        
    Returns:
        VectorStoreManager instance
    """
    from fetcher import ContentFetcherFactory
    
    # Step 1: Fetch raw content
    print("Step 1: Fetching documentation...")
    raw_content = ContentFetcherFactory.fetch_documentation(
        documentation_source, 
        **fetcher_kwargs
    )
    
    # Step 2: Process into structured documents with parallel contextual chunking
    print("Step 2: Processing documents with parallel contextual chunking...")
    processed_docs = process_documentation(
        raw_content, 
        chunk_size_tokens=chunk_size_tokens,
        summary_length_words=summary_length_words,
        summarization_model=summarization_model,
        max_workers=max_workers
    )
    
    print(f"Processed {len(processed_docs)} contextualized chunks")
    
    # Generate default collection name if not provided
    if not collection_name:
        if 'github.com' in documentation_source:
            # Extract repo name from URL
            repo_path = documentation_source.rstrip('/').split('/')[-2:]
            collection_name = f"{repo_path[0]}-{repo_path[1]}" if len(repo_path) >= 2 else "github-docs"
        else:
            collection_name = "documentation"
    
    # Step 3: Create vector store
    print(f"Step 3: Creating vector store with collection '{collection_name}'...")
    vector_manager = VectorStoreManager(
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    vector_manager.create_vector_store(processed_docs)
    
    print("Vector store created successfully!")
    return vector_manager


if __name__ == "__main__":
    # Example usage
    from fetcher import fetch_github_docs
    
    try:
        # Create vector store with parallel contextual chunking
        # vector_manager = create_documentation_vector_store(
        #     "https://github.com/modelcontextprotocol/docs",
        #     chunk_size_tokens=5000,
        #     summary_length_words=20000,
        #     summarization_model="gpt-4o",
        #     max_workers=10
        # )
        # Example: Process only the English docs folder from FastAPI
        vector_manager = create_documentation_vector_store(
            "https://github.com/fastapi/fastapi",
            chunk_size_tokens=5000,
            summary_length_words=20000,
            summarization_model="gpt-4o",
            max_workers=10,
            include_folders=["docs/en"]  # Only include the English documentation folder
        )
        # Test similarity search
        results = vector_manager.similarity_search(
            "how to create a server",
            k=3
        )
        
        print(f"\nFound {len(results)} relevant chunks:")
        for result in results:
            print(f"- {result.section_title}")
            print(f"  Tokens: {result.metadata.get('total_tokens_with_context', 'unknown')}")
            print(f"  Preview: {result.content[:200]}...")
            print()
            
    except Exception as e:
        print(f"Error: {e}")