from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import json
import asyncio
from typing import Dict, List, Optional, Union
from datetime import datetime
from urllib.parse import urljoin, urlparse
import json
import asyncio
from pathlib import Path
import re
import subprocess
import tempfile
from utils import normalize_url
from models import RawContent, Crawl4AIConfig

try:
    from crawl4ai import AsyncWebCrawler
    from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig, CacheMode
    from crawl4ai.deep_crawling import BFSDeepCrawlStrategy, DFSDeepCrawlStrategy, BestFirstCrawlingStrategy
    from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter, DomainFilter, ContentTypeFilter
    from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
    from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False




class ContentFetcher(ABC):
    """Abstract base class for content fetchers"""
    
    @abstractmethod
    def fetch(self, source: str) -> RawContent:
        """Fetch content from the given source"""
        pass

class WebsiteFetcher(ContentFetcher):
    """Fetcher for website documentation with intelligent crawling"""
    
    def __init__(self, max_pages: int = 100, delay: float = 1.0):
        self.max_pages = max_pages
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; DocumentIngestion/1.0; +https://github.com/naptha/devagent)'
        })
    
    def fetch(self, source: str) -> RawContent:
        """Crawl website documentation starting from base URL"""
        base_url = source  # This is our base URL prefix
        content = {}
        metadata = {}
        
        visited_urls = set()
        to_visit = [base_url]
        pages_content = {}  # Dict with URL as key and HTML content as value
        
        print(f"Starting crawl from base URL: {base_url}")
        
        while to_visit and len(visited_urls) < self.max_pages:
            current_url = to_visit.pop(0)
            if current_url in visited_urls:
                continue
            
            # Only process URLs that start with our base URL
            if not current_url.startswith(base_url):
                continue
                
            print(f"Crawling: {current_url}")
            
            try:
                response = self.session.get(current_url, timeout=10)
                response.raise_for_status()
                
                # Store the HTML content directly
                html_content = response.text
                pages_content[current_url] = html_content
                visited_urls.add(current_url)
                
                # Parse HTML to find links
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find all links on this page
                for a in soup.find_all('a', href=True):
                    href = a.get('href')
                    if href:
                        # Convert relative URLs to absolute URLs
                        absolute_url = urljoin(current_url, href)
                        
                        # Remove hash fragments (everything after #)
                        if '#' in absolute_url:
                            absolute_url = absolute_url.split('#')[0]
                        
                        # Skip if URL is empty after removing hash
                        if not absolute_url:
                            continue
                        
                        # Only add if it starts with our base URL and hasn't been visited
                        if (absolute_url.startswith(base_url) and 
                            absolute_url not in visited_urls and 
                            absolute_url not in to_visit):
                            to_visit.append(absolute_url)
                            print(f"  Found link: {absolute_url}")
                
                time.sleep(self.delay)  # Be respectful to the server
                
            except Exception as e:
                error_msg = str(e)
                metadata[f'error_{current_url}'] = error_msg
                print(f"Error fetching {current_url}: {error_msg}")
        
        # Store the pages content as JSON string
        content['pages'] = json.dumps(pages_content, indent=2)
        content['page_count'] = str(len(pages_content))
        
        # Create a summary of all HTML content
        all_urls = list(pages_content.keys())
        content['all_urls'] = json.dumps(all_urls)
        
        metadata['base_url'] = base_url
        metadata['crawled_pages'] = str(len(visited_urls))
        metadata['failed_pages'] = str(len([k for k in metadata.keys() if k.startswith('error_')]))
        metadata['total_links_found'] = str(len(to_visit) + len(visited_urls))
        
        print(f"Crawl completed: {len(visited_urls)} pages crawled")
        
        return RawContent(
            source_type="website",
            source_url=source,
            content=content,
            metadata=metadata,
            fetch_timestamp=datetime.now().isoformat()
        )
    
class Crawl4AIFetcher(ContentFetcher):
    """Advanced website fetcher using Crawl4AI for markdown content
    
    This fetcher provides clean markdown content extraction with various crawling strategies.
    Metadata is automatically serialized to JSON strings to match RawContent schema requirements.
    """
    
    def __init__(self, config: Optional[Crawl4AIConfig] = None):
        if not CRAWL4AI_AVAILABLE:
            raise ImportError(
                "crawl4ai is not installed. Install it with: pip install crawl4ai"
            )
        
        self.config = config or Crawl4AIConfig()
        self.browser_config = BrowserConfig(
            verbose=self.config.verbose,
            headless=True
        )
        
    def fetch(self, source: str) -> RawContent:
        """Fetch markdown content from website using Crawl4AI"""
        import threading
        import asyncio
        
        result = None
        exception = None
        
        def run_async_in_thread():
            nonlocal result, exception
            try:
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(self._async_fetch(source))
                finally:
                    loop.close()
            except Exception as e:
                exception = e
        
        # Always run in a separate thread to avoid event loop conflicts
        thread = threading.Thread(target=run_async_in_thread)
        thread.start()
        thread.join()
        
        if exception:
            raise exception
        return result
    
    async def _async_fetch(self, source: str) -> RawContent:
        """Async implementation of fetch"""
        base_url = normalize_url(source)  # Normalize the base URL
        content = {}
        metadata = {}
        
        async with AsyncWebCrawler(config=self.browser_config) as crawler:
            if self.config.crawl_strategy == "simple":
                # Simple single-page crawling
                result = await self._simple_crawl(crawler, base_url)
                if result:
                    normalized_url = normalize_url(base_url)
                    content[normalized_url] = result.get("markdown", "")
                    # For simple crawl, store metadata under the normalized URL key
                    if result.get("metadata"):
                        metadata[normalized_url] = result["metadata"]
            else:
                # Deep crawling
                results = await self._deep_crawl(crawler, base_url)
                
                # Deduplicate results based on normalized URLs
                seen_urls = set()
                for result in results:
                    if result and result.get("url"):
                        normalized_url = normalize_url(result["url"])
                        
                        # Skip if we've already processed this normalized URL
                        if normalized_url in seen_urls:
                            print(f"🔄 Skipping duplicate URL: {result['url']} -> {normalized_url}")
                            continue
                            
                        seen_urls.add(normalized_url)
                        content[normalized_url] = result.get("markdown", "")
                        if result.get("metadata"):
                            metadata[normalized_url] = result["metadata"]
        
        # Serialize metadata dictionaries to JSON strings to match RawContent schema
        serialized_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, dict):
                serialized_metadata[key] = json.dumps(value)
            else:
                serialized_metadata[key] = str(value)
        
        return RawContent(
            source_type="website",
            source_url=base_url,  # Already normalized above
            content=content,
            metadata=serialized_metadata,
            fetch_timestamp=datetime.now().isoformat()
        )
    
    async def _simple_crawl(self, crawler: AsyncWebCrawler, url: str) -> Optional[Dict]:
        """Simple single-page crawling"""
        run_config = CrawlerRunConfig(
            cache_mode=CacheMode.ENABLED if self.config.cache_enabled else CacheMode.DISABLED,
            word_count_threshold=self.config.word_count_threshold,
            remove_overlay_elements=self.config.remove_overlay_elements,
            process_iframes=True,
            exclude_external_links=True,
            scraping_strategy=LXMLWebScrapingStrategy()
        )
        
        try:
            result = await crawler.arun(url=url, config=run_config)
            if result.success:
                return {
                    "url": url,
                    "markdown": result.markdown.fit_markdown if result.markdown else "",
                    "metadata": {
                        "status_code": result.status_code,
                        "title": result.metadata.get("title", ""),
                        "description": result.metadata.get("description", ""),
                        "word_count": len(result.markdown.fit_markdown.split()) if result.markdown else 0,
                        "links_found": len(result.links.get("internal", [])) if result.links else 0
                    }
                }
            else:
                print(f"Failed to crawl {url}: {result.error_message}")
                return None
        except Exception as e:
            print(f"Error crawling {url}: {str(e)}")
            return None
    
    async def _deep_crawl(self, crawler: AsyncWebCrawler, base_url: str) -> List[Dict]:
        """Deep crawling with various strategies"""
        results = []
        
        # Create filter chain for same-domain crawling
        parsed_base = urlparse(base_url)
        base_domain = parsed_base.netloc.lower()  # Normalize domain case
        base_path = parsed_base.path.rstrip('/')   # Remove trailing slash for pattern matching
        
        filter_chain = FilterChain([
            DomainFilter(allowed_domains=[base_domain]),
            URLPatternFilter(patterns=[f"{base_path}*"]) if base_path else None,
            ContentTypeFilter(allowed_types=["text/html"])
        ])
        
        # Remove None filters
        filter_chain = FilterChain([f for f in filter_chain.filters if f is not None])
        
        # Create crawling strategy
        strategy = self._create_crawl_strategy(filter_chain)
        
        run_config = CrawlerRunConfig(
            deep_crawl_strategy=strategy,
            scraping_strategy=LXMLWebScrapingStrategy(),
            cache_mode=CacheMode.ENABLED if self.config.cache_enabled else CacheMode.DISABLED,
            word_count_threshold=self.config.word_count_threshold,
            remove_overlay_elements=self.config.remove_overlay_elements,
            process_iframes=True,
            exclude_external_links=not self.config.include_external,
            stream=self.config.stream_results,
            verbose=self.config.verbose
        )
        
        try:
            if self.config.stream_results:
                async for result in await crawler.arun(url=base_url, config=run_config):
                    if result.success:
                        processed_result = self._process_crawl_result(result)
                        if processed_result:
                            results.append(processed_result)
                            print(f"Crawled: {result.url}")
                    else:
                        print(f"Failed to crawl {result.url}: {result.error_message}")
            else:
                crawl_results = await crawler.arun(url=base_url, config=run_config)
                if isinstance(crawl_results, list):
                    for result in crawl_results:
                        if result.success:
                            processed_result = self._process_crawl_result(result)
                            if processed_result:
                                results.append(processed_result)
                                print(f"Crawled: {result.url}")
                        else:
                            print(f"Failed to crawl {result.url}: {result.error_message}")
                else:
                    # Single result
                    if crawl_results.success:
                        processed_result = self._process_crawl_result(crawl_results)
                        if processed_result:
                            results.append(processed_result)
                            print(f"Crawled: {crawl_results.url}")
                    else:
                        print(f"Failed to crawl {crawl_results.url}: {crawl_results.error_message}")
        
        except Exception as e:
            print(f"Error during deep crawl: {str(e)}")
        
        return results
    
    def _create_crawl_strategy(self, filter_chain: FilterChain):
        """Create appropriate crawling strategy based on config"""
        common_params = {
            "max_depth": self.config.max_depth,
            "max_pages": self.config.max_pages,
            "include_external": self.config.include_external,
            "filter_chain": filter_chain
        }
        
        if self.config.crawl_strategy == "bfs":
            return BFSDeepCrawlStrategy(**common_params)
        elif self.config.crawl_strategy == "dfs":
            return DFSDeepCrawlStrategy(**common_params)
        elif self.config.crawl_strategy == "best_first":
            scorer = None
            if self.config.keywords:
                scorer = KeywordRelevanceScorer(
                    keywords=self.config.keywords,
                    weight=0.7
                )
            return BestFirstCrawlingStrategy(
                url_scorer=scorer,
                **common_params
            )
        else:
            # Default to BFS
            return BFSDeepCrawlStrategy(**common_params)
    
    def _process_crawl_result(self, result) -> Optional[Dict]:
        """Process a single crawl result"""
        if not result.success:
            return None
        
        # Clean URL by removing hash fragments
        url = result.url.split('#')[0] if '#' in result.url else result.url
        if not url:
            return None
        
        markdown_content = ""
        if result.markdown:
            # Use fit_markdown for the most relevant content
            markdown_content = result.markdown.fit_markdown or result.markdown.raw_markdown or ""
        
        # Create metadata dictionary - will be serialized to JSON string later
        metadata = {
            "status_code": result.status_code,
            "title": result.metadata.get("title", "") if result.metadata else "",
            "description": result.metadata.get("description", "") if result.metadata else "",
            "word_count": len(markdown_content.split()) if markdown_content else 0,
            "links_found": len(result.links.get("internal", [])) if result.links else 0,
            "depth": result.metadata.get("depth", 0) if result.metadata else 0,
            "score": result.metadata.get("score", 0.0) if result.metadata else 0.0
        }
        
        return {
            "url": url,
            "markdown": markdown_content,
            "metadata": metadata
        }

class AdaptiveCrawl4AIFetcher(ContentFetcher):
    """Adaptive website fetcher using Crawl4AI's adaptive crawling"""
    
    def __init__(self, 
                 query: str,
                 confidence_threshold: float = 0.8,
                 max_pages: int = 30,
                 strategy: str = "statistical",
                 verbose: bool = False):
        if not CRAWL4AI_AVAILABLE:
            raise ImportError(
                "crawl4ai is not installed. Install it with: pip install crawl4ai"
            )
        
        self.query = query
        self.confidence_threshold = confidence_threshold
        self.max_pages = max_pages
        self.strategy = strategy
        self.verbose = verbose
        
        try:
            from crawl4ai import AdaptiveCrawler, AdaptiveConfig
            self.adaptive_config = AdaptiveConfig(
                strategy=strategy,
                confidence_threshold=confidence_threshold,
                max_pages=max_pages
            )
        except ImportError:
            raise ImportError(
                "AdaptiveCrawler is not available in this version of crawl4ai"
            )
    
    def fetch(self, source: str) -> RawContent:
        """Fetch content using adaptive crawling"""
        return asyncio.run(self._async_adaptive_fetch(source))
    
    async def _async_adaptive_fetch(self, source: str) -> RawContent:
        """Async adaptive fetch implementation"""
        from crawl4ai import AdaptiveCrawler
        
        browser_config = BrowserConfig(verbose=self.verbose)
        content = {}
        metadata = {}
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            adaptive = AdaptiveCrawler(crawler, self.adaptive_config)
            
            try:
                result = await adaptive.digest(
                    start_url=source,
                    query=self.query
                )
                
                # Get relevant content
                relevant_pages = adaptive.get_relevant_content(top_k=self.max_pages)
                
                for page in relevant_pages:
                    url = page.get('url', '')
                    if url:
                        content[url] = page.get('content', '')
                        metadata[url] = {
                            'relevance_score': page.get('score', 0.0),
                            'word_count': len(page.get('content', '').split())
                        }
                
                # Add overall crawl metadata
                metadata['_crawl_stats'] = {
                    'confidence_score': result.metrics.get('confidence', 0.0),
                    'pages_crawled': len(relevant_pages),
                    'query': self.query,
                    'strategy': self.strategy
                }
                
                if self.verbose:
                    adaptive.print_stats()
                
            except Exception as e:
                print(f"Error during adaptive crawl: {str(e)}")
                # Fallback to simple crawl
                simple_fetcher = Crawl4AIFetcher(Crawl4AIConfig(crawl_strategy="simple"))
                return await simple_fetcher._async_fetch(source)
        
        # Serialize metadata dictionaries to JSON strings to match RawContent schema
        serialized_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, dict):
                serialized_metadata[key] = json.dumps(value)
            else:
                serialized_metadata[key] = str(value)
        
        return RawContent(
            source_type="website",
            source_url=source,
            content=content,
            metadata=serialized_metadata,
            fetch_timestamp=datetime.now().isoformat()
        )

# Factory function for easy usage
def create_crawl4ai_fetcher(
    crawl_type: str = "simple",
    max_pages: int = 50,
    max_depth: int = 2,
    keywords: List[str] = None,
    query: str = None,
    verbose: bool = False,
    **kwargs
) -> ContentFetcher:
    """
Factory function to create appropriate Crawl4AI fetcher

CRAWLING STRATEGIES & WHEN TO USE EACH:

1. SIMPLE CRAWLING (crawl_type="simple")
   ✅ Best for: Single page content extraction
   ✅ Use when: You need content from one specific page
   ✅ Examples: Landing pages, specific documentation pages, blog posts
   ✅ Advantages: Fast, minimal resource usage, no link following
   ❌ Avoid when: You need content from multiple related pages

2. DEEP CRAWLING - BFS Strategy (crawl_type="deep", default)
   ✅ Best for: Systematic exploration of website sections
   ✅ Use when: You want comprehensive coverage of a documentation site
   ✅ Examples: API documentation, knowledge bases, product catalogs
   ✅ Advantages: Discovers all pages at each level before going deeper
   ❌ Avoid when: You need targeted content or have bandwidth constraints

3. DEEP CRAWLING - DFS Strategy (crawl_type="deep", crawl_strategy="dfs")
   ✅ Best for: Following specific paths deeply
   ✅ Use when: You want to explore one branch completely before others
   ✅ Examples: Tutorial sequences, step-by-step guides, hierarchical docs
   ✅ Advantages: Good for finding deep content quickly
   ❌ Avoid when: You need broad coverage or balanced exploration

4. DEEP CRAWLING - Best First Strategy (crawl_type="deep", crawl_strategy="best_first")
   ✅ Best for: Quality-focused crawling with limited resources
   ✅ Use when: You want the most relevant pages first
   ✅ Examples: Research, competitive analysis, specific topic exploration
   ✅ Advantages: Prioritizes relevant content, efficient resource usage
   ❌ Avoid when: You need comprehensive coverage regardless of relevance

5. ADAPTIVE CRAWLING (crawl_type="adaptive")
   ✅ Best for: Intelligent, query-driven content discovery
   ✅ Use when: You have specific information needs
   ✅ Examples: Research tasks, Q&A preparation, knowledge building
   ✅ Advantages: Stops when sufficient information is found, query-aware
   ❌ Avoid when: You need complete site coverage or structured data

CONFIGURATION RECOMMENDATIONS:

For API Documentation:
- Use: Deep crawling (BFS) with keywords=["api", "endpoint", "reference"]
- Settings: max_depth=3, max_pages=50-100

For Research/Knowledge Building:
- Use: Adaptive crawling with specific query
- Settings: confidence_threshold=0.7-0.8, max_pages=20-30

For Quick Content Extraction:
- Use: Simple crawling for single pages
- Use: Best First with keywords for targeted multi-page

For Complete Site Analysis:
- Use: Deep crawling (BFS) with high max_pages
- Settings: max_depth=4-5, stream_results=True

Args:
    crawl_type: "simple", "deep", or "adaptive"
    max_pages: Maximum number of pages to crawl
    max_depth: Maximum crawl depth (for deep crawling)
    keywords: Keywords for relevance scoring
    query: Query for adaptive crawling
    verbose: Enable verbose logging
    **kwargs: Additional configuration options

Returns:
    ContentFetcher instance
"""
    if crawl_type == "adaptive":
        if not query:
            raise ValueError("Query is required for adaptive crawling")
        return AdaptiveCrawl4AIFetcher(
            query=query,
            max_pages=max_pages,
            verbose=verbose,
            **kwargs
        )
    else:
        strategy = "simple" if crawl_type == "simple" else "bfs"
        config = Crawl4AIConfig(
            max_pages=max_pages,
            max_depth=max_depth,
            crawl_strategy=strategy,
            keywords=keywords or [],
            verbose=verbose,
            **kwargs
        )
        return Crawl4AIFetcher(config)

class GitHubFetcher(ContentFetcher):
    """Enhanced GitHub repository fetcher for markdown documentation"""
    
    def __init__(self, 
                 github_token: Optional[str] = None,
                 include_code_files: bool = False,
                 max_file_size_mb: int = 5,
                 clone_depth: int = 1,
                 include_folders: Optional[List[str]] = None):
        """
        Initialize GitHub fetcher
        
        Args:
            github_token: GitHub personal access token for private repos
            include_code_files: Whether to include code files alongside markdown
            max_file_size_mb: Maximum file size to process (in MB)
            clone_depth: Git clone depth (1 for shallow clone)
            include_folders: List of folder paths to include (e.g., ['docs/en', 'guides']). 
                           If None, includes all folders.
        """
        self.github_token = github_token
        self.include_code_files = include_code_files
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.clone_depth = clone_depth
        self.include_folders = include_folders
        
        # File extensions to process
        self.markdown_extensions = {'.md', '.mdx', '.markdown', '.mdown'}
        self.code_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.go', '.rs', '.rb', '.php'}
    
    def fetch(self, source: str) -> RawContent:
        """
        Fetch documentation from GitHub repository
        
        Args:
            source: GitHub repository URL (e.g., https://github.com/owner/repo)
            
        Returns:
            RawContent with all markdown files and metadata
        """
        # Parse GitHub URL
        repo_info = self._parse_github_url(source)
        if not repo_info:
            raise ValueError(f"Invalid GitHub URL: {source}")
        
        content = {}
        metadata = {}
        
        # Create temporary directory for cloning
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir) / "repo"
            
            try:
                # Clone repository
                clone_success = self._clone_repository(source, repo_path, repo_info)
                if not clone_success:
                    raise RuntimeError(f"Failed to clone repository: {source}")
                
                # Extract repository metadata
                repo_metadata = self._extract_repo_metadata(repo_path, repo_info)
                metadata.update(repo_metadata)
                
                # Find and process markdown files
                markdown_files = self._find_markdown_files(repo_path)
                print(f"Found {len(markdown_files)} markdown files")
                
                # Process each markdown file
                for file_path in markdown_files:
                    try:
                        relative_path = str(file_path.relative_to(repo_path))
                        file_content = self._process_markdown_file(file_path, relative_path)
                        if file_content:
                            content[relative_path] = file_content
                    except Exception as e:
                        metadata[f'error_{relative_path}'] = str(e)
                        print(f"Error processing {relative_path}: {e}")
                
                # Optionally include code files for context
                if self.include_code_files:
                    code_files = self._find_code_files(repo_path)
                    print(f"Found {len(code_files)} code files")
                    
                    for file_path in code_files[:20]:  # Limit to first 20 code files
                        try:
                            relative_path = str(file_path.relative_to(repo_path))
                            file_content = self._process_code_file(file_path, relative_path)
                            if file_content:
                                content[f"code:{relative_path}"] = file_content
                        except Exception as e:
                            metadata[f'error_code_{relative_path}'] = str(e)
                
                # Add processing statistics
                metadata['files_processed'] = str(len(content))
                metadata['markdown_files_found'] = str(len(markdown_files))
                if self.include_code_files:
                    code_files_processed = sum(1 for k in content.keys() if k.startswith('code:'))
                    metadata['code_files_processed'] = str(code_files_processed)
                
            except Exception as e:
                metadata['fetch_error'] = str(e)
                print(f"Error fetching repository: {e}")
                raise
        
        return RawContent(
            source_type="github",
            source_url=source,
            content=content,
            metadata=metadata,
            fetch_timestamp=datetime.now().isoformat()
        )
    
    def _parse_github_url(self, url: str) -> Optional[Dict[str, str]]:
        """Parse GitHub URL to extract owner and repo"""
        try:
            parsed = urlparse(url)
            if 'github.com' not in parsed.netloc:
                return None
            
            path_parts = parsed.path.strip('/').split('/')
            if len(path_parts) < 2:
                return None
            
            owner = path_parts[0]
            repo = path_parts[1]
            
            # Remove .git suffix if present
            if repo.endswith('.git'):
                repo = repo[:-4]
            
            return {
                'owner': owner,
                'repo': repo,
                'full_name': f"{owner}/{repo}",
                'url': url
            }
        except Exception:
            return None
    
    def _clone_repository(self, url: str, target_path: Path, repo_info: Dict[str, str]) -> bool:
        """Clone GitHub repository to target path"""
        try:
            # Prepare clone command
            clone_url = url
            if self.github_token:
                # Use token authentication for private repos
                parsed = urlparse(url)
                clone_url = f"https://{self.github_token}@{parsed.netloc}{parsed.path}"
            
            # Clone with shallow depth for efficiency
            cmd = [
                'git', 'clone',
                '--depth', str(self.clone_depth),
                '--quiet',
                clone_url,
                str(target_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Successfully cloned {repo_info['full_name']}")
                return True
            else:
                print(f"Git clone failed: {result.stderr}")
                return False
                
        except subprocess.SubprocessError as e:
            print(f"Subprocess error during clone: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error during clone: {e}")
            return False
    
    def _extract_repo_metadata(self, repo_path: Path, repo_info: Dict[str, str]) -> Dict[str, str]:
        """Extract metadata about the repository"""
        metadata = {}
        
        # Basic repo info
        metadata['repo_owner'] = repo_info['owner']
        metadata['repo_name'] = repo_info['repo']
        metadata['repo_full_name'] = repo_info['full_name']
        
        try:
            # Get latest commit info
            result = subprocess.run(
                ['git', 'log', '-1', '--format=%H|%ai|%s'],
                cwd=repo_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                commit_info = result.stdout.strip().split('|')
                if len(commit_info) >= 3:
                    metadata['latest_commit_hash'] = commit_info[0]
                    metadata['latest_commit_date'] = commit_info[1]
                    metadata['latest_commit_message'] = commit_info[2]
        except Exception as e:
            metadata['git_info_error'] = str(e)
        
        # Look for common documentation indicators
        readme_files = list(repo_path.glob('README*'))
        if readme_files:
            metadata['has_readme'] = 'true'
            metadata['readme_files'] = json.dumps([f.name for f in readme_files])
        
        # Check for documentation directories
        doc_dirs = []
        for potential_dir in ['docs', 'documentation', 'doc', 'guide', 'guides']:
            doc_path = repo_path / potential_dir
            if doc_path.exists() and doc_path.is_dir():
                doc_dirs.append(potential_dir)
        
        if doc_dirs:
            metadata['documentation_directories'] = json.dumps(doc_dirs)
        
        # Look for package.json, requirements.txt, etc. for project type
        project_files = {
            'package.json': 'javascript',
            'requirements.txt': 'python',
            'Cargo.toml': 'rust',
            'go.mod': 'go',
            'pom.xml': 'java',
            'composer.json': 'php'
        }
        
        detected_types = []
        for file_name, project_type in project_files.items():
            if (repo_path / file_name).exists():
                detected_types.append(project_type)
        
        if detected_types:
            metadata['project_types'] = json.dumps(detected_types)
        
        return metadata
    
    def _find_markdown_files(self, repo_path: Path) -> List[Path]:
        """Find all markdown files in the repository, optionally filtered by include_folders"""
        markdown_files = []
        
        if self.include_folders:
            # Search only in specified folders
            for folder in self.include_folders:
                folder_path = repo_path / folder
                if folder_path.exists() and folder_path.is_dir():
                    print(f"Searching in folder: {folder}")
                    for ext in self.markdown_extensions:
                        markdown_files.extend(folder_path.rglob(f'*{ext}'))
                else:
                    print(f"Warning: Folder '{folder}' not found in repository")
        else:
            # Search entire repository
            for ext in self.markdown_extensions:
                markdown_files.extend(repo_path.rglob(f'*{ext}'))
        
        # Filter by file size and skip hidden directories
        filtered_files = []
        for file_path in markdown_files:
            try:
                # Skip files in hidden directories (starting with .)
                if any(part.startswith('.') for part in file_path.parts):
                    continue
                
                # Check file size
                if file_path.stat().st_size > self.max_file_size_bytes:
                    continue
                
                # Additional check: if include_folders is specified, ensure file is in one of them
                if self.include_folders:
                    relative_path = file_path.relative_to(repo_path)
                    if not any(str(relative_path).startswith(folder) for folder in self.include_folders):
                        continue
                
                filtered_files.append(file_path)
            except Exception:
                continue
        
        # Sort by path for consistent ordering
        return sorted(filtered_files)
    
    def _find_code_files(self, repo_path: Path) -> List[Path]:
        """Find relevant code files for additional context, optionally filtered by include_folders"""
        code_files = []
        
        if self.include_folders:
            # Search only in specified folders
            for folder in self.include_folders:
                folder_path = repo_path / folder
                if folder_path.exists() and folder_path.is_dir():
                    for ext in self.code_extensions:
                        code_files.extend(folder_path.rglob(f'*{ext}'))
        else:
            # Search entire repository
            for ext in self.code_extensions:
                code_files.extend(repo_path.rglob(f'*{ext}'))
        
        # Filter and prioritize
        filtered_files = []
        for file_path in code_files:
            try:
                # Skip files in hidden directories, node_modules, __pycache__, etc.
                skip_dirs = {'.git', 'node_modules', '__pycache__', '.venv', 'venv', 'dist', 'build'}
                if any(part in skip_dirs for part in file_path.parts):
                    continue
                
                # Check file size
                if file_path.stat().st_size > self.max_file_size_bytes:
                    continue
                
                # Additional check: if include_folders is specified, ensure file is in one of them
                if self.include_folders:
                    relative_path = file_path.relative_to(repo_path)
                    if not any(str(relative_path).startswith(folder) for folder in self.include_folders):
                        continue
                
                filtered_files.append(file_path)
            except Exception:
                continue
        
        # Prioritize files that are likely to be examples or main modules
        priority_patterns = [r'example', r'demo', r'sample', r'main', r'index', r'__init__']
        
        def file_priority(file_path: Path) -> int:
            name_lower = file_path.name.lower()
            for i, pattern in enumerate(priority_patterns):
                if re.search(pattern, name_lower):
                    return i
            return len(priority_patterns)
        
        # Sort by priority, then by path
        filtered_files.sort(key=lambda f: (file_priority(f), str(f)))
        
        return filtered_files
    
    def _process_markdown_file(self, file_path: Path, relative_path: str) -> Optional[str]:
        """Process a single markdown file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Basic cleanup - remove excessive whitespace
            content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
            
            # Add file path as metadata comment
            processed_content = f"<!-- File: {relative_path} -->\n\n{content}"
            
            return processed_content
            
        except Exception as e:
            print(f"Error reading {relative_path}: {e}")
            return None
    
    def _process_code_file(self, file_path: Path, relative_path: str) -> Optional[str]:
        """Process a single code file for context"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Add code block formatting for markdown compatibility
            file_ext = file_path.suffix.lstrip('.')
            processed_content = f"<!-- Code File: {relative_path} -->\n\n```{file_ext}\n{content}\n```"
            
            return processed_content
            
        except Exception as e:
            print(f"Error reading code file {relative_path}: {e}")
            return None

class ContentFetcherFactory:
    """Factory for creating appropriate content fetchers"""
    
    @staticmethod
    def create_fetcher(source_type: str, **kwargs) -> ContentFetcher:
        """Create appropriate fetcher based on source type"""
        if source_type == "website":
            # Use advanced Crawl4AI fetcher if available, fallback to basic
            try:
                crawl_type = kwargs.get('crawl_type', 'deep')
                max_pages = kwargs.get('max_pages', 50)
                max_depth = kwargs.get('max_depth', 2)
                keywords = kwargs.get('keywords', [])
                
                return create_crawl4ai_fetcher(
                    crawl_type=crawl_type,
                    max_pages=max_pages,
                    max_depth=max_depth,
                    keywords=keywords,
                    **kwargs
                )
            except ImportError:
                print("Crawl4AI not available, falling back to basic web fetcher")
                from .fetcher import WebsiteFetcher
                return WebsiteFetcher(**kwargs)
        
        elif source_type == "github":
            return GitHubFetcher(**kwargs)
        
        elif source_type == "openapi":
            # TODO: Implement OpenAPI fetcher
            raise NotImplementedError("OpenAPI fetching is not implemented yet")
        
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
    
    @staticmethod
    def detect_source_type(source: str) -> str:
        """Auto-detect source type from URL or content"""
        if 'github.com' in source:
            return 'github'
        elif source.endswith(('.json', '.yaml', '.yml')) or 'openapi' in source or 'swagger' in source:
            return 'openapi'
        else:
            return 'website'

    @staticmethod
    def fetch_documentation(source: str, **kwargs) -> RawContent:
        """
        High-level function to fetch documentation from any source
        
        Args:
            source: URL or path to documentation source
            **kwargs: Additional configuration options
            
        Returns:
            RawContent with fetched documentation
        """
        source_type = ContentFetcherFactory.detect_source_type(source)
        fetcher = ContentFetcherFactory.create_fetcher(source_type, **kwargs)
        
        print(f"Fetching {source_type} documentation from: {source}")
        return fetcher.fetch(source)
    
# Convenience functions for common use cases
def fetch_github_docs(repo_url: str, 
                     github_token: Optional[str] = None,
                     include_code: bool = False,
                     include_folders: Optional[List[str]] = None) -> RawContent:
    """
    Fetch documentation from GitHub repository
    
    Args:
        repo_url: GitHub repository URL
        github_token: GitHub personal access token for private repos
        include_code: Whether to include code files alongside markdown
        include_folders: List of folder paths to include (e.g., ['docs/en', 'guides']). 
                        If None, includes all folders.
    """
    return ContentFetcherFactory.fetch_documentation(
        repo_url,
        github_token=github_token,
        include_code_files=include_code,
        include_folders=include_folders
    )

def fetch_website_docs(website_url: str,
                      crawl_type: str = "deep",
                      max_pages: int = 50,
                      keywords: List[str] = None) -> RawContent:
    """Fetch documentation from website"""
    return ContentFetcherFactory.fetch_documentation(
        website_url,
        crawl_type=crawl_type,
        max_pages=max_pages,
        keywords=keywords or []
    )

if __name__ == "__main__":
    import pickle
    # Test the fetcher
    result = fetch_github_docs("https://github.com/modelcontextprotocol/docs", include_code=True)

    # get the raw content and print the word count for each file
    for file_path, content in result.content.items():
        print(file_path)
        print(f"Word count: {len(content.split())}")

    # save the result to a file as pickle
    with open("github_docs.pkl", "wb") as f:
        pickle.dump(result, f)