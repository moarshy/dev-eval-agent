import asyncio
from typing import Dict, List, Optional, Union
from datetime import datetime
from urllib.parse import urljoin, urlparse
from pydantic import BaseModel, Field
import json

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

from .fetcher import RawContent, ContentFetcher

class Crawl4AIConfig(BaseModel):
    """Configuration for Crawl4AI-based fetching"""
    max_pages: int = Field(default=50, description="Maximum number of pages to crawl")
    max_depth: int = Field(default=2, description="Maximum crawl depth")
    delay: float = Field(default=1.0, description="Delay between requests in seconds")
    include_external: bool = Field(default=False, description="Whether to include external links")
    crawl_strategy: str = Field(default="bfs", description="Crawling strategy: bfs, dfs, best_first, or simple")
    stream_results: bool = Field(default=False, description="Whether to stream results as they come")
    keywords: List[str] = Field(default_factory=list, description="Keywords for relevance scoring (best_first only)")
    verbose: bool = Field(default=False, description="Enable verbose logging")
    cache_enabled: bool = Field(default=True, description="Enable caching")
    remove_overlay_elements: bool = Field(default=True, description="Remove popups and overlays")
    word_count_threshold: int = Field(default=10, description="Minimum words per content block")
    
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
        return asyncio.run(self._async_fetch(source))
    
    async def _async_fetch(self, source: str) -> RawContent:
        """Async implementation of fetch"""
        base_url = source
        content = {}
        metadata = {}
        
        async with AsyncWebCrawler(config=self.browser_config) as crawler:
            if self.config.crawl_strategy == "simple":
                # Simple single-page crawling
                result = await self._simple_crawl(crawler, base_url)
                if result:
                    content[base_url] = result.get("markdown", "")
                    # For simple crawl, store metadata under the URL key
                    if result.get("metadata"):
                        metadata[base_url] = result["metadata"]
            else:
                # Deep crawling
                results = await self._deep_crawl(crawler, base_url)
                for result in results:
                    if result and result.get("url"):
                        content[result["url"]] = result.get("markdown", "")
                        if result.get("metadata"):
                            metadata[result["url"]] = result["metadata"]
        
        # Serialize metadata dictionaries to JSON strings to match RawContent schema
        serialized_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, dict):
                serialized_metadata[key] = json.dumps(value)
            else:
                serialized_metadata[key] = str(value)
        
        return RawContent(
            source_type="website",
            source_url=base_url,
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
        base_domain = parsed_base.netloc
        base_path = parsed_base.path.rstrip('/')
        
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

# Example usage
if __name__ == "__main__":
    async def test_crawl4ai():
        # Test simple crawling
        print("Testing simple crawling...")
        simple_fetcher = create_crawl4ai_fetcher(crawl_type="simple", verbose=True)
        result = simple_fetcher.fetch("https://example.com")
        print(f"Simple crawl found {len(result.content)} pages")
        
        # Test deep crawling
        print("\nTesting deep crawling...")
        deep_fetcher = create_crawl4ai_fetcher(
            crawl_type="deep", 
            max_pages=10, 
            max_depth=2,
            verbose=True
        )
        result = deep_fetcher.fetch("https://example.com")
        print(f"Deep crawl found {len(result.content)} pages")
        
        # Test adaptive crawling
        print("\nTesting adaptive crawling...")
        adaptive_fetcher = create_crawl4ai_fetcher(
            crawl_type="adaptive",
            query="documentation and examples",
            max_pages=15,
            verbose=True
        )
        result = adaptive_fetcher.fetch("https://example.com")
        print(f"Adaptive crawl found {len(result.content)} pages")
        
        # Print some results
        for url, content in list(result.content.items())[:2]:
            print(f"\n--- {url} ---")
            print(content[:200] + "..." if len(content) > 200 else content)
    
    # Run the test
    # asyncio.run(test_crawl4ai())
    print("Crawl4AI fetcher implementation ready!")