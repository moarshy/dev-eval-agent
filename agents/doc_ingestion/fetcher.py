from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import json

class RawContent(BaseModel):
    """Raw content fetched from various sources"""
    source_type: str = Field(..., description="Type of source: openapi, website, or github")
    source_url: str = Field(..., description="Original source URL or file path")
    content: Dict[str, str] = Field(..., description="Key-value pairs of extracted content")
    metadata: Dict[str, str] = Field(default_factory=dict, description="Additional metadata about the source")
    fetch_timestamp: str = Field(..., description="ISO timestamp of when content was fetched")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

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
    


class OpenAPIFetcher(ContentFetcher):
    """Fetcher for OpenAPI specifications - Not implemented yet"""
    
    def fetch(self, source: str) -> RawContent:
        raise NotImplementedError("OpenAPI fetching is not implemented yet")

class GitHubFetcher(ContentFetcher):
    """Fetcher for GitHub repositories - Not implemented yet"""
    
    def __init__(self, github_token: Optional[str] = None):
        self.github_token = github_token
    
    def fetch(self, source: str) -> RawContent:
        raise NotImplementedError("GitHub fetching is not implemented yet")

class ContentFetcherFactory:
    """Factory for creating appropriate content fetchers"""
    
    @staticmethod
    def create_fetcher(source_type: str, **kwargs) -> ContentFetcher:
        """Create appropriate fetcher based on source type"""
        if source_type == "website":
            return WebsiteFetcher(**kwargs)
        elif source_type == "openapi":
            return OpenAPIFetcher(**kwargs)
        elif source_type == "github":
            return GitHubFetcher(**kwargs)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
    
    @staticmethod
    def detect_source_type(source: str) -> str:
        """Auto-detect source type from URL or content"""
        if 'github.com' in source:
            return 'github'
        elif source.endswith(('.json', '.yaml', '.yml')) or 'openapi' in source:
            return 'openapi'
        else:
            return 'website'