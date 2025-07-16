import re
from urllib.parse import urlparse

def normalize_url(url: str) -> str:
    """
    Normalize URL to avoid duplicates caused by trailing slashes, case differences, etc.
    
    Args:
        url: Raw URL string
        
    Returns:
        Normalized URL string
    """
    if not url or not isinstance(url, str):
        return url
    
    try:
        # Parse the URL
        parsed = urlparse(url.strip())
        
        # Normalize domain to lowercase
        domain = parsed.netloc.lower()
        
        # Normalize path: remove trailing slash (including for root paths)
        path = parsed.path
        if path.endswith('/'):
            path = path.rstrip('/')
        # Ensure we have at least an empty path (not None)
        if not path:
            path = ""
        
        # Remove common tracking parameters
        query = parsed.query
        if query:
            # Remove common tracking params like utm_*, fbclid, gclid, etc.
            query_params = []
            for param in query.split('&'):
                if '=' in param:
                    key, _ = param.split('=', 1)
                    # Keep only non-tracking parameters
                    if not re.match(r'^(utm_|fbclid|gclid|_ga|_gl)', key.lower()):
                        query_params.append(param)
            query = '&'.join(query_params)
        
        # Reconstruct URL without fragment (anchor links)
        normalized = f"{parsed.scheme}://{domain}{path}"
        if query:
            normalized += f"?{query}"
            
        return normalized
        
    except Exception:
        # If URL parsing fails, return original
        return url