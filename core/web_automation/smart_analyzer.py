from bs4 import BeautifulSoup
import re
from typing import Dict, List, Any
import json
from pathlib import Path
import asyncio
from datetime import datetime

class SmartAnalyzer:
    def __init__(self):
        self.cache_dir = Path(__file__).parent / 'cache'
        self.cache_dir.mkdir(exist_ok=True)
        self.load_patterns()

    def load_patterns(self):
        """Load analysis patterns and keywords"""
        self.patterns = {
            'contact': r'[\w\.-]+@[\w\.-]+\.\w+',  # Email pattern
            'phone': r'\+?1?\d{9,15}',  # Phone pattern
            'price': r'\$\d+(?:\.\d{2})?',  # Price pattern
            'date': r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}'  # Date pattern
        }
        
        self.keywords = {
            'business': ['company', 'business', 'enterprise', 'industry', 'market'],
            'technology': ['software', 'hardware', 'cloud', 'AI', 'data'],
            'contact': ['contact', 'email', 'phone', 'address'],
            'product': ['product', 'service', 'solution', 'feature']
        }

    def analyze_content(self, html_content: str, url: str) -> Dict[str, Any]:
        """Smart content analysis with caching"""
        cache_key = self._generate_cache_key(url)
        cached_result = self._get_cached_result(cache_key)
        
        if cached_result:
            return cached_result

        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract key information
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'url': url,
            'title': self._extract_title(soup),
            'main_content': self._extract_main_content(soup),
            'key_points': self._extract_key_points(soup),
            'metadata': self._extract_metadata(soup),
            'structured_data': self._extract_structured_data(soup),
            'relevance_score': self._calculate_relevance(soup)
        }
        
        # Cache the results
        self._cache_result(cache_key, analysis)
        return analysis

    def _generate_cache_key(self, url: str) -> str:
        """Generate a unique cache key for the URL"""
        from hashlib import md5
        return md5(url.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Dict[str, Any]:
        """Get cached analysis result if exists and fresh"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            data = json.loads(cache_file.read_text())
            # Check if cache is less than 24 hours old
            cached_time = datetime.fromisoformat(data['timestamp'])
            if (datetime.now() - cached_time).days < 1:
                return data
        return None

    def _cache_result(self, cache_key: str, data: Dict[str, Any]):
        """Cache analysis results"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract and clean page title"""
        title = soup.title.string if soup.title else ''
        return self._clean_text(title)

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content intelligently"""
        # Remove unwanted elements
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()
            
        # Find main content area
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'content|main|article'))
        if main_content:
            return self._clean_text(main_content.get_text())
        return self._clean_text(soup.get_text())

    def _extract_key_points(self, soup: BeautifulSoup) -> List[str]:
        """Extract key points from content"""
        key_points = []
        
        # Look for headings
        for heading in soup.find_all(['h1', 'h2', 'h3']):
            text = self._clean_text(heading.get_text())
            if text and len(text) > 5:
                key_points.append(text)
                
        # Look for bullet points
        for list_item in soup.find_all('li'):
            text = self._clean_text(list_item.get_text())
            if text and len(text) > 10:
                key_points.append(text)
                
        return key_points[:10]  # Return top 10 key points

    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract metadata from page"""
        metadata = {}
        
        # Get meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name', '').lower()
            content = meta.get('content', '')
            if name and content:
                metadata[name] = content
                
        return metadata

    def _extract_structured_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract structured data (Schema.org, etc.)"""
        structured_data = {}
        
        # Find JSON-LD
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                data = json.loads(script.string)
                structured_data.update(data)
            except:
                continue
                
        return structured_data

    def _calculate_relevance(self, soup: BeautifulSoup) -> float:
        """Calculate content relevance score"""
        text = soup.get_text().lower()
        score = 0.0
        
        # Check keyword presence
        for category, keywords in self.keywords.items():
            category_score = sum(1 for keyword in keywords if keyword.lower() in text)
            score += category_score / len(keywords)
            
        # Normalize score to 0-1 range
        return min(score / len(self.keywords), 1.0)

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
            
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove special characters
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text 