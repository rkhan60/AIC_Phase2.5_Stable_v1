import validators
from playwright.sync_api import sync_playwright
from urllib.parse import urlparse, quote_plus
import logging
import json
from pathlib import Path
import os
from .smart_analyzer import SmartAnalyzer
from typing import List, Dict, Any
import asyncio
from bs4 import BeautifulSoup

class WebController:
    def __init__(self):
        self.allowed_domains = self._load_allowed_domains()
        self.setup_logging()
        self.analyzer = SmartAnalyzer()
        
    def setup_logging(self):
        """Setup logging for web automation"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('web_automation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_allowed_domains(self):
        """Load list of allowed domains from configuration"""
        config_path = Path(__file__).parent / 'config' / 'allowed_domains.json'
        
        # Create default allowed domains if config doesn't exist
        if not config_path.exists():
            default_domains = {
                "allowed": [
                    "google.com",
                    "wikipedia.org",
                    "github.com",
                    "stackoverflow.com",
                    "python.org"
                ],
                "blocked": [
                    "localhost",
                    "127.0.0.1",
                    "192.168.",
                    "10.",
                    "172."
                ]
            }
            
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(default_domains, f, indent=4)
                
        with open(config_path) as f:
            return json.load(f)
    
    def is_url_allowed(self, url):
        """Check if URL is allowed based on security rules"""
        if not validators.url(url):
            self.logger.warning(f"Invalid URL format: {url}")
            return False
            
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        # Check against blocked domains
        for blocked in self.allowed_domains["blocked"]:
            if blocked in domain:
                self.logger.warning(f"Blocked domain attempted: {domain}")
                return False
        
        # Check if domain is in allowed list
        for allowed in self.allowed_domains["allowed"]:
            if allowed in domain:
                return True
                
        self.logger.warning(f"Domain not in allowed list: {domain}")
        return False
    
    def browse_safely(self, url, action=None):
        """Safely browse a URL with optional action and smart analysis"""
        if not self.is_url_allowed(url):
            raise ValueError(f"Access to {url} is not allowed")
            
        self.logger.info(f"Accessing URL: {url}")
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            try:
                page.goto(url)
                
                if action:
                    action(page)
                    
                content = page.content()
                title = page.title()
                
                # Perform smart analysis
                analysis = self.analyzer.analyze_content(content, url)
                
                return {
                    "title": title,
                    "url": url,
                    "analysis": analysis
                }
                
            except Exception as e:
                self.logger.error(f"Error accessing {url}: {str(e)}")
                raise
            
            finally:
                browser.close()
    
    def smart_search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Perform an intelligent search with analysis"""
        search_url = f"https://www.google.com/search?q={quote_plus(query)}&num={num_results}"
        results = []
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            try:
                page.goto(search_url)
                content = page.content()
                soup = BeautifulSoup(content, 'html.parser')
                
                # Extract search results
                for result in soup.select('div.g'):
                    link = result.find('a')
                    if not link:
                        continue
                        
                    url = link.get('href')
                    if not url or not url.startswith('http'):
                        continue
                        
                    if self.is_url_allowed(url):
                        try:
                            analysis = self.browse_safely(url)
                            if analysis['analysis']['relevance_score'] > 0.3:  # Filter by relevance
                                results.append(analysis)
                        except Exception as e:
                            self.logger.warning(f"Failed to analyze {url}: {str(e)}")
                            
                return sorted(results, key=lambda x: x['analysis']['relevance_score'], reverse=True)
                
            finally:
                browser.close()
    
    def add_allowed_domain(self, domain):
        """Add a new domain to allowed list"""
        if domain not in self.allowed_domains["allowed"]:
            self.allowed_domains["allowed"].append(domain)
            self._save_allowed_domains()
            
    def _save_allowed_domains(self):
        """Save the current allowed domains configuration"""
        config_path = Path(__file__).parent / 'config' / 'allowed_domains.json'
        with open(config_path, 'w') as f:
            json.dump(self.allowed_domains, f, indent=4) 