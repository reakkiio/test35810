"""
Scout Crawler Module - Ultra Advanced Web Crawling System
"""

import concurrent.futures
import urllib.parse
import time
import hashlib
import re
import json
import sqlite3
import threading
import queue
import logging
import mimetypes
import pickle
import asyncio
import aiohttp
import random
from urllib import robotparser
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Set, Tuple, Callable, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

try:
    from webscout.litagent import LitAgent
except ImportError:
    LitAgent = None
    
try:
    from curl_cffi.requests import Session
except ImportError:
    import requests
    Session = requests.Session

from .scout import Scout
from .text_analyzer import ScoutTextAnalyzer


@dataclass
class CrawlConfig:
    """Configuration for the crawler."""
    max_pages: int = 1000
    max_depth: int = 10
    delay: float = 0.5
    obey_robots: bool = True
    crawl_subdomains: bool = True
    max_workers: int = 10
    timeout: int = 30
    retry_attempts: int = 3
    include_external_links: bool = False
    extract_metadata: bool = True
    extract_structured_data: bool = True
    extract_semantic_content: bool = True
    

@dataclass
class PageData:
    """Comprehensive page data for LLM training."""
    url: str
    title: str
    text: str
    clean_text: str
    markdown_text: str
    links: List[str]
    internal_links: List[str]
    external_links: List[str]
    metadata: Dict[str, Any]
    structured_data: Dict[str, Any]
    semantic_content: Dict[str, Any]
    headers: Dict[str, str]
    status_code: int
    content_type: str
    language: str
    timestamp: str
    depth: int
    word_count: int
    

class ScoutCrawler:
    """
    Ultra-advanced web crawling utility optimized for LLM data collection.
    """
    def __init__(self, base_url: str, max_pages: int = 50, tags_to_remove: List[str] = None, session: Optional[Session] = None, delay: float = 0.5, obey_robots: bool = True, allowed_domains: Optional[List[str]] = None):
        """
        Initialize the web crawler.

        Args:
            base_url (str): Starting URL to crawl
            max_pages (int, optional): Maximum number of pages to crawl
            tags_to_remove (List[str], optional): List of tags to remove
        """
        self.base_url = base_url
        self.max_pages = max_pages
        self.tags_to_remove = tags_to_remove if tags_to_remove is not None else [
            "script",
            "style"
        ]
        self.visited_urls = set()
        self.crawled_pages = []
        self.session = session or Session()
        self.agent = LitAgent()
        # Use all headers and generate fingerprint
        self.session.headers = self.agent.generate_fingerprint()
        self.session.headers.setdefault("User-Agent", self.agent.chrome())
        self.delay = delay
        self.obey_robots = obey_robots
        # Allow crawling of subdomains by default
        base_domain = urllib.parse.urlparse(base_url).netloc.split('.')
        self.base_domain = '.'.join(base_domain[-2:]) if len(base_domain) > 1 else base_domain[0]
        self.allowed_domains = allowed_domains or [self.base_domain]
        self.last_request_time = 0
        self.url_hashes = set()
        if obey_robots:
            self.robots = robotparser.RobotFileParser()
            robots_url = urllib.parse.urljoin(base_url, '/robots.txt')
            try:
                self.robots.set_url(robots_url)
                self.robots.read()
            except Exception:
                self.robots = None
        else:
            self.robots = None

    def _normalize_url(self, url: str) -> str:
        url = url.split('#')[0]
        url = re.sub(r'\?.*$', '', url)  # Remove query params
        return url.rstrip('/')

    def _is_valid_url(self, url: str) -> bool:
        """
        Check if a URL is valid and within the same domain.

        Args:
            url (str): URL to validate

        Returns:
            bool: Whether the URL is valid
        """
        try:
            parsed_base = urllib.parse.urlparse(self.base_url)
            parsed_url = urllib.parse.urlparse(url)
            if parsed_url.scheme not in ["http", "https"]:
                return False
            # Allow crawling subdomains
            if not parsed_url.netloc.endswith(self.base_domain):
                return False
            if self.obey_robots and self.robots:
                return self.robots.can_fetch("*", url)
            return True
        except Exception:
            return False

    def _is_duplicate(self, url: str) -> bool:
        norm = self._normalize_url(url)
        url_hash = hashlib.md5(norm.encode()).hexdigest()
        if url_hash in self.url_hashes:
            return True
        self.url_hashes.add(url_hash)
        return False

    def _extract_main_text(self, soup):
        # Try to extract main content (simple heuristic)
        main = soup.find('main')
        if main:
            return main.get_text(separator=" ", strip=True)
        article = soup.find('article')
        if article:
            return article.get_text(separator=" ", strip=True)
        # fallback to body
        body = soup.find('body')
        if body:
            return body.get_text(separator=" ", strip=True)
        return soup.get_text(separator=" ", strip=True)

    def _crawl_page(self, url: str, depth: int = 0) -> Dict[str, Union[str, List[str]]]:
        """
        Crawl a single page and extract information.

        Args:
            url (str): URL to crawl
            depth (int, optional): Current crawl depth

        Returns:
            Dict[str, Union[str, List[str]]]: Crawled page information
        """
        if url in self.visited_urls or self._is_duplicate(url):
            return {}
        # Log URL to crawl
        print(f"Attempting to crawl URL: {url} (depth: {depth})")

        # Throttle requests
        now = time.time()
        if self.last_request_time:
            elapsed = now - self.last_request_time
            if elapsed < self.delay:
                time.sleep(self.delay - elapsed)
        self.last_request_time = time.time()
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            if not response.headers.get('Content-Type', '').startswith('text/html'):
                return {}
            scout = Scout(response.content, features="lxml")
            title_result = scout.find("title")
            title = title_result[0].get_text() if title_result else ""
            
            # Remove only script and style tags before extracting text
            for tag_name in self.tags_to_remove:
                for tag in scout._soup.find_all(tag_name):
                    tag.decompose()
                    
            visible_text = self._extract_main_text(scout._soup)
            
            # Extract links from header, footer, nav, etc.
            essential_links = []
            for essential_tag in ['header', 'nav', 'footer']:
                elements = scout.find_all(essential_tag)
                for element in elements:
                    links = element.find_all('a', href=True)
                    essential_links.extend(
                        urllib.parse.urljoin(url, link.get('href'))
                        for link in links
                        if link.get('href') and self._is_valid_url(urllib.parse.urljoin(url, link.get('href')))
                    )

            all_links = [
                urllib.parse.urljoin(url, link.get('href'))
                for link in scout.find_all('a', href=True)
                if self._is_valid_url(urllib.parse.urljoin(url, link.get('href')))
            ]

            combined_links = list(set(all_links + essential_links))

            page_info = {
                'url': url,
                'title': title,
                'links': combined_links,
                'text': visible_text,
                'depth': depth,
                'timestamp': datetime.utcnow().isoformat(),
                'headers': dict(response.headers),
            }
            self.visited_urls.add(url)
            self.crawled_pages.append(page_info)
            return page_info
        except Exception as e:
            print(f"Error crawling {url}: {e}")
            return {}

    def crawl(self):
        """
        Start web crawling from base URL and yield each crawled page in real time.

        Yields:
            Dict[str, Union[str, List[str]]]: Crawled page information
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(self._crawl_page, self.base_url, 0)}
            submitted_links: set[str] = set()

            while futures:
                if self.max_pages is not None and len(self.visited_urls) >= self.max_pages:
                    break
                done, not_done = concurrent.futures.wait(
                    futures, return_when=concurrent.futures.FIRST_COMPLETED
                )
                futures = not_done

                for future in done:
                    page_info = future.result()

                    if page_info:
                        yield page_info
                        
                        if self.max_pages is not None and len(self.visited_urls) >= self.max_pages:
                            return

                        for link in page_info.get("links", []):
                            if (
                                (self.max_pages is None or len(self.visited_urls) < self.max_pages)
                                and link not in self.visited_urls
                                and link not in submitted_links
                            ):
                                submitted_links.add(link)
                                futures.add(
                                    executor.submit(
                                        self._crawl_page,
                                        link,
                                        page_info.get("depth", 0) + 1,
                                    )
                                )
                    else:
                        print(f"No page info retrieved from crawling")
