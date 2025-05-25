"""
Scout Crawler Module
"""

import concurrent.futures
import urllib.parse
import time
import hashlib
import re
from urllib import robotparser
from datetime import datetime
from typing import Dict, List, Optional, Union
from webscout.litagent import LitAgent
from curl_cffi.requests import Session

from .scout import Scout


class ScoutCrawler:
    """
    Advanced web crawling utility for Scout library.
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
            "style",
            "header",
            "footer",
            "nav",
            "aside",
            "form",
            "button",
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
        self.allowed_domains = allowed_domains or [urllib.parse.urlparse(base_url).netloc]
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
            if parsed_url.netloc not in self.allowed_domains:
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
            for tag_name in self.tags_to_remove:
                for tag in scout._soup.find_all(tag_name):
                    tag.extract()
            visible_text = self._extract_main_text(scout._soup)
            page_info = {
                'url': url,
                'title': title,
                'links': [
                    urllib.parse.urljoin(url, link.get('href'))
                    for link in scout.find_all('a', href=True)
                    if self._is_valid_url(urllib.parse.urljoin(url, link.get('href')))
                ],
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
                if len(self.visited_urls) >= self.max_pages:
                    break
                done, not_done = concurrent.futures.wait(
                    futures, return_when=concurrent.futures.FIRST_COMPLETED
                )
                futures = not_done

                for future in done:
                    page_info = future.result()

                    if page_info:
                        yield page_info

                    if len(self.visited_urls) >= self.max_pages:
                        return

                    for link in page_info.get("links", []):
                        if (
                            len(self.visited_urls) < self.max_pages
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
