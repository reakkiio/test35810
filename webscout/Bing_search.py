"""
BingSearch - A Bing search library with advanced features
"""
from time import sleep
from curl_cffi.requests import Session
from urllib.parse import urlencode, unquote, urlparse, parse_qs
import base64
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor
from webscout.litagent import LitAgent
class BingSearchResult:
    """Class to represent a Bing search result with metadata."""
    def __init__(self, url: str, title: str, description: str):
        self.url = url
        self.title = title
        self.description = description
        self.metadata: Dict[str, Any] = {}

    def __repr__(self) -> str:
        return f"BingSearchResult(url={self.url}, title={self.title}, description={self.description})"

class BingImageResult:
    """Class to represent a Bing image search result."""
    def __init__(self, title: str, image: str, thumbnail: str, url: str, source: str):
        self.title = title
        self.image = image
        self.thumbnail = thumbnail
        self.url = url
        self.source = source
    def __repr__(self):
        return f"BingImageResult(title={self.title}, image={self.image}, url={self.url}, source={self.source})"

class BingNewsResult:
    """Class to represent a Bing news search result."""
    def __init__(self, title: str, url: str, description: str, source: str = ""):
        self.title = title
        self.url = url
        self.description = description
        self.source = source
    def __repr__(self):
        return f"BingNewsResult(title={self.title}, url={self.url}, source={self.source})"

class BingSearch:
    """Bing search implementation with configurable parameters and advanced features."""
    _executor: ThreadPoolExecutor = ThreadPoolExecutor()

    def __init__(
        self,
        timeout: int = 10,
        proxies: Optional[Dict[str, str]] = None,
        verify: bool = True,
        lang: str = "en-US",
        sleep_interval: float = 0.0,
        impersonate: str = "chrome110"
    ):
        self.timeout = timeout
        self.proxies = proxies if proxies else {}
        self.verify = verify
        self.lang = lang
        self.sleep_interval = sleep_interval
        self._base_url = "https://www.bing.com"
        self.session = Session(
            proxies=self.proxies,
            verify=self.verify,
            timeout=self.timeout,
            impersonate=impersonate
        )
        self.session.headers.update(LitAgent().generate_fingerprint())

    def _selectors(self, element):
        selectors = {
            'url': 'h2 a',
            'title': 'h2',
            'text': 'p',
            'links': 'ol#b_results > li.b_algo',
            'next': 'div#b_content nav[role="navigation"] a.sb_pagN'
        }
        return selectors[element]

    def _first_page(self, query):
        url = f'{self._base_url}/search?q={query}&search=&form=QBLH'
        return {'url': url, 'data': None}

    def _next_page(self, soup):
        selector = self._selectors('next')
        next_page_tag = soup.select_one(selector)
        url = None
        if next_page_tag and next_page_tag.get('href'):
            url = self._base_url + next_page_tag['href']
        return {'url': url, 'data': None}

    def _get_url(self, tag):
        url = tag.get('href', '')
        resp = url
        try:
            parsed_url = urlparse(url)
            query_params = parse_qs(parsed_url.query)
            if "u" in query_params:
                encoded_url = query_params["u"][0][2:]
                try:
                    decoded_bytes = base64.urlsafe_b64decode(encoded_url + '===')
                except base64.binascii.Error as e:
                    print(f"Error decoding Base64 string: {e}")
                    return url
                resp = decoded_bytes.decode('utf-8')
        except Exception as e:
            print(f"Error decoding Base64 string: {e}")
        return resp

    def _make_request(self, term: str, results: int, start: int = 0) -> str:
        params = {
            "q": term,
            "count": results,
            "first": start + 1,
            "setlang": self.lang,
        }
        url = self._base_url + "/search"
        try:
            resp = self.session.get(
                url=url,
                params=params,
            )
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            if hasattr(e, 'response') and e.response is not None:
                raise Exception(f"Bing search failed with status {e.response.status_code}: {str(e)}")
            else:
                raise Exception(f"Bing search failed: {str(e)}")

    def text(
        self,
        keywords: str,
        region: str = None,
        safesearch: str = "moderate",
        max_results: int = 10,
        unique: bool = True
    ) -> List[BingSearchResult]:
        """
        Perform a text search on Bing.

        Args:
            keywords (str): The search keywords.
            region (str, optional): The region for the search. Defaults to None.
            safesearch (str): The safe search level ("on", "moderate", "off"). Defaults to "moderate".
            max_results (int): The maximum number of results to fetch. Defaults to 10.
            unique (bool): Whether to exclude duplicate URLs from the results. Defaults to True.

        Returns:
            List[BingSearchResult]: A list of Bing search results.
        """
        if not keywords:
            raise ValueError("Search keywords cannot be empty")
        from bs4 import BeautifulSoup
        safe_map = {
            "on": "Strict",
            "moderate": "Moderate",
            "off": "Off"
        }
        safe = safe_map.get(safesearch.lower(), "Moderate")
        fetched_results = []
        fetched_links = set()
        def fetch_page(url):
            try:
                resp = self.session.get(url)
                resp.raise_for_status()
                return resp.text
            except Exception as e:
                if hasattr(e, 'response') and e.response is not None:
                    raise Exception(f"Bing search failed with status {e.response.status_code}: {str(e)}")
                else:
                    raise Exception(f"Bing search failed: {str(e)}")

        # Fix: get the first page URL
        url = self._first_page(keywords)['url']
        urls_to_fetch = [url]
        while len(fetched_results) < max_results and urls_to_fetch:
            html_pages = list(self._executor.map(fetch_page, urls_to_fetch))
            urls_to_fetch = []
            for html in html_pages:
                soup = BeautifulSoup(html, "html.parser")
                selector_links = self._selectors('links')
                result_blocks = soup.select(selector_links)
                for result in result_blocks:
                    link_tag = result.select_one(self._selectors('url'))
                    if not link_tag:
                        continue
                    url_val = self._get_url(link_tag)
                    title_tag = result.select_one(self._selectors('title'))
                    title = title_tag.get_text(strip=True) if title_tag else ''
                    desc_tag = result.select_one(self._selectors('text'))
                    description = desc_tag.get_text(strip=True) if desc_tag else ''
                    if url_val and title:
                        if unique and url_val in fetched_links:
                            continue
                        fetched_results.append(BingSearchResult(url_val, title, description))
                        fetched_links.add(url_val)
                        if len(fetched_results) >= max_results:
                            break
                if len(fetched_results) >= max_results:
                    break
                next_page_info = self._next_page(soup)
                if next_page_info['url']:
                    urls_to_fetch.append(next_page_info['url'])
                sleep(self.sleep_interval)
            next_page_info = self._next_page(soup)
            url = next_page_info['url']
            sleep(self.sleep_interval)
        return fetched_results[:max_results]

    def suggestions(self, query: str, region: str = None) -> List[str]:
        """
        Fetches search suggestions for a given query.

        Args:
            query (str): The search query for which suggestions are needed.
            region (str, optional): The region code (e.g., "en-US") for localized suggestions.

        Returns:
            List[str]: A list of suggestion strings related to the query.
        """
        if not query:
            raise ValueError("Search query cannot be empty")
        params = {
            "query": query,
            "mkt": region if region else "en-US"
        }
        url = f"https://api.bing.com/osjson.aspx?{urlencode(params)}"
        try:
            resp = self.session.get(url)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and len(data) > 1 and isinstance(data[1], list):
                return data[1]
            return []
        except Exception as e:
            if hasattr(e, 'response') and e.response is not None:
                raise Exception(f"Bing suggestions failed with status {e.response.status_code}: {str(e)}")
            else:
                raise Exception(f"Bing suggestions failed: {str(e)}")

    def images(
        self,
        keywords: str,
        region: str = None,
        safesearch: str = "moderate",
        max_results: int = 10
    ) -> List[BingImageResult]:
        """
        Perform an image search on Bing.

        Args:
            keywords (str): The search keywords.
            region (str, optional): The region for the search. Defaults to None.
            safesearch (str): The safe search level ("on", "moderate", "off"). Defaults to "moderate".
            max_results (int): The maximum number of results to fetch. Defaults to 10.

        Returns:
            List[BingImageResult]: A list of Bing image search results.
        """
        if not keywords:
            raise ValueError("Search keywords cannot be empty")
        from bs4 import BeautifulSoup
        safe_map = {
            "on": "Strict",
            "moderate": "Moderate",
            "off": "Off"
        }
        safe = safe_map.get(safesearch.lower(), "Moderate")
        params = {
            "q": keywords,
            "count": max_results,
            "setlang": self.lang,
            "safeSearch": safe,
        }
        if region:
            params["mkt"] = region
        url = f"{self._base_url}/images/search?{urlencode(params)}"
        try:
            resp = self.session.get(url)
            resp.raise_for_status()
            html = resp.text
        except Exception as e:
            if hasattr(e, 'response') and e.response is not None:
                raise Exception(f"Bing image search failed with status {e.response.status_code}: {str(e)}")
            else:
                raise Exception(f"Bing image search failed: {str(e)}")
        soup = BeautifulSoup(html, "html.parser")
        results = []
        for item in soup.select("a.iusc"):
            try:
                m = item.get("m")
                import json
                meta = json.loads(m) if m else {}
                image_url = meta.get("murl", "")
                thumb_url = meta.get("turl", "")
                title = meta.get("t", "")
                page_url = meta.get("purl", "")
                source = meta.get("surl", "")
                if image_url:
                    results.append(BingImageResult(title, image_url, thumb_url, page_url, source))
                    if len(results) >= max_results:
                        break
            except Exception:
                continue
        return results[:max_results]

    def news(
        self,
        keywords: str,
        region: str = None,
        safesearch: str = "moderate",
        max_results: int = 10,
    ) -> List['BingNewsResult']:
        """Bing news search."""
        if not keywords:
            raise ValueError("Search keywords cannot be empty")
        from bs4 import BeautifulSoup
        safe_map = {
            "on": "Strict",
            "moderate": "Moderate",
            "off": "Off"
        }
        safe = safe_map.get(safesearch.lower(), "Moderate")
        params = {
            "q": keywords,
            "form": "QBNH",
            "safeSearch": safe,
        }
        if region:
            params["mkt"] = region
        url = f"{self._base_url}/news/search?{urlencode(params)}"
        try:
            resp = self.session.get(url)
            resp.raise_for_status()
        except Exception as e:
            if hasattr(e, 'response') and e.response is not None:
                raise Exception(f"Bing news search failed with status {e.response.status_code}: {str(e)}")
            else:
                raise Exception(f"Bing news search failed: {str(e)}")
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for item in soup.select("div.news-card, div.card, div.newsitem, div.card-content, div.t_s_main"):
            a_tag = item.find("a")
            title = a_tag.get_text(strip=True) if a_tag else ''
            url_val = a_tag['href'] if a_tag and a_tag.has_attr('href') else ''
            desc_tag = item.find("div", class_="snippet") or item.find("div", class_="news-card-snippet") or item.find("div", class_="snippetText")
            description = desc_tag.get_text(strip=True) if desc_tag else ''
            source_tag = item.find("div", class_="source")
            source = source_tag.get_text(strip=True) if source_tag else ''
            if url_val and title:
                results.append(BingNewsResult(title, url_val, description, source))
                if len(results) >= max_results:
                    break
        # Fallback: try main news list if above selectors fail
        if not results:
            for item in soup.select("a.title"):
                title = item.get_text(strip=True)
                url_val = item['href'] if item.has_attr('href') else ''
                description = ''
                source = ''
                if url_val and title:
                    results.append(BingNewsResult(title, url_val, description, source))
                    if len(results) >= max_results:
                        break
        return results[:max_results]

if __name__ == "__main__":
    from rich import print
    bing = BingSearch(
        timeout=10,
        proxies=None,
        verify=True
    )
    print("TEXT SEARCH RESULTS:")
    text_results = bing.text(
        keywords="Python programming",
        region="us",
        safesearch="moderate",
        max_results=30
    )
    for result in text_results:
        print(f"Title: {result.title}")
        print(f"URL: {result.url}")
        print(f"Description: {result.description}")
        print("---")
    print("\nSEARCH SUGGESTIONS:")
    suggestions = bing.suggestions("how to")
    print(suggestions)

    print("\nIMAGE SEARCH RESULTS:")
    image_results = bing.images(
        keywords="Python programming",
        region="us",
        safesearch="moderate",
        max_results=10
    )
    for result in image_results:
        print(f"Title: {result.title}")
        print(f"Image URL: {result.image}")
        print(f"Page URL: {result.url}")
        print(f"Source: {result.source}")
        print("---")

    print("\nNEWS SEARCH RESULTS:")
    news_results = bing.news(
        keywords="Python programming",
        region="us",
        safesearch="moderate",
        max_results=10
    )
    for result in news_results:
        print(f"Title: {result.title}")
        print(f"URL: {result.url}")
        print(f"Description: {result.description}")
        print(f"Source: {result.source}")
        print("---")
