from webscout.scout import ScoutCrawler
from rich import print
# Crawl a website with default settings
crawler = ScoutCrawler('https://en.wikipedia.org/wiki/India', tags_to_remove=None, max_pages=3)  # Default: max_pages=50


# Start crawling
for page in crawler.crawl():
    print(f"URL: {page['url']}")
    print(f"Title: {page['title']}")
    print(page['text'][:100])  # Print first 100 characters of text
    print(f"Links found: {len(page['links'])}")
    print(f"Crawl depth: {page['depth']}")