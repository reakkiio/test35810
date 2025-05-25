from webscout.scout import ScoutCrawler

# Crawl a website with default settings
crawler = ScoutCrawler('https://en.wikipedia.org/wiki/Model_Context_Protocol', tags_to_remove=None )  # Default: max_pages=50


# Start crawling
for page in crawler.crawl():
    print(f"URL: {page['url']}")
    print(f"Title: {page['title']}")
    print(f"Links found: {len(page['links'])}")
    print(f"Crawl depth: {page['depth']}")