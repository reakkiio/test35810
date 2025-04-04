# 🕵️ Scout: Next-Gen Web Parsing Library

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained-Yes-brightgreen.svg)](https://github.com/OE-LUCIFER/Webscout)
[![Documentation](https://img.shields.io/badge/Docs-Wiki-orange)](https://github.com/OE-LUCIFER/Webscout/wiki)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen.svg)](https://github.com/OE-LUCIFER/Webscout/pulls)

</div>

## 📋 Overview

Scout is a powerful, flexible, and performant HTML parsing library that makes web scraping a breeze! It provides intelligent HTML/XML parsing with advanced features like web crawling, text analysis, semantic extraction, and Markdown conversion. Scout goes beyond traditional parsing libraries with its intuitive API and comprehensive feature set.

<details open>
<summary><b>Why Choose Scout?</b></summary>

- **Powerful Parsing**: Multiple parser backends with intelligent markup handling
- **Advanced Analysis**: Built-in text and web content analysis tools
- **Concurrent Crawling**: Efficient multi-threaded web crawling
- **Flexible API**: Intuitive interface similar to BeautifulSoup but with enhanced capabilities
- **Format Conversion**: Convert HTML to JSON, Markdown, and more

</details>

## 📑 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Features](#-features)
- [Advanced Usage](#-advanced-usage)
- [API Reference](#-api-reference)
- [Dependencies](#-dependencies)
- [Supported Python Versions](#-supported-python-versions)
- [Contributing](#-contributing)
- [License](#-license)

## 📦 Installation

```bash
pip install webscout
```

Or install the latest version from GitHub:

```bash
pip install git+https://github.com/OE-LUCIFER/Webscout.git
```

## 🚀 Quick Start

### Basic Parsing

```python
from webscout.scout import Scout

# Parse HTML content
html_content = """
<html>
    <body>
        <h1>Hello, Scout!</h1>
        <div class="content">
            <p>Web parsing made easy.</p>
            <a href="https://example.com">Link</a>
        </div>
    </body>
</html>
"""

scout = Scout(html_content)

# Find elements
title = scout.find('h1')
links = scout.find_all('a')

# Extract text
print(title[0].get_text())  # Output: Hello, Scout!
print(links.attrs('href'))  # Output: ['https://example.com']
```

### Web Crawling

```python
from webscout.scout import ScoutCrawler

# Crawl a website with default settings
crawler = ScoutCrawler('https://example.com')  # Default: max_pages=50

# Or customize the crawler
crawler = ScoutCrawler(
    'https://example.com',                      # base_url
    max_pages=100,                              # maximum pages to crawl
    tags_to_remove=['script', 'style', 'nav']   # tags to remove from content
)

# Start crawling
crawled_pages = crawler.crawl()

for page in crawled_pages:
    print(f"URL: {page['url']}")
    print(f"Title: {page['title']}")
    print(f"Links found: {len(page['links'])}")
    print(f"Crawl depth: {page['depth']}")
```

### Text Analysis

```python
from webscout.scout import Scout

# Parse a webpage
html = """<div><h1>Climate Change</h1><p>Email us at info@example.com or call 555-123-4567.</p>
<p>Visit https://climate-action.org for more information.</p></div>"""
scout = Scout(html)

# Analyze text and extract entities
analysis = scout.analyze_text()
print(f"Word frequencies: {analysis['word_count']}")
print(f"Entities found: {analysis['entities']}")
```

## ✨ Features

### 🔍 Multiple Parser Support

Scout supports multiple HTML/XML parsers, allowing you to choose the best tool for your specific needs:

| Parser | Description | Best For |
|--------|-------------|----------|
| `html.parser` | Python's built-in parser | General-purpose parsing, no dependencies |
| `lxml` | Fast C-based parser | Performance-critical applications |
| `html5lib` | Highly compliant HTML5 parser | Handling malformed HTML |
| `lxml-xml` | XML parser | XML document parsing |

```python
# Choose your parser
scout = Scout(html_content, features='lxml')  # For speed
scout = Scout(html_content, features='html5lib')  # For compliance
```

### 🌐 Advanced Parsing Capabilities

Scout provides powerful tools for navigating and manipulating HTML/XML documents:

- **Element Selection**: Find elements by tag name, attributes, CSS selectors, and more
- **Tree Traversal**: Navigate parent-child relationships and sibling elements
- **Content Extraction**: Extract text, attributes, and structured data
- **Document Manipulation**: Modify, replace, or remove elements

```python
# CSS selector support
elements = scout.select('div.content > p')

# Advanced find with attribute matching
results = scout.find_all('a', attrs={'class': 'external', 'rel': 'nofollow'})

# Tree traversal
parent = element.find_parent('div')
siblings = element.find_next_siblings('p')
```

### 🧠 Intelligent Analysis

Scout includes built-in analysis tools for extracting insights from web content:

#### Text Analysis

```python
# Extract and analyze text
text = scout.get_text()
word_counts = scout.text_analyzer.count_words(text)
entities = scout.text_analyzer.extract_entities(text)
```

#### Web Structure Analysis

```python
# Analyze page structure
structure = scout.analyze_page_structure()
print(f"Most common tags: {structure['tag_distribution']}")
print(f"Page depth: {max(structure['depth_analysis'].keys())}")
```

#### Semantic Information Extraction

```python
# Extract semantic information
semantics = scout.extract_semantic_info()
print(f"Headings: {semantics['headings']}")
print(f"Lists: {len(semantics['lists']['ul']) + len(semantics['lists']['ol'])}")
print(f"Tables: {semantics['tables']['count']}")
```

### 🕸️ Web Crawling

Scout includes a powerful concurrent web crawler for fetching and analyzing multiple pages:

```python
from webscout.scout import ScoutCrawler

# Create a crawler with default settings
crawler = ScoutCrawler('https://example.com')  # Default: max_pages=50

# Or customize the crawler with specific options
crawler = ScoutCrawler(
    'https://example.com',                      # base_url
    max_pages=100,                              # maximum pages to crawl
    tags_to_remove=['script', 'style', 'nav']   # tags to remove from content
)

# Start crawling
pages = crawler.crawl()

# Process results
for page in pages:
    print(f"URL: {page['url']}")
    print(f"Title: {page['title']}")
    print(f"Links: {len(page['links'])}")
    print(f"Depth: {page['depth']}")
```

The crawler automatically:
- Stays within the same domain as the base URL
- Uses concurrent requests for faster crawling
- Removes unwanted tags (like scripts and styles) for cleaner text extraction
- Tracks crawl depth for each page

### 📄 Format Conversion

Scout can convert HTML to various formats:

```python
# Convert to JSON
json_data = scout.to_json(indent=2)

# Convert to Markdown
markdown = scout.to_markdown(heading_style='ATX')

# Pretty-print HTML
pretty_html = scout.prettify()
```

## 🔬 Advanced Usage

### Working with Search Results

Scout's search methods return a `ScoutSearchResult` object with powerful methods for processing results:

```python
from webscout.scout import Scout

scout = Scout(html_content)

# Find all paragraphs
paragraphs = scout.find_all('p')

# Extract all text from results
all_text = paragraphs.texts(separator='\n')

# Extract specific attributes
hrefs = paragraphs.attrs('href')

# Filter results with a predicate function
important = paragraphs.filter(lambda p: 'important' in p.get('class', []))

# Transform results
word_counts = paragraphs.map(lambda p: len(p.get_text().split()))

# Analyze text in results
analysis = paragraphs.analyze_text()
```

### URL Handling and Analysis

```python
from webscout.scout import Scout

scout = Scout(html_content)

# Parse and analyze URLs
links = scout.extract_links(base_url='https://example.com')
for link in links:
    url_components = scout.url_parse(link['href'])
    print(f"Domain: {url_components['netloc']}")
    print(f"Path: {url_components['path']}")
```

### Metadata Extraction

```python
from webscout.scout import Scout

scout = Scout(html_content)

# Extract metadata
metadata = scout.extract_metadata()
print(f"Title: {metadata['title']}")
print(f"Description: {metadata['description']}")
print(f"Open Graph: {metadata['og_metadata']}")
print(f"Twitter Card: {metadata['twitter_metadata']}")
```

### Content Hashing and Caching

```python
from webscout.scout import Scout

scout = Scout(html_content)

# Generate content hash
content_hash = scout.hash_content(method='sha256')

# Use caching for expensive operations
if not scout.cache('parsed_data'):
    data = scout.extract_semantic_info()
    scout.cache('parsed_data', data)

cached_data = scout.cache('parsed_data')
```

## 📚 API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `Scout` | Main class for HTML parsing and traversal |
| `ScoutCrawler` | Web crawler for fetching and parsing multiple pages |
| `ScoutTextAnalyzer` | Text analysis utilities |
| `ScoutWebAnalyzer` | Web page analysis utilities |
| `ScoutSearchResult` | Enhanced search results with filtering and analysis |
| `Tag` | Represents an HTML/XML tag |
| `NavigableString` | Represents text within an HTML/XML document |

### Key Methods

#### Scout Class

- `__init__(markup, features='html.parser', from_encoding=None)`: Initialize with HTML content
- `find(name, attrs={}, recursive=True, text=None)`: Find first matching element
- `find_all(name, attrs={}, recursive=True, text=None, limit=None)`: Find all matching elements
- `select(selector)`: Find elements using CSS selector
- `get_text(separator=' ', strip=False)`: Extract text from document
- `analyze_text()`: Perform text analysis
- `analyze_page_structure()`: Analyze document structure
- `extract_semantic_info()`: Extract semantic information
- `extract_links(base_url=None)`: Extract all links
- `extract_metadata()`: Extract metadata from document
- `to_json(indent=2)`: Convert to JSON
- `to_markdown(heading_style='ATX')`: Convert to Markdown
- `prettify(formatter='minimal')`: Pretty-print HTML

#### ScoutCrawler Class

- `__init__(base_url, max_pages=50, tags_to_remove=None)`: Initialize the crawler
- `crawl()`: Start crawling from the base URL
- `_crawl_page(url, depth=0)`: Crawl a single page (internal method)
- `_is_valid_url(url)`: Check if a URL is valid (internal method)

For detailed API documentation, please refer to the [documentation](https://github.com/OE-LUCIFER/Webscout/wiki).

## 🔧 Dependencies

- `requests`: HTTP library for making web requests
- `lxml`: XML and HTML processing library (optional, recommended)
- `html5lib`: Standards-compliant HTML parser (optional)
- `markdownify`: HTML to Markdown conversion
- `concurrent.futures`: Asynchronous execution (standard library)

## 🌈 Supported Python Versions

- Python 3.8+

## 🤝 Contributing

Contributions are welcome! Here's how you can contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please make sure to update tests as appropriate.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<div align="center">
  <p>Made with ❤️ by the Webscout team</p>
  <p>
    <a href="https://github.com/OE-LUCIFER/Webscout">GitHub</a> •
    <a href="https://github.com/OE-LUCIFER/Webscout/wiki">Documentation</a> •
    <a href="https://github.com/OE-LUCIFER/Webscout/issues">Report Bug</a> •
    <a href="https://github.com/OE-LUCIFER/Webscout/issues">Request Feature</a>
  </p>
</div>
