import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
from typing import Dict, Optional


class WebContentExtractor:
    """
    Extracts main content from web pages for AI analysis.
    Simple implementation that fetches and parses HTML content.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.timeout = 30

    def extract_content(self, url: str) -> Dict[str, Optional[str]]:
        """
        Extract main content from a web page.

        Returns:
            Dict with keys: 'title', 'content', 'url', 'error'
        """
        result = {
            'title': None,
            'content': None,
            'url': url,
            'error': None
        }

        try:
            # Validate URL
            if not self._is_valid_url(url):
                result['error'] = "Invalid URL format"
                return result

            # Fetch the page
            print(" - Fetching webpage...")
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract title
            result['title'] = self._extract_title(soup)

            # Extract main content
            result['content'] = self._extract_main_content(soup)

            if not result['content']:
                result['error'] = "No readable content found on the page"
                return result

            print(f" - Extracted content: \"{result['title']}\" ({len(result['content'].split())} words)")

        except requests.exceptions.Timeout:
            result['error'] = "Request timed out"
        except requests.exceptions.ConnectionError:
            result['error'] = "Could not connect to the website"
        except requests.exceptions.HTTPError as e:
            result['error'] = f"HTTP error: {e}"
        except requests.exceptions.RequestException as e:
            result['error'] = f"Request failed: {e}"
        except Exception as e:
            result['error'] = f"Unexpected error: {e}"

        return result

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL has a valid format."""
        try:
            parsed = urlparse(url)
            return bool(parsed.scheme and parsed.netloc)
        except:
            return False

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title with fallbacks."""
        # Try different title sources in order of preference
        title_selectors = [
            'h1',
            'title',
            '[property="og:title"]',
            '[name="twitter:title"]',
            '.title',
            '.headline'
        ]

        for selector in title_selectors:
            element = soup.select_one(selector)
            if element:
                title = element.get_text(strip=True) if selector not in ['title'] else element.get_text(strip=True)
                if selector == '[property="og:title"]' or selector == '[name="twitter:title"]':
                    title = element.get('content', '').strip()
                if title and len(title) > 5:  # Avoid very short titles
                    return title

        return "Untitled Page"

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main readable content from the page."""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer',
                           'aside', 'advertisement', 'ads', 'sidebar']):
            element.decompose()

        # Try to find main content areas
        content_selectors = [
            'article',
            'main',
            '[role="main"]',
            '.content',
            '.article',
            '.post',
            '.entry',
            '#content',
            '#main',
            '.main-content'
        ]

        # Try each selector
        for selector in content_selectors:
            content_element = soup.select_one(selector)
            if content_element:
                text = self._clean_text(content_element.get_text())
                if len(text.split()) > 50:  # Must have substantial content
                    return text

        # Fallback: try to extract from body, but filter out likely navigation/junk
        body = soup.find('body')
        if body:
            # Remove common non-content elements
            for element in body(['nav', 'header', 'footer', 'aside', 'menu',
                               'breadcrumb', 'pagination', 'sidebar']):
                element.decompose()

            # Look for paragraphs and headings
            content_elements = body.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
            if content_elements:
                texts = [self._clean_text(elem.get_text()) for elem in content_elements]
                content = '\n\n'.join([t for t in texts if t and len(t.split()) > 3])
                if content and len(content.split()) > 50:
                    return content

        # Last resort: get all text and clean it
        text = self._clean_text(soup.get_text())
        return text if text else ""

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove multiple line breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

        # Clean up common junk patterns
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            # Skip very short lines that are likely navigation
            if len(line) < 3:
                continue
            # Skip lines that look like navigation or metadata
            if re.match(r'^(Home|About|Contact|Menu|Search|Login|Register|\d+|→|←|»|«)$', line, re.IGNORECASE):
                continue
            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines).strip()