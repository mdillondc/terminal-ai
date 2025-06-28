import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from typing import Dict, Optional
import re
import time
import json
from datetime import datetime, timedelta

from print_helper import print_info
from settings_manager import SettingsManager
from llm_client_manager import LLMClientManager


class WebContentExtractor:
    """
    Extracts main content from web pages for AI analysis.
    Attempts various bypass methods when access is blocked.
    """

    def __init__(self, llm_client_manager: LLMClientManager):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36'
        })
        self.timeout = 30

        # Store the LLM client manager for content evaluation
        self.llm_client_manager = llm_client_manager
        self.settings_manager = SettingsManager.getInstance()

    def extract_content(self, url: str) -> Dict[str, Optional[str]]:
        """
        Extract main content from a web page with access restriction bypass capabilities.
        Handles paywalls, bot detection, login walls, rate limiting, and other blocking methods.

        Returns:
            Dict with keys: 'title', 'content', 'url', 'error', 'warning'
        """
        result = {
            'title': None,
            'content': None,
            'url': url,
            'error': None,
            'warning': None
        }

        try:
            # Validate URL
            if not self._is_valid_url(url):
                result['error'] = "Invalid URL format"
                return result

            # Try normal extraction first
            print_info("Fetching webpage...")
            normal_result = self._basic_extraction(url)

            # Check if we got an error that might be bypassed (403, 429, etc.)
            if normal_result['error']:
                error_msg = normal_result['error'].lower()
                if any(code in error_msg for code in ['403', '429', 'forbidden', 'blocked', 'bot']):
                    if '403' in error_msg:
                        print_info("Access denied (HTTP 403) - attempting bypass methods...")
                    elif '429' in error_msg:
                        print_info("Rate limited (HTTP 429) - attempting bypass methods...")
                    elif 'bot' in error_msg:
                        print_info("Bot detection triggered - attempting bypass methods...")
                    else:
                        print_info("Access blocked - attempting bypass methods...")

                    bypass_result = self._try_access_bypass(url, "")
                    if bypass_result['content'] and len(bypass_result['content'].split()) > 100:
                        return bypass_result
                    else:
                        print_info("All bypass methods failed - returning original error")
                        return normal_result
                else:
                    return normal_result

            # Check for access restrictions in content using LLM
            if normal_result['content']:
                is_blocked = self._llm_evaluate_content(normal_result['content'])
                if is_blocked:
                    print_info("Access restriction detected - attempting bypass methods...")
                    bypass_result = self._try_access_bypass(url, normal_result['content'])
                    if bypass_result['content']:
                        return bypass_result
                    else:
                        content_length = len(normal_result.get('content', '').split()) if normal_result.get('content') else 0
                        print_info("All bypass methods failed - content appears blocked")
                        print_info(f"Limited content ({content_length} words) added to context - may not be sufficient for analysis")
                        normal_result['warning'] = "Content may be incomplete due to access restrictions"
                        normal_result['bypass_failed'] = True
                        return normal_result
            else:
                # No content to evaluate
                bypass_result = self._try_access_bypass(url, "")
                if bypass_result['content']:
                    return bypass_result
                else:
                    content_length = len(normal_result.get('content', '').split()) if normal_result.get('content') else 0
                    print_info("All bypass methods failed - content appears blocked")
                    print_info(f"Limited content ({content_length} words) added to context - may not be sufficient for analysis")
                    normal_result['warning'] = "Content may be incomplete due to access restrictions"
                    normal_result['bypass_failed'] = True
                    return normal_result

            return normal_result

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
        # Remove unwanted elements - comprehensive HTML-based approach
        unwanted_selectors = [
            # Basic unwanted elements
            'script', 'style', 'nav', 'header', 'footer', 'aside', 'advertisement', 'ads', 'sidebar',

            # Cookie/Privacy/GDPR (attribute-based - works across all sites)
            '[class*="cookie"]', '[id*="cookie"]', '[class*="gdpr"]', '[id*="gdpr"]',
            '[class*="consent"]', '[id*="consent"]', '[class*="privacy"]', '[id*="privacy"]',

            # Popups/Modals/Banners (generic patterns)
            '[class*="banner"]', '[class*="popup"]', '[class*="modal"]', '[class*="overlay"]',
            '[class*="dialog"]', '[class*="notice"]', '[class*="alert"]',

            # Newsletter/Subscription (generic patterns)
            '[class*="newsletter"]', '[class*="subscribe"]', '[class*="signup"]', '[class*="email"]',

            # Donation/Support/Membership (works for Guardian, NYT, etc.)
            '[class*="support"]', '[class*="contribution"]', '[class*="donate"]', '[class*="donation"]',
            '[class*="membership"]', '[class*="reader"]', '[class*="fund"]', '[class*="appeal"]',
            '[class*="fundrais"]', '[class*="patron"]', '[class*="sponsor"]', '[class*="campaign"]',

            # Advertisement/Promotion related
            '[class*="promo"]', '[class*="promotion"]', '[class*="ad-"]', '[class*="cta"]',

            # Social/Sharing popups
            '[class*="share"]', '[class*="social"]', '[class*="follow"]',

            # Common generic class names (works across sites)
            '.banner', '.popup', '.modal', '.overlay', '.notice', '.alert', '.toast',
            '.subscription', '.newsletter', '.signup', '.donate', '.support', '.contribute',
            '.membership', '.reader-support', '.fundraising', '.appeal',

            # Common generic IDs
            '#banner', '#popup', '#modal', '#overlay', '#notice', '#alert',
            '#subscription', '#newsletter', '#donate', '#support', '#membership',
            '#support-banner', '#donation-banner', '#contribution-banner', '#membership-banner',
            '#reader-support', '#fundraising-banner', '#appeal-banner',

            # Generic "call to action" and "revenue" selectors
            '[class*="cta"]', '[class*="call-to-action"]', '[class*="revenue"]'
        ]

        for selector in unwanted_selectors:
            for element in soup.select(selector):
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
            # Remove common non-content elements including additional popups
            non_content_selectors = [
                'nav', 'header', 'footer', 'aside', 'menu', 'breadcrumb', 'pagination', 'sidebar',
                # Additional popup and banner removal for fallback extraction
                '[class*="cookie"]', '[class*="gdpr"]', '[class*="consent"]', '[class*="privacy"]',
                '[class*="banner"]', '[class*="popup"]', '[class*="modal"]', '[class*="overlay"]',
                '[class*="newsletter"]', '[class*="subscribe"]', '[class*="support"]',
                '[class*="contribution"]', '[class*="donate"]', '[class*="membership"]',
                '[class*="donation"]', '[class*="reader-revenue"]', '[class*="fundrais"]',
                '[class*="patron"]', '[class*="sponsor"]', '[class*="appeal"]',
                '[class*="campaign"]', '[class*="cta"]', '[class*="revenue"]'
            ]

            for selector in non_content_selectors:
                for element in body.select(selector):
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

        # Cookie and popup related patterns to filter out - very specific only
        cookie_patterns = [
            r'^we use cookies.*$', r'^this site uses cookies.*$',
            r'^accept all cookies$', r'^reject all cookies$', r'^manage cookies$',
            r'^cookie policy$', r'^cookie settings$', r'^privacy policy$',
            r'^subscribe to our newsletter$', r'^newsletter signup$',
            r'^by continuing to browse.*$', r'^by using this website.*$'
        ]

        # Combine all unwanted text patterns (minimal since HTML removal handles most)
        unwanted_text_patterns = cookie_patterns + [
            r'^support.*journalism.*$', r'^donate.*$', r'^contribute.*$',
            r'^become.*member.*$', r'^join.*$', r'^help.*keep.*free.*$'
        ]

        for line in lines:
            line = line.strip()
            # Skip very short lines that are likely navigation
            if len(line) < 3:
                continue
            # Skip lines that look like navigation or metadata
            if re.match(r'^(Home|About|Contact|Menu|Search|Login|Register|\d+|→|←|»|«)$', line, re.IGNORECASE):
                continue
            # Skip obvious popup/banner text that HTML removal missed
            if any(re.match(pattern, line, re.IGNORECASE) for pattern in unwanted_text_patterns):
                continue
            # Skip very generic popup text
            if line.lower() in ['ok', 'accept', 'decline', 'agree', 'disagree', 'yes', 'no', 'close', 'x']:
                continue
            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines).strip()

    def _basic_extraction(self, url: str) -> Dict[str, Optional[str]]:
        """Basic content extraction without access restriction handling."""
        result = {
            'title': None,
            'content': None,
            'url': url,
            'error': None,
            'warning': None
        }

        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            result['title'] = self._extract_title(soup)
            result['content'] = self._extract_main_content(soup)

            if not result['content']:
                result['error'] = "No readable content found on the page"
                return result

            print_info(f"Extracted content: \"{result['title']}\" ({len(result['content'].split())} words)")

        except requests.exceptions.Timeout:
            result['error'] = "Request timed out"
        except requests.exceptions.ConnectionError:
            result['error'] = "Could not connect to the website"
        except requests.exceptions.HTTPError as e:
            result['error'] = f"HTTP error: {e}"
        except requests.exceptions.RequestException as e:
            result['error'] = f"Request failed: {e}"

        return result

    def _strip_markdown_json(self, text: str) -> str:
        """Strip markdown code block formatting from JSON responses."""
        # Remove ```json and ``` markers
        text = text.strip()
        if text.startswith('```json'):
            text = text[7:]  # Remove ```json
        elif text.startswith('```'):
            text = text[3:]   # Remove ```
        if text.endswith('```'):
            text = text[:-3]  # Remove trailing ```
        return text.strip()

    def _llm_evaluate_content(self, content: Optional[str]) -> Optional[str]:
        """Use LLM to evaluate if content is complete or blocked by restrictions."""
        if not content:
            return None

        try:
            # Get current model from settings
            current_model = self.settings_manager.setting_get("model")

            prompt = """Is this web content blocked by a paywall or access restriction?

Content to analyze:
{content}

Think like a human reader: Does this feel complete or are you being stopped and asked to pay/login?

Common signs of blocked content:
- Content ends abruptly mid-sentence
- Subscription prompts like "Subscribe to continue"
- Very short content that feels incomplete
- Login requirements

Respond in JSON format only:
{{
  "blocked": true or false
}}""".format(content=content)

            messages = [{"role": "user", "content": prompt}]

            response = self.llm_client_manager.create_chat_completion(
                model=current_model,
                messages=messages,
                temperature=0.1,
                max_tokens=100
            )

            response_text = response.choices[0].message.content.strip()

            # Strip markdown formatting and parse JSON
            clean_json = self._strip_markdown_json(response_text)

            # Parse JSON response - strict validation, no fallbacks
            try:
                result = json.loads(clean_json)
                if not isinstance(result, dict):
                    print_info(f"ERROR: LLM returned invalid JSON structure (not a dictionary): {response_text}")
                    return None

                if result.get("blocked") == True:
                    return "blocked"
                else:
                    return None  # Content is complete
            except json.JSONDecodeError:
                print_info(f"ERROR: LLM returned invalid JSON for content evaluation: {response_text}")
                print_info("Cannot proceed with content evaluation - LLM must return valid JSON")
                return None

        except Exception as e:
            print_info(f"LLM evaluation failed, falling back to basic checks: {str(e)}")
            # Simple fallback - very short content is likely incomplete
            if len(content.split()) < 30:
                return "insufficient content"
            return None

    def _llm_evaluate_bypass(self, original_content: Optional[str], bypass_content: Optional[str]) -> bool:
        """Use LLM to compare original and bypass content to determine if bypass was successful."""
        if not bypass_content:
            return False
        if not original_content:
            original_content = ""

        try:
            # Get current model from settings
            current_model = self.settings_manager.setting_get("model")

            prompt = """Is this web content blocked by a paywall or access restriction?

Content to analyze:
{content}

Think like a human reader: Does this feel complete or are you being stopped and asked to pay/login?

Respond in JSON format only:
{{
  "blocked": true or false
}}""".format(content=bypass_content)

            messages = [{"role": "user", "content": prompt}]

            response = self.llm_client_manager.create_chat_completion(
                model=current_model,
                messages=messages,
                temperature=0.1,
                max_tokens=50
            )

            response_text = response.choices[0].message.content.strip()

            # Strip markdown formatting if present
            clean_json = self._strip_markdown_json(response_text)

            # Parse JSON response - strict validation, no fallbacks
            try:
                result = json.loads(clean_json)
                if not isinstance(result, dict):
                    print_info(f"ERROR: LLM returned invalid JSON structure for bypass evaluation: {response_text}")
                    return False
                return not result.get("blocked", True)  # If not blocked, bypass was successful
            except json.JSONDecodeError:
                print_info(f"ERROR: LLM returned invalid JSON for bypass evaluation: {response_text}")
                print_info("Cannot proceed with bypass evaluation - LLM must return valid JSON")
                return False

        except Exception as e:
            print_info(f"LLM bypass evaluation failed: {str(e)}")
            # Fallback to simple length comparison
            if not original_content or not bypass_content:
                return bool(bypass_content and len(bypass_content.split()) > 50)
            original_words = len(original_content.split())
            bypass_words = len(bypass_content.split())
            return bypass_words > original_words * 1.5  # 50% more content suggests success

    def _try_access_bypass(self, url: str, original_content: str) -> Dict[str, Optional[str]]:
        """Try multiple methods to bypass access restrictions."""
        bypass_methods = [
            ("alternative user agents", self._try_bot_user_agent),
            ("print version URL", self._try_print_version),
            ("AMP version URL", self._try_amp_version),
            ("Archive.org (Wayback Machine)", self._try_archive_org)
        ]

        for method_name, method_func in bypass_methods:
            try:
                print_info(f"Attempting bypass using {method_name}...")
                result = method_func(url)

                # Check if we got content
                if result and result.get('content'):
                    # Use LLM to evaluate if bypass was successful
                    bypass_successful = self._llm_evaluate_bypass(original_content, result.get('content'))

                    if bypass_successful:
                        # Success!
                        specific_method = result.get('method', method_name)
                        content_length = len(result.get('content', '').split())
                        print_info(f"Success: Bypassed using {specific_method} - extracted {content_length} words")
                        result['warning'] = f"Access restriction bypassed using {specific_method}"
                        return result
                    else:
                        # Failed validation
                        print_info(f"Failed: {method_name} - bypass did not improve content")
                else:
                    print_info(f"Failed: {method_name} - no content retrieved")

            except Exception as e:
                print_info(f"Failed: {method_name} - {str(e)}")
                continue

        # All methods failed
        return {'content': None, 'error': 'All bypass methods failed'}



    def _try_archive_org(self, url: str) -> Dict[str, Optional[str]]:
        """Try to fetch content from Archive.org (Wayback Machine)."""
        # Generate timestamps dynamically based on current date
        now = datetime.now()
        timestamps = []

        # Add timestamps going back in time: current month, 1 month ago, 2 months ago, 6 months ago, 1 year ago, 2 years ago
        time_deltas = [0, 1, 2, 6, 12, 24]  # months back

        for months_back in time_deltas:
            target_date = now - timedelta(days=months_back * 30)  # Approximate months
            timestamp = target_date.strftime("%Y%m01000000")  # First of the month
            readable_date = target_date.strftime("%b %Y")
            timestamps.append((readable_date, timestamp))

        for time_desc, timestamp in timestamps:
            try:
                archive_url = f"https://web.archive.org/web/{timestamp}/{url}"

                # Use different session to avoid rate limiting conflicts
                archive_session = requests.Session()
                archive_session.headers.update({
                    'User-Agent': 'Mozilla/5.0 (compatible; Archive-Request/1.0)'
                })

                response = archive_session.get(archive_url, timeout=15)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')

                    # Remove archive.org specific elements
                    for element in soup(['script[src*="archive.org"]', '#wm-ipp-base', '.wb-autocomplete-suggestions']):
                        if element:
                            element.decompose()

                    title = self._extract_title(soup)
                    content = self._extract_main_content(soup)

                    if content:
                        return {
                            'title': title,
                            'content': content,
                            'url': url,
                            'error': None,
                            'warning': None,
                            'method': f"Archive.org snapshot from {time_desc}"
                        }

                time.sleep(0.5)  # Be respectful to archive.org

            except Exception:
                continue

        return {'content': None, 'error': 'No usable Archive.org snapshots found'}

    def _try_bot_user_agent(self, url: str) -> Dict[str, Optional[str]]:
        """Try with various user agents including search engine bots and realistic browsers."""
        user_agent_configs = [
            # Social media crawlers (try first - high success rate)
            ("Facebook crawler", {
                'User-Agent': 'facebookexternalhit/1.1 (+http://www.facebook.com/externalhit_uatext.php)',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive'
            }),
            # Search engine bots
            ("Googlebot", {
                'User-Agent': 'Googlebot/2.1 (+http://www.google.com/bot.html)',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive'
            }),
            ("Bingbot", {
                'User-Agent': 'Mozilla/5.0 (compatible; Bingbot/2.0; +http://www.bing.com/bingbot.htm)',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive'
            }),
            # Realistic browser user agents
            ("Chrome Windows", {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Cache-Control': 'max-age=0'
            }),
            ("Chrome macOS", {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1'
            })
        ]

        for agent_name, headers in user_agent_configs:
            try:
                bot_session = requests.Session()
                bot_session.headers.update(headers)

                response = bot_session.get(url, timeout=self.timeout)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, 'html.parser')
                title = self._extract_title(soup)
                content = self._extract_main_content(soup)

                if content:
                    return {
                        'title': title,
                        'content': content,
                        'url': url,
                        'error': None,
                        'warning': None,
                        'method': agent_name
                    }

            except Exception:
                continue

        return {'content': None, 'error': 'All user agents failed'}

    def _try_print_version(self, url: str) -> Dict[str, Optional[str]]:
        """Try to access a print version of the page."""
        print_variations = [
            ("?print=1", f"{url}?print=1"),
            ("?print=true", f"{url}?print=true"),
            ("/print", f"{url}/print"),
            ("/print/", f"{url.rstrip('/')}/print/"),
            ("?view=print", f"{url}?view=print"),
            ("?format=print", f"{url}?format=print")
        ]

        for variation_desc, print_url in print_variations:
            try:
                response = self.session.get(print_url, timeout=self.timeout)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, 'html.parser')
                title = self._extract_title(soup)
                content = self._extract_main_content(soup)

                if content:
                    return {
                        'title': title,
                        'content': content,
                        'url': url,
                        'error': None,
                        'warning': None,
                        'method': f"print version ({variation_desc})"
                    }

            except Exception:
                continue

        return {'content': None, 'error': 'No working print version found'}

    def _try_amp_version(self, url: str) -> Dict[str, Optional[str]]:
        """Try AMP (Accelerated Mobile Pages) version."""
        parsed = urlparse(url)

        amp_variations = [
            ("amp subdomain", url.replace('www.', 'amp.')),
            ("amp prefix", url.replace('https://', 'https://amp.')),
            ("/amp path", f"{parsed.scheme}://{parsed.netloc}/amp{parsed.path}"),
            ("/amp suffix", f"{url.rstrip('/')}/amp"),
            ("?amp=1 parameter", f"{url}?amp=1")
        ]

        for variation_desc, amp_url in amp_variations:
            try:
                response = self.session.get(amp_url, timeout=self.timeout)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, 'html.parser')

                # Remove AMP-specific elements
                for element in soup(['amp-ad', 'amp-analytics', 'amp-sidebar']):
                    if element:
                        element.decompose()

                title = self._extract_title(soup)
                content = self._extract_main_content(soup)

                if content:
                    return {
                        'title': title,
                        'content': content,
                        'url': url,
                        'error': None,
                        'warning': None,
                        'method': f"AMP version ({variation_desc})"
                    }

            except Exception:
                continue

        return {'content': None, 'error': 'No working AMP version found'}