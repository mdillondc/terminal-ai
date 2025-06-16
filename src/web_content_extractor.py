import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import Dict, Optional, List, Tuple
import re
import time
from datetime import datetime, timedelta
from print_helper import print_info


class WebContentExtractor:
    """
    Extracts main content from web pages for AI analysis.
    Attempts various bypass methods when access is blocked.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36'
        })
        self.timeout = 30

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

                    bypass_result = self._try_access_bypass(url)
                    if bypass_result['content'] and len(bypass_result['content'].split()) > 100:
                        return bypass_result
                    else:
                        print_info("All bypass methods failed - returning original error")
                        return normal_result
                else:
                    return normal_result

            # Check for access restrictions in content
            block_type = self._is_access_blocked(normal_result['content'])
            if block_type:
                print_info(f"Access restriction detected ({block_type}) - attempting bypass methods...")
                bypass_result = self._try_access_bypass(url)
                if bypass_result['content'] and not self._is_access_blocked(bypass_result['content']):
                    return bypass_result
                else:
                    print_info("Content appears incomplete due to access restrictions - no bypass methods worked")
                    normal_result['warning'] = f"Content may be incomplete due to access restrictions ({block_type})"
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

    def _is_access_blocked(self, content: str) -> Optional[str]:
        """Detect common access blocking indicators. Returns block type or None."""
        if not content:
            return None

        content_lower = content.lower()
        total_words = len(content.split())

        # Paywall indicators
        paywall_indicators = [
            "subscribe to continue", "subscription required", "paywall",
            "premium content", "members only", "subscriber exclusive",
            "unlock this article", "start your subscription",
            "support ensures", "independent journalism", "uncompromising quality",
            "enduring impact", "bright future for independent journalism"
        ]

        # Login/registration wall indicators
        login_indicators = [
            "login to continue", "sign in to read more",
            "create account to continue", "please log in to continue",
            "create account to unlock", "register to continue"
        ]

        # Bot detection indicators
        bot_indicators = [
            "access denied", "blocked", "security check", "captcha",
            "unusual traffic", "automated requests", "bot detected",
            "verify you are human", "please verify", "security verification"
        ]

        # Geographic restriction indicators
        geo_indicators = [
            "not available in your region", "geographic restriction",
            "content not available", "region blocked", "location restricted"
        ]

        # Rate limiting indicators
        rate_indicators = [
            "too many requests", "rate limit", "slow down", "try again later",
            "exceeded limit"
        ]

        # Check for specific block types
        for indicator in paywall_indicators:
            if indicator in content_lower:
                return "paywall"

        for indicator in login_indicators:
            if indicator in content_lower:
                return "login required"

        for indicator in bot_indicators:
            if indicator in content_lower:
                return "bot detection"

        for indicator in geo_indicators:
            if indicator in content_lower:
                return "geographic restriction"

        for indicator in rate_indicators:
            if indicator in content_lower:
                return "rate limiting"

        # Pattern-based detection for complex blocking messages
        paywall_patterns = [
            # The Atlantic style - more specific patterns
            ("never miss a story", "free trial"),
            ("uncompromising quality", "independent journalism"),
            ("get started", "already have an account"),
            ("support ensures", "bright future"),
            # NYT style
            ("subscribe", "continue reading"),
            ("create account", "free articles"),
            # WSJ style
            ("subscriber", "exclusive"),
            # General patterns - must be more specific
            ("start your free trial", "sign in"),
            ("subscription", "unlimited access")
        ]

        # Check for pattern combinations - require both patterns to be present
        for pattern1, pattern2 in paywall_patterns:
            if pattern1 in content_lower and pattern2 in content_lower:
                return "paywall"

        # Weaker indicators (need multiple) - removed overly broad terms
        weak_indicators = [
            "subscribe", "subscription", "premium", "member",
            "sign in", "log in", "create account", "register",
            "free trial", "start trial"
        ]

        # For very short content, be more aggressive with detection
        if total_words < 50:
            weak_count = sum(1 for indicator in weak_indicators if indicator in content_lower)
            if weak_count >= 3:  # Require more evidence even for short content
                return "access restriction"
        else:
            # For longer content, need much more evidence
            weak_count = sum(1 for indicator in weak_indicators if indicator in content_lower)
            if weak_count >= 4:  # Raised threshold
                return "access restriction"

            # Check content length vs blocking text ratio - be more conservative
            blocking_words = sum(content_lower.count(indicator) for indicator in weak_indicators)
            if total_words < 150 and blocking_words > 8:  # More restrictive
                return "access restriction"

        return None

    def _try_access_bypass(self, url: str) -> Dict[str, Optional[str]]:
        """Try multiple methods to bypass access restrictions."""
        bypass_methods = [
            ("search engine bot user agent", self._try_bot_user_agent),
            ("print version URL", self._try_print_version),
            ("AMP version URL", self._try_amp_version),
            ("Archive.org (Wayback Machine)", self._try_archive_org)
        ]

        for method_name, method_func in bypass_methods:
            try:
                print_info(f"Attempting bypass using {method_name}...")
                result = method_func(url)
                if (result and result.get('content') and
                    not self._is_access_blocked(result['content']) and
                    len(result['content'].split()) > 100):  # Ensure substantial content
                    result['warning'] = f"Access restriction bypassed using {method_name}"
                    return result
                else:
                    if result and result.get('content'):
                        content_length = len(result['content'].split())
                        block_type = self._is_access_blocked(result['content'])
                        if block_type:
                            print_info(f"{method_name} failed - still blocked ({block_type})")
                        elif content_length <= 100:
                            print_info(f"{method_name} failed - insufficient content ({content_length} words)")
                        else:
                            print_info(f"{method_name} failed - unknown issue")
                    else:
                        print_info(f"{method_name} failed - no content retrieved")
            except Exception as e:
                print_info(f"{method_name} failed - {str(e)}")
                continue

        # All methods failed
        print_info("All bypass methods exhausted")
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
                print_info(f"Checking Archive.org snapshot from {time_desc}...")

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

                    if content and len(content.split()) > 100:
                        print_info(f"Found archived content from {time_desc} ({len(content.split())} words)")
                        return {
                            'title': title,
                            'content': content,
                            'url': url,
                            'error': None,
                            'warning': None
                        }
                    else:
                        print_info(f"{time_desc} snapshot has insufficient content")
                else:
                    print_info(f"No {time_desc} snapshot available (HTTP {response.status_code})")

                time.sleep(0.5)  # Be respectful to archive.org

            except Exception as e:
                print_info(f"{time_desc} snapshot failed: {str(e)}")
                continue

        return {'content': None, 'error': 'No usable Archive.org snapshots found'}

    def _try_bot_user_agent(self, url: str) -> Dict[str, Optional[str]]:
        """Try with various user agents including search engine bots and realistic browsers."""
        user_agent_configs = [
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
            }),
            # Social media crawlers
            ("Facebook crawler", {
                'User-Agent': 'facebookexternalhit/1.1 (+http://www.facebook.com/externalhit_uatext.php)',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive'
            })
        ]

        for agent_name, headers in user_agent_configs:
            try:
                print_info(f"Trying {agent_name} user agent...")
                bot_session = requests.Session()
                bot_session.headers.update(headers)

                response = bot_session.get(url, timeout=self.timeout)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, 'html.parser')
                title = self._extract_title(soup)
                content = self._extract_main_content(soup)

                if content and len(content.split()) > 100:
                    print_info(f"{agent_name} succeeded ({len(content.split())} words)")
                    return {
                        'title': title,
                        'content': content,
                        'url': url,
                        'error': None,
                        'warning': None
                    }
                else:
                    print_info(f"{agent_name} returned insufficient content")

            except Exception as e:
                print_info(f"{agent_name} failed: {str(e)}")
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
                print_info(f"Trying print URL with {variation_desc}...")
                response = self.session.get(print_url, timeout=self.timeout)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, 'html.parser')
                title = self._extract_title(soup)
                content = self._extract_main_content(soup)

                if content and len(content.split()) > 100:
                    print_info(f"Print version {variation_desc} succeeded ({len(content.split())} words)")
                    return {
                        'title': title,
                        'content': content,
                        'url': url,
                        'error': None,
                        'warning': None
                    }
                else:
                    print_info(f"Print version {variation_desc} returned insufficient content")

            except Exception as e:
                print_info(f"Print version {variation_desc} failed: {str(e)}")
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
                print_info(f"Trying AMP URL with {variation_desc}...")
                response = self.session.get(amp_url, timeout=self.timeout)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, 'html.parser')

                # Remove AMP-specific elements
                for element in soup(['amp-ad', 'amp-analytics', 'amp-sidebar']):
                    if element:
                        element.decompose()

                title = self._extract_title(soup)
                content = self._extract_main_content(soup)

                if content and len(content.split()) > 100:
                    print_info(f"AMP version {variation_desc} succeeded ({len(content.split())} words)")
                    return {
                        'title': title,
                        'content': content,
                        'url': url,
                        'error': None,
                        'warning': None
                    }
                else:
                    print_info(f"AMP version {variation_desc} returned insufficient content")

            except Exception as e:
                print_info(f"AMP version {variation_desc} failed: {str(e)}")
                continue

        return {'content': None, 'error': 'No working AMP version found'}