import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from typing import Dict, Optional
import re
import time
import json
import trafilatura
from datetime import datetime, timedelta
from print_helper import print_md
from settings_manager import SettingsManager
from llm_client_manager import LLMClientManager
from constants import LLMSettingConstants


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

        # JSON schema for GPT-5 structured outputs
        self.RESTRICTED_ACCESS_DETECTION_SCHEMA = {
            "type": "object",
            "properties": {
                "blocked": {
                    "type": "boolean",
                    "description": "Whether the content is blocked by paywall or access restriction"
                }
            },
            "required": ["blocked"],
            "additionalProperties": False
        }
        self.settings_manager = SettingsManager.getInstance()

    def extract_content(self, url: str, verbose: bool = True) -> Dict[str, Optional[str]]:
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
            if not self._is_valid_url(url, verbose):
                result['error'] = "Invalid URL format"
                return result

            # Try Trafilatura first with explicit timeout
            method_timeout = getattr(self.settings_manager, 'extraction_method_timeout_seconds', 10)
            if verbose:
                print_md(f"Trying Trafilatura ({method_timeout}s timeout)...")
            try:
                downloaded = self._run_with_timeout(trafilatura.fetch_url, method_timeout, url)
                if downloaded:
                    md_content = self._run_with_timeout(trafilatura.extract, method_timeout, downloaded, output_format="markdown")
                    if md_content:
                        if verbose:
                            word_count = len(md_content.split())
                            print_md(f"Extracted {word_count} words")
                        return {
                            'title': None,
                            'content': md_content,
                            'url': url,
                            'error': None,
                            'warning': None
                        }
            except Exception:
                # Ignore Trafilatura errors and fall back to legacy methods
                pass

            if verbose:
                print_md("Trafilatura failed, trying other methods...")

            # Try normal extraction first
            if verbose:
                method_timeout = getattr(self.settings_manager, 'extraction_method_timeout_seconds', 10)
                effective_timeout = min(self.timeout, method_timeout)
                print_md(f"Fetching webpage ({effective_timeout}s timeout)...")
            normal_result = self._basic_extraction(url, verbose)

            # Check if we got an error - attempt bypass for any fetch failure (timeout, HTTP errors, etc.)
            if normal_result['error']:
                error_msg = normal_result['error'].lower()
                if verbose:
                    status_code = normal_result.get('status_code')
                    if 'http error' in error_msg and status_code:
                        print_md(f"HTTP {status_code} — attempting bypass methods...")
                    elif 'timeout' in error_msg:
                        print_md("Request timed out — attempting bypass methods...")
                    elif 'could not connect' in error_msg or 'connect' in error_msg:
                        print_md("Connection error — attempting bypass methods...")
                    else:
                        print_md("Request failed — attempting bypass methods...")

                bypass_result = self._try_access_bypass(url, "", verbose)
                if bypass_result['content'] and len(bypass_result['content'].split()) > 100:
                    return bypass_result
                else:
                    if verbose:
                        print_md("All bypass methods failed - returning original error")
                    # Jina Reader last-resort fallback (opt-in due to privacy)
                    try:
                        current_model = self.settings_manager.setting_get("model")
                        provider = self.llm_client_manager.get_provider_for_model(current_model)
                        allow_ollama = self.settings_manager.setting_get("allow_jina_with_ollama")
                        if (provider != "ollama" or allow_ollama):
                            if verbose:
                                print_md("Attempting bypass using Jina Reader...")
                            jr_session = requests.Session()
                            method_timeout = getattr(self.settings_manager, 'extraction_method_timeout_seconds', 10)
                            jr_resp = jr_session.get(f"https://r.jina.ai/{url}", timeout=method_timeout)
                            if jr_resp.status_code == 200 and jr_resp.text and len(jr_resp.text.split()) > 50:
                                if verbose:
                                    print_md(f"Success: Bypassed using Jina Reader - extracted {len(jr_resp.text.split())} words")
                                return {
                                    'title': "Web Content",
                                    'content': jr_resp.text,
                                    'url': url,
                                    'error': None,
                                    'warning': "Content via Jina Reader"
                                }
                    except Exception:
                        pass
                    return normal_result

            # Check for access restrictions in content using LLM
            if normal_result['content']:
                is_blocked = self._llm_evaluate_content(normal_result['content'], verbose)
                if is_blocked:
                    if verbose:
                        print_md("Access restriction detected - attempting bypass methods...")
                    bypass_result = self._try_access_bypass(url, normal_result['content'], verbose)
                    if bypass_result['content']:
                        return bypass_result
                    else:
                        content_length = len(normal_result.get('content', '').split()) if normal_result.get('content') else 0
                        if verbose:
                            bypass_error_text = "All bypass methods failed - content appears blocked\n"
                            bypass_error_text += f"    Limited content ({content_length} words) added to context - may not be sufficient for analysis"
                            print_md(bypass_error_text)
                        normal_result['warning'] = "Content may be incomplete due to access restrictions"
                        normal_result['bypass_failed'] = True
                        # Jina Reader last-resort fallback (opt-in due to privacy)
                        try:
                            current_model = self.settings_manager.setting_get("model")
                            provider = self.llm_client_manager.get_provider_for_model(current_model)
                            allow_ollama = self.settings_manager.setting_get("allow_jina_with_ollama")
                            if (provider != "ollama" or allow_ollama):
                                if verbose:
                                    print_md("Attempting bypass using Jina Reader...")
                                jr_session = requests.Session()
                                method_timeout = getattr(self.settings_manager, 'extraction_method_timeout_seconds', 10)
                                jr_resp = jr_session.get(f"https://r.jina.ai/{url}", timeout=method_timeout)
                                if jr_resp.status_code == 200 and jr_resp.text and len(jr_resp.text.split()) > 50:
                                    if verbose:
                                        print_md(f"Success: Bypassed using Jina Reader - extracted {len(jr_resp.text.split())} words")
                                    return {
                                        'title': "Web Content",
                                        'content': jr_resp.text,
                                        'url': url,
                                        'error': None,
                                        'warning': "Content via Jina Reader"
                                    }
                        except Exception:
                            pass
                        return normal_result
            else:
                # No content to evaluate
                bypass_result = self._try_access_bypass(url, "", verbose)
                if bypass_result['content']:
                    return bypass_result
                else:
                    content_length = len(normal_result.get('content', '').split()) if normal_result.get('content') else 0
                    if verbose:
                        bypass_error_text = "All bypass methods failed - content appears blocked\n"
                        bypass_error_text += f"    Limited content ({content_length} words) added to context - may not be sufficient for analysis"
                        print_md(bypass_error_text)
                    normal_result['warning'] = "Content may be incomplete due to access restrictions"
                    normal_result['bypass_failed'] = True
                    # Jina Reader last-resort fallback (opt-in due to privacy)
                    try:
                        current_model = self.settings_manager.setting_get("model")
                        provider = self.llm_client_manager.get_provider_for_model(current_model)
                        allow_ollama = self.settings_manager.setting_get("allow_jina_with_ollama")
                        if (provider != "ollama" or allow_ollama):
                            if verbose:
                                print_md("Attempting bypass using Jina Reader...")
                            jr_session = requests.Session()
                            method_timeout = getattr(self.settings_manager, 'extraction_method_timeout_seconds', 10)
                            jr_resp = jr_session.get(f"https://r.jina.ai/{url}", timeout=method_timeout)
                            if jr_resp.status_code == 200 and jr_resp.text and len(jr_resp.text.split()) > 50:
                                if verbose:
                                    print_md(f"Success: Bypassed using Jina Reader - extracted {len(jr_resp.text.split())} words")
                                return {
                                    'title': "Web Content",
                                    'content': jr_resp.text,
                                    'url': url,
                                    'error': None,
                                    'warning': "Content via Jina Reader"
                                }
                    except Exception:
                        pass
                    return normal_result

            return normal_result

        except Exception as e:
            result['error'] = f"Unexpected error: {e}"
            return result

    def _is_valid_url(self, url: str, verbose: bool = True) -> bool:
        """Check if URL has a valid format."""
        try:
            parsed = urlparse(url)
            return bool(parsed.scheme and parsed.netloc)
        except (ValueError, AttributeError):
            # Invalid URL format
            return False
        except Exception as e:
            if verbose:
                print_md(f"Unexpected error validating URL: {e}")
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

    def _basic_extraction(self, url: str, verbose: bool = True) -> Dict[str, Optional[str]]:
        """Basic content extraction without access restriction handling."""
        result = {
            'title': None,
            'content': None,
            'url': url,
            'error': None,
            'warning': None
        }

        try:
            method_timeout = getattr(self.settings_manager, 'extraction_method_timeout_seconds', 10)
            effective_timeout = min(self.timeout, method_timeout)
            response = self.session.get(url, timeout=effective_timeout)
            response.raise_for_status()

            # PDF handling: detect and extract text via DocumentProcessor using a temp file
            content_type = response.headers.get('Content-Type', '').lower()
            if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
                try:
                    import tempfile
                    import os
                    from document_processor import DocumentProcessor

                    tmp_path = None
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
                    try:
                        tmp.write(response.content)
                        tmp.flush()
                        tmp_path = tmp.name
                    finally:
                        tmp.close()

                    dp = DocumentProcessor()
                    pdf_text = dp.load_pdf_file(tmp_path)
                finally:
                    try:
                        if tmp_path:
                            os.remove(tmp_path)
                    except Exception:
                        pass

                if pdf_text:
                    if verbose:
                        print_md(f"Extracted {len(pdf_text.split())} words")
                    result['title'] = "PDF Document"
                    result['content'] = pdf_text
                    return result

            soup = BeautifulSoup(response.text, 'html.parser')
            result['title'] = self._extract_title(soup)
            result['content'] = self._extract_main_content(soup)

            if not result['content']:
                result['error'] = "No readable content found on the page"
                return result

            if verbose:
                print_md(f"Extracted content: \"{result['title']}\" ({len(result['content'].split())} words)")

        except requests.exceptions.Timeout:
            result['error'] = "Request timed out"
        except requests.exceptions.ConnectionError:
            result['error'] = "Could not connect to the website"
        except requests.exceptions.HTTPError as e:
            result['error'] = f"HTTP error: {e}"
            try:
                result['status_code'] = e.response.status_code if e.response is not None else None
            except Exception:
                pass
        except requests.exceptions.RequestException as e:
            result['error'] = f"Request failed: {e}"

        return result

    def _run_with_timeout(self, func, timeout_seconds, *args, **kwargs):
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeout:
            raise TimeoutError(f"Operation timed out after {timeout_seconds}s")
        finally:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

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

    def _make_paywall_detection_call(self, prompt: str, max_tokens: int):
        """Make LLM call for paywall detection with appropriate parameters based on model type"""
        current_model = self.settings_manager.setting_get("model")
        messages = [{"role": "user", "content": prompt}]

        try:
            # Use different API parameters based on model type
            if LLMSettingConstants.is_gpt5_model(current_model):
                # GPT-5: Use structured outputs (temperature fixed at 1.0)
                return self.llm_client_manager.create_chat_completion(
                    model=current_model,
                    messages=messages,
                    max_tokens=max_tokens,  # Limits AI's response output (e.g., {"blocked": false}), NOT input content length
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "paywall_detection",
                            "schema": self.RESTRICTED_ACCESS_DETECTION_SCHEMA,
                            "strict": True
                        }
                    }
                )
            else:
                # Other models: Use traditional low temperature approach
                return self.llm_client_manager.create_chat_completion(
                    model=current_model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=max_tokens  # Limits AI's response output (e.g., {"blocked": false}), NOT input content length
                )
        except Exception as e:
            raise

    def _llm_evaluate_content(self, content: str, verbose: bool = True) -> Optional[str]:
        """Use LLM to evaluate if content is complete or blocked by restrictions."""
        if not content:
            return None

        try:
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

            response = self._make_paywall_detection_call(prompt, 999)

            response_text = response.choices[0].message.content.strip()

            # Strip markdown formatting and parse JSON
            clean_json = self._strip_markdown_json(response_text)

            # Parse JSON response - strict validation, no fallbacks
            try:
                result = json.loads(clean_json)
                if not isinstance(result, dict):
                    if verbose:
                        print_md(f"ERROR: LLM returned invalid JSON structure (not a dictionary): {response_text}")
                    return None

                if result.get("blocked") == True:
                    return "blocked"
                else:
                    return None  # Content is complete
            except json.JSONDecodeError:
                if verbose:
                    json_error_text = f"ERROR: LLM returned invalid JSON for content evaluation: {response_text}\n"
                    json_error_text += "    Cannot proceed with content evaluation - LLM must return valid JSON"
                    print_md(json_error_text)
                return None

        except Exception as e:
            if verbose:
                print_md(f"LLM evaluation failed, falling back to basic checks: {str(e)}")
            # Simple fallback - very short content is likely incomplete
            if len(content.split()) < 30:
                return "insufficient content"
            return None

    def _llm_evaluate_bypass(self, original_content: str, bypass_content: str, verbose: bool = True) -> bool:
        """Use LLM to compare original and bypass content to determine if bypass was successful."""
        if not bypass_content:
            return False
        if not original_content:
            original_content = ""

        try:
            prompt = """Is this web content blocked by a paywall or access restriction?

Content to analyze:
{content}

Think like a human reader: Does this feel complete or are you being stopped and asked to pay/login?

Respond in JSON format only:
{{
  "blocked": true or false
}}""".format(content=bypass_content)

            response = self._make_paywall_detection_call(prompt, 999)

            response_text = response.choices[0].message.content.strip()

            # Strip markdown formatting if present
            clean_json = self._strip_markdown_json(response_text)

            # Parse JSON response - strict validation, no fallbacks
            try:
                result = json.loads(clean_json)
                if not isinstance(result, dict):
                    if verbose:
                        print_md(f"ERROR: LLM returned invalid JSON structure for bypass evaluation: {response_text}")
                    return False
                return not result.get("blocked", True)  # If not blocked, bypass was successful
            except json.JSONDecodeError:
                if verbose:
                    bypass_json_error_text = f"ERROR: LLM returned invalid JSON for bypass evaluation: {response_text}\n"
                    bypass_json_error_text += "    Cannot proceed with bypass evaluation - LLM must return valid JSON"
                    print_md(bypass_json_error_text)
                return False

        except Exception as e:
            if verbose:
                print_md(f"LLM bypass evaluation failed: {str(e)}")
            # Fallback to simple length comparison
            if not original_content or not bypass_content:
                return bool(bypass_content and len(bypass_content.split()) > 50)
            original_words = len(original_content.split())
            bypass_words = len(bypass_content.split())
            return bypass_words > original_words * 1.5  # 50% more content suggests success

    def _try_access_bypass(self, url: str, original_content: str, verbose: bool = True) -> Dict[str, Optional[str]]:
        """Try multiple methods to bypass access restrictions."""
        bypass_methods = [
            ("alternative user agents", self._try_bot_user_agent),
            ("print version URL", self._try_print_version),
            ("AMP version URL", self._try_amp_version),
            ("Archive.org (Wayback Machine)", self._try_archive_org)
        ]

        for method_name, method_func in bypass_methods:
            try:
                if verbose:
                    method_timeout = getattr(self.settings_manager, 'extraction_method_timeout_seconds', 10)
                    if method_name == "alternative user agents":
                        print_md(f"Attempting bypass using {method_name} ({method_timeout}s timeout per agent)...")
                    else:
                        print_md(f"Attempting bypass using {method_name} ({method_timeout}s timeout)...")
                result = method_func(url, verbose)

                # Check if we got content
                if result and result.get('content'):
                    # Use LLM to evaluate if bypass was successful
                    bypass_successful = self._llm_evaluate_bypass(original_content, result.get('content'), verbose)

                    if bypass_successful:
                        # Success!
                        specific_method = result.get('method', method_name)
                        content_length = len(result.get('content', '').split())
                        if verbose:
                            print_md(f"Success: Bypassed using {specific_method} - extracted {content_length} words")
                        return result
                    else:
                        # Failed validation
                        if verbose:
                            print_md(f"Failed: {method_name} - bypass did not improve content")
                else:
                    if verbose:
                        print_md(f"Failed: {method_name} - no content retrieved")

            except Exception as e:
                if verbose:
                    print_md(f"Failed: {method_name} - {str(e)}")
                continue

        # All methods failed
        return {'content': None, 'error': 'All bypass methods failed'}



    def _try_jina_reader(self, url: str, verbose: bool = True) -> Dict[str, Optional[str]]:
        """Use Jina Reader to fetch Markdown as a bypass method (cloud providers by default)."""
        try:
            # Respect local privacy: skip when provider is Ollama unless explicitly allowed
            current_model = self.settings_manager.setting_get("model")
            provider = self.llm_client_manager.get_provider_for_model(current_model)
            allow_ollama = self.settings_manager.setting_get("allow_jina_with_ollama")
            if provider == "ollama" and not allow_ollama:
                return {'content': None, 'error': 'Jina disabled for Ollama'}

            jr_session = requests.Session()
            method_timeout = getattr(self.settings_manager, 'extraction_method_timeout_seconds', 10)
            jr_resp = jr_session.get(f"https://r.jina.ai/{url}", timeout=min(20, method_timeout))

            # Reject non-200s immediately
            if jr_resp.status_code != 200:
                return {'content': None, 'error': f'Jina Reader HTTP {jr_resp.status_code}'}

            # Detect JSON error payloads even with HTTP 200
            ct = (jr_resp.headers.get('Content-Type') or '').lower()
            body_text = jr_resp.text or ""
            if 'json' in ct or (body_text.lstrip().startswith('{') and body_text.rstrip().endswith('}')):
                try:
                    data = jr_resp.json()
                    if isinstance(data, dict):
                        # Common error fields from Jina: name/code/status/message/data
                        if data.get('name') or data.get('code') or data.get('status') or data.get('message'):
                            if verbose:
                                print_md("Jina Reader returned error JSON; treating as failure")
                            return {'content': None, 'error': 'Jina Reader JSON error'}
                except Exception:
                    # If JSON parsing fails, treat as non-markdown and continue checks below
                    pass

            # Treat body as Markdown only if it's sufficiently long and not JSON-looking
            if body_text.strip() and not body_text.lstrip().startswith('{'):
                return {
                    'title': "Web Content",
                    'content': body_text,
                    'url': url,
                    'error': None,
                    'warning': "Content via Jina Reader",
                    'method': "Jina Reader"
                }

            return {'content': None, 'error': 'Jina Reader failed'}
        except Exception:
            return {'content': None, 'error': 'Jina Reader exception'}

    def _try_archive_org(self, url: str, verbose: bool = True) -> Dict[str, Optional[str]]:
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

                response = archive_session.get(archive_url, timeout=min(15, getattr(self.settings_manager, 'extraction_method_timeout_seconds', 10)))
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

    def _try_bot_user_agent(self, url: str, verbose: bool = True) -> Dict[str, Optional[str]]:
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
                if verbose:
                    print_md(f"Trying user agent: {agent_name}")

                response = bot_session.get(url, timeout=min(self.timeout, getattr(self.settings_manager, 'extraction_method_timeout_seconds', 10)))
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

    def _try_print_version(self, url: str, verbose: bool = True) -> Dict[str, Optional[str]]:
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
                response = self.session.get(print_url, timeout=min(self.timeout, getattr(self.settings_manager, 'extraction_method_timeout_seconds', 10)))
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

    def _try_amp_version(self, url: str, verbose: bool = True) -> Dict[str, Optional[str]]:
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
                response = self.session.get(amp_url, timeout=min(self.timeout, getattr(self.settings_manager, 'extraction_method_timeout_seconds', 10)))
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