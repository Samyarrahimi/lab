"""
NVIDIA Isaac Sim Documentation Crawler and PDF Generator.

Crawls the Isaac Sim 5.1.0 documentation site, discovers all pages via BFS,
and renders each page to a separate PDF using Playwright headless Chromium.

Usage:
    python main.py
    python main.py --output-dir ./my_pdfs --delay 2.0
    python main.py --max-pages 5          # test with a few pages
    python main.py --fresh                 # ignore saved progress
"""

import argparse
import asyncio
import json
import logging
import os
import re
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
BASE_URL = "https://docs.isaacsim.omniverse.nvidia.com/5.1.0"
OUTPUT_DIR = Path("output_pdfs")
PROGRESS_FILE = Path("progress.json")
MANIFEST_FILE = "manifest.json"
REQUEST_DELAY = 1.5  # seconds between HTTP crawl requests
PDF_DELAY = 2.0  # seconds between PDF renders
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30  # seconds
PDF_TIMEOUT = 60_000  # milliseconds for Playwright
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

logger = logging.getLogger("isaacsim_docs_crawler")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging() -> logging.Logger:
    """Configure logging to console (INFO) and file (DEBUG)."""
    root = logging.getLogger("isaacsim_docs_crawler")
    root.setLevel(logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(console)

    file_handler = logging.FileHandler("crawler.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(file_handler)

    return root


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crawl NVIDIA Isaac Sim docs and save each page as PDF."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for generated PDFs (default: output_pdfs)",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=REQUEST_DELAY,
        help="Delay in seconds between crawl requests (default: 1.5)",
    )
    parser.add_argument(
        "--pdf-delay",
        type=float,
        default=PDF_DELAY,
        help="Delay in seconds between PDF renders (default: 2.0)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Limit number of pages to process (for testing)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing progress and start fresh",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# ProgressTracker — resume capability
# ---------------------------------------------------------------------------
class ProgressTracker:
    """Persists crawl and conversion state so the script can resume."""

    def __init__(self, progress_file: Path):
        self.progress_file = progress_file
        self.discovered_urls: set[str] = set()
        self.crawled_urls: set[str] = set()
        self.converted_urls: set[str] = set()
        self.failed_urls: dict[str, str] = {}
        self.load()

    # -- persistence ---------------------------------------------------------

    def load(self) -> None:
        if not self.progress_file.exists():
            return
        try:
            with open(self.progress_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.discovered_urls = set(data.get("discovered_urls", []))
            self.crawled_urls = set(data.get("crawled_urls", []))
            self.converted_urls = set(data.get("converted_urls", []))
            self.failed_urls = dict(data.get("failed_urls", {}))
            logger.info(
                "Resumed progress: %d discovered, %d crawled, %d converted, %d failed",
                len(self.discovered_urls),
                len(self.crawled_urls),
                len(self.converted_urls),
                len(self.failed_urls),
            )
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load progress file: %s", exc)

    def save(self) -> None:
        data = {
            "discovered_urls": sorted(self.discovered_urls),
            "crawled_urls": sorted(self.crawled_urls),
            "converted_urls": sorted(self.converted_urls),
            "failed_urls": self.failed_urls,
        }
        tmp = self.progress_file.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(str(tmp), str(self.progress_file))

    # -- state transitions ---------------------------------------------------

    def mark_crawled(self, url: str) -> None:
        self.crawled_urls.add(url)
        self.save()

    def mark_converted(self, url: str) -> None:
        self.converted_urls.add(url)
        self.failed_urls.pop(url, None)
        self.save()

    def mark_failed(self, url: str, error: str) -> None:
        self.failed_urls[url] = error
        self.save()

    def needs_crawling(self, url: str) -> bool:
        return url not in self.crawled_urls

    def needs_conversion(self, url: str) -> bool:
        return url not in self.converted_urls


# ---------------------------------------------------------------------------
# DocCrawler — BFS link discovery
# ---------------------------------------------------------------------------
class DocCrawler:
    """Discovers all documentation page URLs via breadth-first crawl."""

    EXCLUDED_PATTERNS = [
        "_static/",
        "_sources/",
        "_images/",
        "_downloads/",
        "genindex.html",
        "search.html",
        "py-modindex.html",
    ]

    def __init__(self, base_url: str, delay: float, tracker: ProgressTracker):
        self.base_url = base_url.rstrip("/")
        self.parsed_base = urlparse(self.base_url)
        self.delay = delay
        self.tracker = tracker
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "IsaacSimDocsCrawler/1.0 (+RAG-pipeline)"}
        )

    def is_valid_doc_url(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.netloc and parsed.netloc != self.parsed_base.netloc:
            return False
        if not parsed.path.startswith("/5.1.0/"):
            return False
        if not parsed.path.endswith(".html"):
            return False
        if parsed.query:
            return False
        for pat in self.EXCLUDED_PATTERNS:
            if pat in parsed.path:
                return False
        return True

    def normalize_url(self, href: str, current_url: str) -> str | None:
        href = href.split("#")[0]
        if not href:
            return None
        absolute = urljoin(current_url, href)
        if self.is_valid_doc_url(absolute):
            return absolute
        return None

    def extract_links(self, html: str, page_url: str) -> set[str]:
        soup = BeautifulSoup(html, "html.parser")
        links: set[str] = set()
        for a_tag in soup.find_all("a", href=True):
            norm = self.normalize_url(a_tag["href"], page_url)
            if norm:
                links.add(norm)
        return links

    def crawl(self) -> list[str]:
        """BFS crawl starting from the index page. Returns sorted URL list."""
        start = self.base_url + "/index.html"
        queue: deque[str] = deque()

        # Seed queue with discovered but not-yet-crawled URLs (for resume)
        if self.tracker.discovered_urls:
            for url in sorted(self.tracker.discovered_urls):
                if self.tracker.needs_crawling(url):
                    queue.append(url)
            logger.info("Resuming crawl with %d URLs in queue", len(queue))
        else:
            self.tracker.discovered_urls.add(start)
            queue.append(start)

        while queue:
            url = queue.popleft()
            if not self.tracker.needs_crawling(url):
                continue

            logger.info("Crawling: %s", url)
            try:
                resp = self.session.get(url, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response else 0
                if status == 404:
                    logger.info("404 — skipping %s", url)
                    self.tracker.mark_crawled(url)
                else:
                    logger.warning("HTTP %s for %s", status, url)
                    self.tracker.mark_failed(url, f"HTTP {status}")
                continue
            except requests.RequestException as exc:
                logger.warning("Request failed for %s: %s", url, exc)
                self.tracker.mark_failed(url, str(exc))
                continue

            new_links = self.extract_links(resp.text, url)
            for link in new_links:
                if link not in self.tracker.discovered_urls:
                    self.tracker.discovered_urls.add(link)
                    queue.append(link)

            self.tracker.mark_crawled(url)
            time.sleep(self.delay)

        all_urls = sorted(self.tracker.discovered_urls)
        logger.info("Discovery complete: %d pages found", len(all_urls))
        return all_urls


# ---------------------------------------------------------------------------
# PDFGenerator — headless Chromium rendering
# ---------------------------------------------------------------------------
class PDFGenerator:
    """Renders documentation pages to PDF via Playwright."""

    CLEANUP_CSS = """
    /* Hide navigation chrome */
    .bd-header,
    .bd-header-article,
    header,
    .navbar,
    .bd-sidebar-primary,
    .bd-sidebar-secondary,
    nav,
    .bd-footer,
    footer,
    .prev-next-footer,
    .breadcrumb,
    .pst-breadcrumb,
    .bd-toc,
    .search-button-field,
    .version-switcher,
    [role="navigation"],
    .ethical-ads,
    .announcement,
    .toctree-wrapper {
        display: none !important;
    }

    /* Expand content to full width */
    .bd-main .bd-content .bd-article-container,
    .bd-main .bd-content {
        max-width: 100% !important;
        margin: 0 !important;
        padding: 20px !important;
    }
    .bd-main {
        margin-left: 0 !important;
    }
    .bd-page-width {
        max-width: 100% !important;
    }
    body {
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Wrap long code lines for printing */
    pre {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
    }
    """

    CLEANUP_JS = """
    () => {
        // Remove nav/header/footer elements entirely
        const selectors = [
            'nav', 'header', 'footer',
            '.bd-sidebar-primary', '.bd-sidebar-secondary',
            '.bd-header', '.bd-footer',
            '.prev-next-footer', '.breadcrumb',
            '.version-switcher', '.announcement',
        ];
        selectors.forEach(sel => {
            document.querySelectorAll(sel).forEach(el => el.remove());
        });

        // Expand all collapsed <details> elements
        document.querySelectorAll('details').forEach(el => {
            el.setAttribute('open', '');
        });

        // Remove fixed/sticky positioning (avoids repeated headers in PDF)
        document.querySelectorAll('*').forEach(el => {
            const s = window.getComputedStyle(el);
            if (s.position === 'fixed' || s.position === 'sticky') {
                el.style.position = 'static';
            }
        });
    }
    """

    def __init__(self, output_dir: Path, delay: float, tracker: ProgressTracker):
        self.output_dir = output_dir
        self.delay = delay
        self.tracker = tracker
        self.manifest: dict[str, dict] = {}

    def url_to_filepath(self, url: str) -> Path:
        """Map a URL to a local PDF file path mirroring the URL hierarchy."""
        parsed = urlparse(url)
        # Strip the /5.1.0/ prefix
        rel = parsed.path.replace("/5.1.0/", "", 1)
        rel = rel.replace(".html", ".pdf")
        # Sanitise characters invalid on Windows
        rel = re.sub(r'[<>:"|?*]', "_", rel)
        return self.output_dir / rel

    async def render_pdf(self, page, url: str) -> Path | None:
        """Navigate to *url*, clean up the page, and save as PDF."""
        filepath = self.url_to_filepath(url)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                await page.goto(url, wait_until="networkidle", timeout=PDF_TIMEOUT)
                await page.add_style_tag(content=self.CLEANUP_CSS)
                await page.wait_for_timeout(1000)  # let lazy content settle
                await page.evaluate(self.CLEANUP_JS)

                await page.pdf(
                    path=str(filepath),
                    format="A4",
                    print_background=True,
                    margin={
                        "top": "20mm",
                        "bottom": "20mm",
                        "left": "15mm",
                        "right": "15mm",
                    },
                )
                logger.info("PDF saved: %s", filepath)
                return filepath

            except PlaywrightTimeout:
                logger.warning(
                    "Timeout rendering %s (attempt %d/%d)", url, attempt, MAX_RETRIES
                )
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(self.delay * 2)
            except Exception as exc:
                logger.error("Error rendering %s: %s", url, exc)
                break

        return None

    async def generate_all(self, urls: list[str]) -> None:
        """Launch browser and generate PDFs for all pending URLs."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        pending = [u for u in urls if self.tracker.needs_conversion(u)]
        logger.info(
            "PDFs to generate: %d of %d total (%d already done)",
            len(pending),
            len(urls),
            len(urls) - len(pending),
        )
        if not pending:
            logger.info("Nothing to do — all pages already converted.")
            self.save_manifest()
            return

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={"width": 1280, "height": 900},
                user_agent="IsaacSimDocsCrawler/1.0 (+RAG-pipeline)",
            )
            page = await context.new_page()

            for url in tqdm(pending, desc="Generating PDFs", unit="page"):
                result = await self.render_pdf(page, url)
                if result:
                    self.tracker.mark_converted(url)
                    rel_path = str(result.relative_to(self.output_dir))
                    self.manifest[rel_path] = {
                        "source_url": url,
                        "relative_path": rel_path,
                        "generated_at": datetime.now().isoformat(),
                    }
                else:
                    self.tracker.mark_failed(url, "PDF render failed")

                await asyncio.sleep(self.delay)

            await browser.close()

        self.save_manifest()

    def save_manifest(self) -> None:
        """Write manifest.json inside the output directory."""
        manifest_path = self.output_dir / MANIFEST_FILE
        data = {
            "metadata": {
                "base_url": BASE_URL,
                "generated_at": datetime.now().isoformat(),
                "total_pages": len(self.tracker.discovered_urls),
                "successful": len(self.tracker.converted_urls),
                "failed": len(self.tracker.failed_urls),
            },
            "pages": self.manifest,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Manifest written to %s", manifest_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def main() -> None:
    args = parse_args()
    setup_logging()

    logger.info("=== Isaac Sim Documentation Crawler ===")
    logger.info("Output dir : %s", args.output_dir)
    logger.info("Request delay: %.1fs | PDF delay: %.1fs", args.request_delay, args.pdf_delay)

    if args.fresh and PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
        logger.info("Fresh start — deleted existing progress file.")

    tracker = ProgressTracker(PROGRESS_FILE)

    # Phase 1: Discover all documentation page URLs
    logger.info("--- Phase 1: Discovering documentation pages ---")
    crawler = DocCrawler(BASE_URL, args.request_delay, tracker)
    all_urls = crawler.crawl()

    if args.max_pages:
        all_urls = all_urls[: args.max_pages]
        logger.info("Limited to %d pages (--max-pages)", args.max_pages)

    # Phase 2: Generate PDFs
    logger.info("--- Phase 2: Generating PDFs ---")
    generator = PDFGenerator(args.output_dir, args.pdf_delay, tracker)
    await generator.generate_all(all_urls)

    # Phase 3: Summary
    logger.info("--- Done ---")
    logger.info(
        "Discovered: %d | Converted: %d | Failed: %d",
        len(tracker.discovered_urls),
        len(tracker.converted_urls),
        len(tracker.failed_urls),
    )
    if tracker.failed_urls:
        logger.warning("Failed URLs:")
        for url, err in sorted(tracker.failed_urls.items()):
            logger.warning("  %s — %s", url, err)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Progress has been saved.")
