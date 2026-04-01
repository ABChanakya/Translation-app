#!/usr/bin/env python3

"""
Rawkuma manga scraper – downloads chapter ZIP archives via direct dl.rawkuma.com links,
with fallback to per-page image scraping when ZIPs aren’t available.

Usage
-----
python rawkuma_scraper.py "https://rawkuma.net/manga/.../" --all
python rawkuma_scraper.py URL --only-latest
python rawkuma_scraper.py URL --chapter "Chapter 30"

Dependencies:
    pip install requests beautifulsoup4 tqdm

Notes:
    • Rawkuma provides ZIP archives behind a download button that points to dl.rawkuma.com/?id=XXXX.
    • This script attempts to download the ZIP, retries on failure, then falls back to grabbing individual pages.
    • Outputs ZIPs named 001.zip, 002.zip, etc., and image folders named 001_<ChapterName>/ if needed.
    • Respect copyright: personal use only and delete archives after viewing.
"""

import os
import argparse
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse, parse_qs

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Persistent session with polite headers
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
})

DOWNLOAD_BASE = "https://dl.rawkuma.com/?id={}"
MAX_RETRIES = 3
MIN_SIZE = 1024       # minimum acceptable size in bytes for a ZIP


def get_soup(url: str) -> BeautifulSoup:
    resp = SESSION.get(url, timeout=15)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")


def sanitize(name: str) -> str:
    return re.sub(r"[\\/*?:\"<>|]", "_", name.strip())


def list_chapters(series_url: str) -> list[tuple[str, str]]:
    soup = get_soup(series_url)
    links = soup.select("ul.main li a, div#chapter-list a, a[href*='chapter']")
    chapters, seen = [], set()
    for a in links:
        title = a.get_text(strip=True)
        href = urljoin(series_url, a.get('href', ''))
        if title and href not in seen:
            chapters.append((sanitize(title), href))
            seen.add(href)
    return chapters


def find_download_id(chapter_soup: BeautifulSoup) -> str | None:
    for a in chapter_soup.find_all('a', href=True):
        href = a['href']
        if 'dl.rawkuma.com' in href:
            qs = parse_qs(urlparse(href).query)
            if 'id' in qs:
                return qs['id'][0]
    return None


def fetch_zip_response(dl_url: str, referer: str) -> requests.Response:
    resp = SESSION.get(dl_url, headers={'Referer': referer}, timeout=15)
    # If HTML, try to locate real ZIP via meta-refresh or <a>
    ctype = resp.headers.get('Content-Type', '')
    if 'html' in ctype.lower():
        soup = BeautifulSoup(resp.text, 'lxml')
        meta = soup.find('meta', attrs={'http-equiv': re.compile('refresh', re.I)})
        if meta:
            m = re.search(r'url=([^;]+)', meta['content'], re.I)
            if m:
                next_full = urljoin(resp.url, m.group(1).strip().strip('"'))
                resp = SESSION.get(next_full, headers={'Referer': dl_url}, stream=True, timeout=30)
                resp.raise_for_status()
                return resp
        link = soup.find('a', href=re.compile(r'\.zip$|\.cbz$', re.I))
        if link:
            next_full = urljoin(resp.url, link['href'])
            resp = SESSION.get(next_full, headers={'Referer': dl_url}, stream=True, timeout=30)
            resp.raise_for_status()
            return resp
        raise RuntimeError('Expected ZIP but got HTML at ' + dl_url)
    # Assume direct ZIP
    resp = SESSION.get(dl_url, headers={'Referer': referer}, stream=True, timeout=30)
    resp.raise_for_status()
    return resp


def download_file(dl_url: str, referer: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = fetch_zip_response(dl_url, referer)
            total = int(resp.headers.get('content-length', 0)) or None
            downloaded = 0
            with tqdm(total=total, unit='B', unit_scale=True, desc=dest.name) as bar, open(dest, 'wb') as f:
                for chunk in resp.iter_content(8192):
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    bar.update(len(chunk))
            if total and downloaded < MIN_SIZE:
                print(f"  [!] Download too small ({downloaded} bytes), retry {attempt}/{MAX_RETRIES}")
            else:
                return True
        except Exception as e:
            print(f"  [!] Attempt {attempt} failed: {e}")
        time.sleep(1)
    return False


def download_chapter_images(chapter_soup: BeautifulSoup, chapter_dir: Path):
    chapter_dir.mkdir(parents=True, exist_ok=True)
    imgs = chapter_soup.select("#readerarea img[src], #readerarea img[data-src]")
    if not imgs:
        print("  [!] No images found for fallback.")
        return
    for i, img in enumerate(imgs, 1):
        src = img.get('data-src') or img['src']
        full = urljoin(chapter_soup.base_url or '', src)
        ext = os.path.splitext(urlparse(full).path)[1] or '.jpg'
        fname = chapter_dir / f"{i:03}{ext}"
        print(f"  Fallback download page {i}...")
        download_file(full, referer=chapter_soup.base_url or '', dest=fname)
        time.sleep(1)


def process_chapter(index: int, name: str, url: str, out_dir: Path):
    print(f"\nProcessing {name} (#{index:03})")
    soup = get_soup(url)
    dl_id = find_download_id(soup)
    if not dl_id:
        print(f"  [!] No download ID found for {name}, skipping ZIP...")
        return

    dl_url = DOWNLOAD_BASE.format(dl_id)
    zip_path = out_dir / f"{index:03}.zip"
    if zip_path.exists():
        print(f"  Already have {zip_path.name}")
    else:
        print(f"  Downloading {zip_path.name} from {dl_url}...")
        ok = download_file(dl_url, referer=url, dest=zip_path)
        if not ok:
            print(f"  [!] ZIP failed; falling back to images for {name}")
            img_dir = out_dir / f"{index:03}_{sanitize(name)}"
            download_chapter_images(soup, img_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Download Rawkuma chapter ZIPs with image fallback, sequentially numbered."
    )
    parser.add_argument('url', help='Series URL')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--all', action='store_true', help='All chapters')
    group.add_argument('--only-latest', action='store_true', help='Only newest')
    group.add_argument('--chapter', help='Single chapter by substring')
    parser.add_argument(
        '-o', '--output',
        default='/home/chanakya/chanakya/UNI/translation_tool/data',
        help='Output folder'
    )
    args = parser.parse_args()

    chapters = list_chapters(args.url)
    if not chapters:
        sys.exit("No chapters found; check the URL.")

    if args.only_latest:
        picks = [chapters[0]]
    elif args.all:
        picks = chapters
    else:
        key = args.chapter.lower()
        picks = [c for c in chapters if key in c[0].lower()]
        if not picks:
            sys.exit(f'Chapter matching "{args.chapter}" not found.')

    out_dir = Path(args.output)
    for idx, (name, link) in enumerate(picks, start=1):
        process_chapter(idx, name, link, out_dir)

    print("\nDone.")


if __name__ == '__main__':
    main()
