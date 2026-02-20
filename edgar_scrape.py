#!/usr/bin/env python3
"""
SEC EDGAR scraper using edgartools (based on the tutorial you pasted).

What this script can do:
1) Set SEC identity (required by SEC).
2) Pull a company's filings (e.g., 10-Q, 10-K, 8-K) and export metadata to CSV.
3) Download the latest 10-Q (or any form) and export:
   - itemized section text (e.g., "Item 2") to .txt files
   - XBRL facts to CSV/Parquet (if available)
4) Scan recent 8-Ks for press releases (EX-99.*), download, and extract readable text.
5) (Optional) Pull filings at scale by year/quarter using get_filings().

Install:
  pip install edgartools pandas pyarrow beautifulsoup4 lxml tenacity

Examples:
  python edgar_scrape.py --identity "Your Name your@email.com" --ticker AAPL --form 10-Q --limit 1
  python edgar_scrape.py --identity "Your Name your@email.com" --ticker AAPL --download-8k-press --limit 10
  python edgar_scrape.py --identity "Your Name your@email.com" --scale-form 10-K --year 2024 --quarter 1

Notes:
- SEC rate limits: keep it polite. Use --sleep to add a pause between requests.
- edgartools APIs can evolve; this script follows the tutorial’s API surface.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


# ---------- Helpers ----------

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(s: str, max_len: int = 140) -> str:
    s = re.sub(r"[^\w\-. ]+", "_", s).strip().replace(" ", "_")
    return s[:max_len] if len(s) > max_len else s


def extract_html_from_sec_document(raw: str) -> str:
    """
    Many SEC attachments come wrapped in a <DOCUMENT> ... <TEXT> ... </TEXT> container,
    where HTML is inside the <TEXT> section.

    We try:
    1) If <TEXT>...</TEXT> exists, take that.
    2) Else, return raw as-is.
    """
    m = re.search(r"<TEXT>\s*(.*)\s*</TEXT>", raw, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1)
    return raw


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    # drop script/style
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ---------- Retry wrapper for SEC calls ----------
# (Rate-limit blocks and transient network issues happen; retry helps.)

class SecTransientError(Exception):
    """Raise this to trigger retries on known transient conditions."""


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((SecTransientError, TimeoutError, ConnectionError)),
)
def with_retries(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        msg = str(e).lower()
        # crude heuristics for SEC throttling / temporary denial
        if any(k in msg for k in ["too many requests", "rate", "throttle", "temporarily", "denied", "blocked"]):
            raise SecTransientError(str(e)) from e
        raise


@dataclass
class RunContext:
    out_dir: Path
    sleep_s: float


# ---------- Core tasks ----------

def setup_edgar(identity: str):
    """
    Imports edgartools and sets identity.
    Kept inside a function so import errors show a clean message.
    """
    try:
        from edgar import set_identity  # edgartools style import (per tutorial)
    except Exception as e:
        raise RuntimeError(
            "Could not import edgartools. Install with: pip install edgartools"
        ) from e

    set_identity(identity)


def fetch_company(ticker_or_cik: str):
    try:
        from edgar import Company
    except Exception as e:
        raise RuntimeError(
            "Could not import Company from edgar. Check your edgartools installation."
        ) from e

    return with_retries(Company, ticker_or_cik)


def export_filings_metadata(filings, out_csv: Path) -> pd.DataFrame:
    """
    filings: object returned by Company.get_filings(...) or edgar.get_filings(...)
    """
    try:
        df = filings.to_pandas()
    except Exception:
        # fallback: try to coerce to DataFrame if possible
        df = pd.DataFrame(getattr(filings, "data", []))
    df.to_csv(out_csv, index=False)
    return df


def download_latest_filing_and_export(company, form: str, ctx: RunContext, items: Optional[list[str]] = None):
    print(f"📄 Fetching filings for {company} form={form} ...")
    filings = with_retries(company.get_filings, form=form)
    time.sleep(ctx.sleep_s)

    out_meta = ctx.out_dir / f"{safe_filename(str(company))}_{form}_filings.csv"
    df = export_filings_metadata(filings, out_meta)
    print(f"✅ Saved filings metadata: {out_meta} ({len(df)} rows)")

    latest = with_retries(filings.latest)  # latest filing object
    time.sleep(ctx.sleep_s)

    # Save a quick pointer to the latest filing
    latest_info_txt = ctx.out_dir / f"{safe_filename(str(company))}_{form}_latest.txt"
    latest_info_txt.write_text(str(latest), encoding="utf-8")
    print(f"✅ Saved latest filing summary: {latest_info_txt}")

    # Parse filing object (TenQ/TenK/EightK/etc.)
    obj = with_retries(latest.obj)
    time.sleep(ctx.sleep_s)

    # Export item text if available
    available_items = getattr(obj, "items", None)
    if available_items:
        print(f"🧩 Items available: {available_items}")
        target_items = items or list(available_items)

        items_dir = ensure_dir(ctx.out_dir / f"{form}_items")
        for it in target_items:
            try:
                text = obj[it]
            except Exception:
                # some objects may not support indexing by item
                continue
            (items_dir / f"{safe_filename(it)}.txt").write_text(str(text), encoding="utf-8")

        print(f"✅ Saved item text files to: {items_dir}")
    else:
        print("ℹ️ No .items found on this filing object (skipping item text export).")

    # Export XBRL facts if present
    try:
        xbrl = with_retries(latest.xbrl)
        time.sleep(ctx.sleep_s)

        facts_df = xbrl.facts.data  # pandas DataFrame per tutorial
        xbrl_csv = ctx.out_dir / f"{safe_filename(str(company))}_{form}_xbrl_facts.csv"
        facts_df.to_csv(xbrl_csv, index=False)

        # Optional Parquet (nice for big facts tables)
        xbrl_parquet = ctx.out_dir / f"{safe_filename(str(company))}_{form}_xbrl_facts.parquet"
        facts_df.to_parquet(xbrl_parquet, index=False)

        print(f"✅ Saved XBRL facts: {xbrl_csv}")
        print(f"✅ Saved XBRL facts (parquet): {xbrl_parquet}")
    except Exception as e:
        print(f"ℹ️ XBRL export skipped (not available or failed): {e}")


def download_8k_press_releases(company, ctx: RunContext, limit: int = 10):
    print(f"📰 Fetching recent 8-Ks for {company} ...")
    filings = with_retries(company.get_filings, form="8-K")
    time.sleep(ctx.sleep_s)

    # Get the most recent N 8-K filings and parse them
    recent = with_retries(filings.latest, limit)  # returns a list-like of filings per tutorial usage
    time.sleep(ctx.sleep_s)

    eightk_objs = []
    for f in recent:
        try:
            eightk_objs.append(with_retries(f.obj))
            time.sleep(ctx.sleep_s)
        except Exception as e:
            print(f"⚠️ Skipped one 8-K (parse error): {e}")

    with_pr = [x for x in eightk_objs if getattr(x, "has_press_release", False)]
    print(f"✅ Found {len(with_pr)} 8-K filings with press releases (out of {len(eightk_objs)} parsed)")

    pr_dir = ensure_dir(ctx.out_dir / "8k_press_releases")

    for idx, eightk in enumerate(with_pr, start=1):
        pr = getattr(eightk, "press_releases", None)
        if not pr:
            continue

        attachments = getattr(pr, "attachments", None)
        if not attachments:
            continue

        # attachments behaves like a list container in the tutorial
        # We'll iterate defensively.
        for a_i in range(0, 50):  # cap, just in case
            try:
                att = attachments.get(a_i)
            except Exception:
                break
            if att is None:
                break

            try:
                raw = with_retries(att.download)
                time.sleep(ctx.sleep_s)
            except Exception as e:
                print(f"⚠️ Could not download attachment {a_i}: {e}")
                continue

            raw_str = raw.decode("utf-8", errors="ignore") if isinstance(raw, (bytes, bytearray)) else str(raw)

            html = extract_html_from_sec_document(raw_str)
            text = html_to_text(html)

            base = f"press_release_{idx:02d}_att_{a_i:02d}"
            (pr_dir / f"{base}.raw.txt").write_text(raw_str, encoding="utf-8")
            (pr_dir / f"{base}.text.txt").write_text(text, encoding="utf-8")

            print(f"✅ Saved press release: {pr_dir / f'{base}.text.txt'}")

    print(f"📁 Press releases output folder: {pr_dir}")


def scrape_at_scale(ctx: RunContext, form: str, year: int, quarter: int):
    print(f"📦 Scraping at scale: form={form} year={year} quarter={quarter}")
    try:
        from edgar import get_filings
    except Exception as e:
        raise RuntimeError("Could not import get_filings from edgar.") from e

    filings = with_retries(get_filings, year=year, quarter=quarter, form=form)
    time.sleep(ctx.sleep_s)

    out_csv = ctx.out_dir / f"scale_{form}_{year}_Q{quarter}_filings.csv"
    df = export_filings_metadata(filings, out_csv)
    print(f"✅ Saved scale filings metadata: {out_csv} ({len(df)} rows)")

    # If you want to loop through and download each filing, you can,
    # but for big batches it’s easy to hit SEC limits. Keep it conservative.
    # This script stops at metadata export by default.


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scrape SEC EDGAR via edgartools.")
    p.add_argument("--identity", required=True, help='SEC identity, e.g. "Your Name your@email.com"')
    p.add_argument("--ticker", default="AAPL", help="Ticker or CIK (default: AAPL)")
    p.add_argument("--form", default="10-Q", help="Form type to pull for the company (default: 10-Q)")
    p.add_argument("--limit", type=int, default=10, help="How many recent filings to consider (default: 10)")
    p.add_argument("--out", default="edgar_output", help="Output directory (default: edgar_output)")
    p.add_argument("--sleep", type=float, default=0.5, help="Sleep seconds between SEC calls (default: 0.5)")

    p.add_argument(
        "--items",
        nargs="*",
        default=None,
        help='Optional item names to export, e.g. --items "Item 2" "Item 1A". If omitted, exports all available.',
    )

    p.add_argument(
        "--download-8k-press",
        action="store_true",
        help="If set, scan recent 8-K filings for press releases and download/extract them.",
    )

    # Scale mode
    p.add_argument("--scale-form", default=None, help="Run scale scraping with get_filings(form=...).")
    p.add_argument("--year", type=int, default=2024, help="Year for scale scraping (default: 2024)")
    p.add_argument("--quarter", type=int, default=1, help="Quarter for scale scraping (1-4, default: 1)")

    return p.parse_args()


def main() -> int:
    args = parse_args()
    ctx = RunContext(out_dir=ensure_dir(Path(args.out).resolve()), sleep_s=max(args.sleep, 0.0))

    # 1) Setup identity (required by SEC)
    setup_edgar(args.identity)
    print("✅ SEC identity set.")

    # 2) Company mode
    company = fetch_company(args.ticker)
    print(f"🏢 Company loaded: {company}")

    # Save a small “company snapshot” text file
    (ctx.out_dir / f"{safe_filename(args.ticker)}_company.txt").write_text(str(company), encoding="utf-8")

    # 3) Download latest filing + items + XBRL
    download_latest_filing_and_export(company, args.form, ctx, items=args.items)

    # 4) Optional: 8-K press releases
    if args.download_8k_press:
        download_8k_press_releases(company, ctx, limit=args.limit)

    # 5) Optional: scale scraping
    if args.scale_form:
        scrape_at_scale(ctx, form=args.scale_form, year=args.year, quarter=args.quarter)

    print(f"🎉 Done. Output folder: {ctx.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())