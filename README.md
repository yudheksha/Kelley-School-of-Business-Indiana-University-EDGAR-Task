# SEC EDGAR Scraper (Python)

A small Python project that pulls **SEC EDGAR filings** and exports useful **filing data** for a company (e.g., 10-K / 10-Q / 8-K).

## What it does

- Fetches a company’s filings and exports **filing metadata** to CSV  
- Downloads the **latest filing** for a selected form type  
- Extracts and exports:
  - **Item/section text** (when available)
  - **XBRL facts** (when available)
- Optionally downloads and extracts **8-K press release exhibits** (EX-99.*)

## Tech used

- Python
- edgartools
- pandas
- BeautifulSoup / lxml (text extraction)
- tenacity (retries)
- pyarrow (Parquet export)

## Output (high level)

- Filing metadata CSVs  
- Latest filing content/summary files  
- Itemized section text files (if supported for that filing)  
- XBRL facts exported to CSV / Parquet (if available)  
- Optional folder for 8-K press release attachments + extracted text

## Notes

- Some filings won’t have itemized sections or XBRL available (depends on the filing).
- Be mindful of SEC access rules and rate limits when pulling data.
