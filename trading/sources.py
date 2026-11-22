"""Limited SEC EDGAR collection for public issuer disclosures."""
from html.parser import HTMLParser
import re
import time
from urllib.parse import quote

import requests

from .events import Source, now

FORMS = {"8-K", "10-Q", "10-K", "6-K", "20-F"}


class FilingText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "ix:hidden", "head"}:
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style", "ix:hidden", "head"} and self.hidden:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden and data.strip():
            self.parts.append(data.strip())


def sec_sources(symbol: str, cik: str, contact: str, limit: int = 3, session=None) -> list[Source]:
    if not re.fullmatch(r"[0-9]{1,10}", cik) or not 1 <= limit <= 10 or "@" not in contact:
        raise ValueError("Provide a numeric CIK, a descriptive SEC contact with email, and limit 1-10")
    session = requests.Session() if session is None else session
    headers = {"User-Agent": contact}
    def get(url: str):
        response = session.get(url, headers=headers, timeout=(5, 20), allow_redirects=False)
        if response.status_code != 200:
            raise ValueError(f"SEC data request returned HTTP {response.status_code}")
        if len(response.content) > 8_000_000:
            raise ValueError("SEC response exceeds the documented collector size limit")
        return response

    listing = get(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json").json()
    if symbol not in listing.get("tickers", ()):
        raise ValueError("Ticker does not match the SEC CIK mapping")
    recent = listing["filings"]["recent"]
    result = []
    rows = zip(*(recent[field] for field in (
        "form", "accessionNumber", "primaryDocument", "acceptanceDateTime")), strict=True)
    for form, accession, document, published in rows:
        if (form not in FORMS or not re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession)
                or not re.fullmatch(r"[\w.-]+", document) or document in {".", ".."} or not published):
            continue
        time.sleep(.25)
        url = (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
               f"{accession.replace('-', '')}/{quote(document)}")
        parser = FilingText()
        parser.feed(get(url).text)
        excerpt = " ".join(" ".join(parser.parts).split())[:12000]
        if excerpt:
            result.append(Source(symbol, "sec_filing", url, published, now().isoformat(),
                "Public EDGAR issuer disclosure; operator must verify excerpt reuse/processing rights.", excerpt))
        if len(result) >= limit:
            break
    return result
