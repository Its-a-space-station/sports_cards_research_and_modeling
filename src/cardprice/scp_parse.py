# src/cardprice/scp_parse.py
"""Parse SportsCardsPro card pages: sales tables, attributes, price summary, chart data."""

import json
import re

import pandas as pd
from bs4 import BeautifulSoup, Tag

GRADE_RES = [
    # grader first: "PSA 10", "SGC-10", "PSA GEM MT 10", "PSA MINT 9", "SGC PERFECT 10"
    re.compile(
        r"\b(PSA|BGS|SGC|CGC)\s*-?\s*"
        r"(?:GEM\s*(?:MT|MINT)\s*|MINT\s+|PERFECT\s+|PRISTINE\s+)?"
        r"(10|9\.5|9|8\.5|8)\b",
        re.IGNORECASE,
    ),
    # grade first: "Gem Mint 10 PSA"
    re.compile(r"\bGEM\s*(?:MT|MINT)\s*(10)\s+(PSA|BGS|SGC|CGC)\b", re.IGNORECASE),
]
PRICE_RE = re.compile(r"\$([\d,]+(?:\.\d{2})?)")
CHART_RE = re.compile(r"VGPC\.chart_data\s*=\s*(\{.*?\})\s*;", re.DOTALL)


def parse_grade_from_title(title: str) -> str | None:
    m = GRADE_RES[0].search(title)
    if m:
        return f"{m.group(1).lower()}_{m.group(2)}"
    m = GRADE_RES[1].search(title)
    if m:
        return f"{m.group(2).lower()}_{m.group(1)}"
    return None


def _amount(text: str) -> float:
    m = PRICE_RE.search(text)
    if not m:
        raise ValueError(f"no price in text: {text!r}")
    return float(m.group(1).replace(",", ""))


def _parse_price_cell(td: Tag) -> tuple[float, float | None, bool]:
    """Return (price_paid, list_price, best_offer) for a numeric price cell.

    A Best-Offer sale renders two spans: the accepted price first
    (title="best offer accepted price") and the struck-through list price
    second (class "listed-price-inline"); a plain sale has one bare
    "js-price" span. Key on the span markers per the spike doc, falling back
    to DOM order (accepted first, list second).
    """
    spans = td.find_all("span", class_="js-price")
    accepted = next((s for s in spans if s.get("title") == "best offer accepted price"), None)
    listed = next((s for s in spans if "listed-price-inline" in (s.get("class") or [])), None)
    if accepted is not None:
        return (
            _amount(accepted.get_text()),
            _amount(listed.get_text()) if listed else None,
            listed is not None,
        )
    amounts = [float(a.replace(",", "")) for a in PRICE_RE.findall(td.get_text(" ", strip=True))]
    if not amounts:
        raise ValueError(f"no price in cell: {td.get_text(' ', strip=True)!r}")
    if len(amounts) == 1:
        return amounts[0], None, False
    return amounts[0], amounts[1], True


def parse_sales_tables(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for div in soup.find_all("div", class_=re.compile(r"^completed-auctions-")):
        table = div.find("table")
        if not table:
            continue  # tab-bar button, not a sales-table container
        bucket = next(c for c in div["class"] if c.startswith("completed-auctions-")).replace(
            "completed-auctions-", ""
        )
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) < 4:
                continue
            date_text = tds[0].get_text(strip=True)
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_text):
                continue  # header or non-sale row
            title = tds[2].get_text(strip=True)
            price, list_price, best_offer = _parse_price_cell(tds[3])
            rows.append(
                {
                    "sale_date": date_text,
                    "title": title,
                    "price": price,
                    "list_price": list_price,
                    "best_offer": best_offer,
                    "grade": parse_grade_from_title(title),
                    "bucket": bucket,
                }
            )
    df = pd.DataFrame(rows)
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    return df


def parse_attributes(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="attribute")
    out = {}
    if table:
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) == 2:
                key = tds[0].get_text(strip=True).rstrip(":").lower().replace(" ", "_")
                val = tds[1].get_text(strip=True)
                out[key] = val.lower() == "yes" if val.lower() in ("yes", "no") else val
    return out


def parse_price_summary(html: str) -> dict[str, float]:
    """table#price_data: pair the header row's grade labels with the first data
    row's value cells positionally. Value cells may hold two amounts
    ('$34.50$0.00' = price + week-change); take the first."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="price_data")
    out = {}
    if table:
        rows = table.find_all("tr")
        if len(rows) >= 2:
            labels = [c.get_text(strip=True) for c in rows[0].find_all(["th", "td"])]
            values = [c.get_text(strip=True) for c in rows[1].find_all(["th", "td"])]
            for label, val in zip(labels, values):
                amounts = PRICE_RE.findall(val)
                if label and amounts:
                    out[label] = float(amounts[0].replace(",", ""))
    return out


def parse_chart_data(html: str) -> dict[str, list[tuple[pd.Timestamp, float]]]:
    m = CHART_RE.search(html)
    if not m:
        return {}
    raw = json.loads(m.group(1))
    out = {}
    for key, points in raw.items():
        series = [
            (pd.Timestamp(ts, unit="ms", tz="UTC").tz_convert(None).normalize(), cents / 100)
            for ts, cents in points
        ]
        out[key] = sorted(series, key=lambda p: p[0])
    return out


def calibrate_chart_grades(html: str) -> pd.DataFrame:
    """Long-form monthly price history with grade labels, resolved by matching
    each chart key's latest price to the price-summary table."""
    summary = {v: k for k, v in parse_price_summary(html).items()}  # price -> label
    rows = []
    for key, series in parse_chart_data(html).items():
        if not series:
            continue
        label = summary.get(series[-1][1])
        grade = label.lower().replace(" ", "_") if label else f"key:{key}"
        for ts, price in series:
            if price <= 0:
                continue
            rows.append({"grade": grade, "date": ts, "price": price})
    return pd.DataFrame(rows)
