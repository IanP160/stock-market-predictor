from __future__ import annotations
import datetime as dt
import requests
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from .config import NEWSAPI_KEY

_analyzer = SentimentIntensityAnalyzer()

def _score_text(text: str) -> float:
    if not text:
        return 0.0
    return float(_analyzer.polarity_scores(text)["compound"])

def fetch_news_sentiment(
    query: str,
    start: dt.date,
    end: dt.date,
    language: str = "en",
    page_size: int = 100,
    max_pages: int = 5,
) -> pd.DataFrame:
    """Fetch NewsAPI articles and aggregate daily sentiment.

    Returns a DataFrame with columns: Date, sentiment
    """
    if not NEWSAPI_KEY:
        raise RuntimeError("NEWSAPI_KEY is not set. Export NEWSAPI_KEY in your environment.")

    url = "https://newsapi.org/v2/everything"
    headers = {"X-Api-Key": NEWSAPI_KEY}

    all_rows = []
    for page in range(1, max_pages + 1):
        params = {
            "q": query,
            "from": start.isoformat(),
            "to": end.isoformat(),
            "language": language,
            "sortBy": "relevancy",
            "pageSize": page_size,
            "page": page,
        }
        r = requests.get(url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        payload = r.json()
        articles = payload.get("articles", []) or []
        if not articles:
            break

        for a in articles:
            published = a.get("publishedAt") or ""
            try:
                d = pd.to_datetime(published).tz_convert(None).date()
            except Exception:
                continue
            title = a.get("title") or ""
            desc = a.get("description") or ""
            content = a.get("content") or ""
            text = " ".join([title, desc, content]).strip()
            all_rows.append({"Date": d, "score": _score_text(text)})

        # stop early if fewer than page_size returned
        if len(articles) < page_size:
            break

    if not all_rows:
        return pd.DataFrame({"Date": [], "sentiment": []})

    df = pd.DataFrame(all_rows)
    daily = df.groupby("Date", as_index=False)["score"].mean().rename(columns={"score": "sentiment"})
    daily["Date"] = pd.to_datetime(daily["Date"])
    return daily
