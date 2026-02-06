import os
import re
import math
import io
import base64
import datetime as dt
from collections import Counter

import requests
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

# Optional exports
try:
    import pandas as pd
except Exception:
    pd = None

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Zipf fit
try:
    import numpy as np
except Exception:
    np = None

# Wordcloud
try:
    from wordcloud import WordCloud
except Exception:
    WordCloud = None


# -----------------------------
# Configuration
# -----------------------------

GUARDIAN_API_KEY = os.getenv("GUARDIAN_API_KEY")
GUARDIAN_SEARCH_URL = "https://content.guardianapis.com/search"

DEFAULT_PAGE_SIZE = 200
MAX_PAGES = 10

WINDOW_BEFORE = 5
WINDOW_AFTER = 5


# -----------------------------
# NLTK (lazy, safe)
# -----------------------------

_NLTK_READY = False
_STOPWORDS = None
_LEMMATIZER = None


def ensure_nltk():
    global _NLTK_READY, _STOPWORDS, _LEMMATIZER
    if _NLTK_READY:
        return

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)

    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4", quiet=True)

    _STOPWORDS = set(stopwords.words("english"))
    _LEMMATIZER = WordNetLemmatizer()
    _NLTK_READY = True


def get_nlp():
    ensure_nltk()
    return _STOPWORDS, _LEMMATIZER


# -----------------------------
# Flask app
# -----------------------------

app = Flask(__name__, static_folder=None)
CORS(app)


# -----------------------------
# FRONTEND ROUTING (CRITICAL)
# -----------------------------

@app.route("/")
def serve_index():
    return send_from_directory("frontend", "index.html")


@app.route("/<path:path>")
def serve_frontend_assets(path):
    return send_from_directory("frontend", path)


# -----------------------------
# Health check
# -----------------------------

@app.get("/health")
def health():
    return jsonify({"status": "ok"})


# -----------------------------
# Helpers
# -----------------------------

def parse_month_selection(selection: str):
    m = re.search(r"\((\d+)\)", selection or "")
    offset = int(m.group(1)) if m else 1

    today = dt.date.today()
    first = today.replace(day=1)
    year = first.year
    month = first.month - offset

    while month <= 0:
        month += 12
        year -= 1

    start = dt.date(year, month, 1)
    end_month = month + 1
    end_year = year
    if end_month == 13:
        end_month = 1
        end_year += 1

    end = dt.date(end_year, end_month, 1)
    return start, end


def guardian_fetch_all(query, start, end):
    if not GUARDIAN_API_KEY:
        raise RuntimeError("GUARDIAN_API_KEY not set")

    page = 1
    results = []
    to_date = (end - dt.timedelta(days=1)).isoformat()

    while page <= MAX_PAGES:
        params = {
            "q": query,
            "api-key": GUARDIAN_API_KEY,
            "page-size": DEFAULT_PAGE_SIZE,
            "page": page,
            "from-date": start.isoformat(),
            "to-date": to_date,
            "show-fields": "bodyText,headline,trailText",
        }
        r = requests.get(GUARDIAN_SEARCH_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()["response"]

        results.extend(data["results"])
        if page >= data["pages"]:
            break
        page += 1

    return results


def normalize_text(text):
    STOPWORDS, LEMMATIZER = get_nlp()
    tokens = re.findall(r"[a-z]+", (text or "").lower())
    out = []
    for t in tokens:
        if len(t) <= 2 or t in STOPWORDS:
            continue
        l = LEMMATIZER.lemmatize(t)
        if len(l) > 2 and l not in STOPWORDS:
            out.append(l)
    return out


def aggregate_text(results):
    parts = []
    for r in results:
        f = r.get("fields") or {}
        parts.extend([f.get("headline", ""), f.get("trailText", ""), f.get("bodyText", "")])
    return "\n".join(parts)


def zipf_from_counter(freq, top_k=200):
    if not freq:
        return {"ranks": [], "freqs": [], "log_ranks": [], "log_freqs": [], "fit": None}

    items = freq.most_common(top_k)
    freqs = [c for _, c in items]
    ranks = list(range(1, len(freqs) + 1))

    if np is not None and len(freqs) >= 5:
        x = np.log(ranks)
        y = np.log(freqs)
        b, a = np.polyfit(x, y, 1)
        r2 = 1 - np.sum((y - (a + b * x)) ** 2) / np.sum((y - y.mean()) ** 2)
        return {
            "ranks": ranks,
            "freqs": freqs,
            "log_ranks": x.tolist(),
            "log_freqs": y.tolist(),
            "fit": {"intercept": float(a), "slope": float(b), "r2": float(r2)},
        }

    return {
        "ranks": ranks,
        "freqs": freqs,
        "log_ranks": [math.log(r) for r in ranks],
        "log_freqs": [math.log(f) for f in freqs],
        "fit": None,
    }


# -----------------------------
# API ROUTES (UNCHANGED)
# -----------------------------

@app.post("/simple_wordcount")
def simple_wordcount():
    payload = request.get_json(force=True)
    keyword = payload.get("keyword", "").strip()
    month = payload.get("month", "")

    if not keyword:
        return jsonify({"error": "Missing keyword"}), 400

    start, end = parse_month_selection(month)
    results = guardian_fetch_all(keyword, start, end)
    raw = aggregate_text(results)
    tokens = normalize_text(raw)
    freq = Counter(tokens)

    zipf = zipf_from_counter(freq)
    return jsonify({
        "keyword": keyword,
        "month": start.strftime("%Y-%m"),
        "articles": len(results),
        "top_words": [{"word": w, "count": c} for w, c in freq.most_common(50)],
        "zipf": zipf,
    })


@app.post("/trends")
def trends():
    payload = request.get_json(force=True)
    keyword = payload.get("keyword", "").strip()
    if not keyword:
        return jsonify({"error": "Missing keyword"}), 400

    ranges = [parse_month_selection(f"({i})") for i in range(1, 7)]
    months = []
    data = {}

    for i, (start, end) in enumerate(ranges):
        label = start.strftime("%Y-%m")
        months.append(label)
        results = guardian_fetch_all(keyword, start, end)
        tokens = normalize_text(aggregate_text(results))
        data[label] = Counter(tokens)

    return jsonify({
        "keyword": keyword,
        "months": months,
        "month_top_words": {
            m: [{"word": w, "count": c} for w, c in data[m].most_common(25)]
            for m in months
        }
    })


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
