import os
import re
import math
import io
import base64
import datetime as dt
from collections import Counter

import requests
from flask import Flask, request, jsonify, send_file
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

# Wordcloud (optional but recommended)
try:
    from wordcloud import WordCloud
except Exception:
    WordCloud = None


# -----------------------------
# Configuration
# -----------------------------

GUARDIAN_API_KEY = os.getenv("GUARDIAN_API_KEY", "fb8f2423-edf4-4824-9ae5-0f4310802dae")
GUARDIAN_SEARCH_URL = "https://content.guardianapis.com/search"

DEFAULT_PAGE_SIZE = 200
MAX_PAGES = 10  # safeguard for API usage

WINDOW_BEFORE = 5
WINDOW_AFTER = 5


# -----------------------------
# NLTK setup (kept lightweight)
# -----------------------------

def ensure_nltk():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")
    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4")


ensure_nltk()
STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


# -----------------------------
# Flask App
# -----------------------------

app = Flask(__name__)

from flask import send_from_directory

@app.route("/")
def home():
    return send_from_directory("frontend", "index.html")

CORS(app)  # allow browser fetch from file:// or localhost


# -----------------------------
# Helpers
# -----------------------------

def parse_month_selection(selection: str):
    """
    Frontend typically sends something like:
      - "Last month (1)"
      - "2 months ago (2)"
      - "6 months ago (6)"
    Extract the trailing number in parentheses and use it as offset.
    Offset 1 = last completed month.
    """
    if not selection:
        offset = 1
    else:
        m = re.search(r"\((\d+)\)", selection)
        offset = int(m.group(1)) if m else 1

    today = dt.date.today()
    first_of_this_month = today.replace(day=1)

    year = first_of_this_month.year
    month = first_of_this_month.month - offset
    while month <= 0:
        month += 12
        year -= 1

    start = dt.date(year, month, 1)

    end_year = year
    end_month = month + 1
    if end_month == 13:
        end_month = 1
        end_year += 1
    end = dt.date(end_year, end_month, 1)
    return start, end


def month_ranges_last_n_completed_months(n: int):
    """
    Returns list of (month_label, start_date, end_date) for last n completed months,
    newest first.
    """
    ranges = []
    for i in range(1, n + 1):
        start, end = parse_month_selection(f"({i})")
        label = start.strftime("%Y-%m")
        ranges.append((label, start, end))
    return ranges  # newest -> older


def guardian_fetch_all(query: str, from_date: dt.date, to_date: dt.date):
    """
    Fetch all Guardian results between from_date (inclusive) and to_date (exclusive),
    returning list of results with fields.
    """
    if not GUARDIAN_API_KEY or GUARDIAN_API_KEY.strip() == "":
        raise ValueError("Missing GUARDIAN_API_KEY (set env var or hardcode).")

    all_results = []
    page = 1

    from_str = from_date.isoformat()
    # Guardian 'to-date' is inclusive; we approximate exclusive end with -1 day.
    to_inclusive = (to_date - dt.timedelta(days=1)).isoformat()

    while page <= MAX_PAGES:
        params = {
            "q": query,
            "api-key": GUARDIAN_API_KEY,
            "page-size": DEFAULT_PAGE_SIZE,
            "page": page,
            "from-date": from_str,
            "to-date": to_inclusive,
            "show-fields": "bodyText,headline,trailText",
        }
        r = requests.get(GUARDIAN_SEARCH_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        resp = data.get("response", {})
        results = resp.get("results", [])
        all_results.extend(results)

        pages = resp.get("pages", page)
        if page >= pages:
            break
        page += 1

    return all_results


def normalize_text(text: str) -> list[str]:
    """
    - lowercase
    - keep alphabetic tokens (a-z)
    - remove stopwords
    - lemmatize
    - remove 1 and 2 letter words
    """
    if not text:
        return []

    text = text.lower()
    tokens = re.findall(r"[a-z]+", text)

    cleaned = []
    for tok in tokens:
        if len(tok) <= 2:
            continue
        if tok in STOPWORDS:
            continue
        lemma = LEMMATIZER.lemmatize(tok)
        if len(lemma) <= 2:
            continue
        if lemma in STOPWORDS:
            continue
        cleaned.append(lemma)

    return cleaned


def aggregate_article_text(results) -> str:
    chunks = []
    for item in results:
        fields = item.get("fields") or {}
        body = fields.get("bodyText") or ""
        headline = fields.get("headline") or ""
        trail = fields.get("trailText") or ""
        chunks.append(headline)
        chunks.append(trail)
        chunks.append(body)
    return "\n".join(chunks)


def zipf_data_from_counter(freq: Counter, top_k: int = 200):
    """
    Returns ranks and frequencies (sorted), plus log-log fit line parameters.
    """
    if not freq:
        return {"ranks": [], "freqs": [], "log_ranks": [], "log_freqs": [], "fit": None}

    items = freq.most_common(top_k)
    freqs = [c for _, c in items]
    ranks = list(range(1, len(freqs) + 1))

    fit = None
    if np is not None and len(freqs) >= 5:
        x = np.log(np.array(ranks, dtype=float))
        y = np.log(np.array(freqs, dtype=float))
        b, a = np.polyfit(x, y, 1)  # y = b*x + a
        yhat = a + b * x
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
        r2 = 1.0 - (ss_res / ss_tot if ss_tot != 0 else 0.0)
        fit = {"intercept": float(a), "slope": float(b), "r2": float(r2)}
        log_ranks = x.tolist()
        log_freqs = y.tolist()
    else:
        log_ranks = [math.log(r) for r in ranks]
        log_freqs = [math.log(f) for f in freqs]

    return {
        "ranks": ranks,
        "freqs": freqs,
        "log_ranks": log_ranks,
        "log_freqs": log_freqs,
        "fit": fit,
    }


# -----------------------------
# MULTI-WORD context window counts
# -----------------------------

def context_window_counts(raw_text: str, search_term: str, before=5, after=5) -> Counter:
    """
    Combined context distribution: before + after words around each occurrence.
    Supports multi-word phrases by matching token sequences.
    Matching tokenization: [a-z]+ (simple, stable, same as before).
    Context is normalized via normalize_text().
    """
    if not raw_text or not search_term:
        return Counter()

    tokens = re.findall(r"[a-z]+", raw_text.lower())
    term_tokens = re.findall(r"[a-z]+", search_term.lower())
    if not term_tokens:
        return Counter()

    n = len(term_tokens)
    ctx = []
    i = 0
    while i <= len(tokens) - n:
        if tokens[i:i + n] == term_tokens:
            lo = max(0, i - before)
            hi = min(len(tokens), i + n + after)
            ctx.extend(tokens[lo:i] + tokens[i + n:hi])  # exclude the phrase itself
            i += n
        else:
            i += 1

    norm = normalize_text(" ".join(ctx))
    return Counter(norm)


def wordcloud_base64_from_counter(counter: Counter):
    """
    Returns base64 PNG if wordcloud is installed; else returns None.
    """
    if WordCloud is None or not counter:
        return None

    wc = WordCloud(width=900, height=450, background_color="white", collocations=False)
    wc.generate_from_frequencies(dict(counter))

    img = wc.to_image()
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def counter_to_top_list(counter: Counter, n=50):
    return [{"word": w, "count": int(c)} for w, c in counter.most_common(n)]


# -----------------------------
# Routes
# -----------------------------

@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/simple_wordcount")
def simple_wordcount():
    payload = request.get_json(force=True) or {}
    keyword = (payload.get("keyword") or "").strip()
    month_sel = (payload.get("month") or "").strip()

    if not keyword:
        return jsonify({"error": "Missing 'keyword'"}), 400

    start, end = parse_month_selection(month_sel)
    results = guardian_fetch_all(keyword, start, end)

    raw_text = aggregate_article_text(results)
    tokens = normalize_text(raw_text)
    freq = Counter(tokens)

    zipf = zipf_data_from_counter(freq, top_k=200)

    ctx_counts = context_window_counts(raw_text, keyword, before=WINDOW_BEFORE, after=WINDOW_AFTER)
    wc_b64 = wordcloud_base64_from_counter(ctx_counts)

    return jsonify({
        "keyword": keyword,
        "month": start.strftime("%Y-%m"),
        "articles": len(results),
        "top_words": counter_to_top_list(freq, n=50),
        "zipf": zipf,
        "context": {
            "window_before": WINDOW_BEFORE,
            "window_after": WINDOW_AFTER,
            "top_context_words": counter_to_top_list(ctx_counts, n=80),

            # original keys
            "wordcloud_png_base64": wc_b64,
            "wordcloud_available": (wc_b64 is not None),

            # compatibility aliases (so old frontend code still works)
            "wordcloud": wc_b64,
            "context_wordcloud_png_base64": wc_b64,
        },

        # top-level alias (some frontends expect this)
        "context_wordcloud_png_base64": wc_b64,
    })


@app.post("/trends")
def trends():
    payload = request.get_json(force=True) or {}
    keyword = (payload.get("keyword") or "").strip()

    if not keyword:
        return jsonify({"error": "Missing 'keyword'"}), 400

    ranges = month_ranges_last_n_completed_months(6)

    month_word_counts = {}
    month_total_tokens = {}
    month_top = {}

    for label, start, end in ranges:
        results = guardian_fetch_all(keyword, start, end)
        raw_text = aggregate_article_text(results)
        tokens = normalize_text(raw_text)
        freq = Counter(tokens)

        month_word_counts[label] = freq
        month_total_tokens[label] = max(1, sum(freq.values()))
        month_top[label] = counter_to_top_list(freq, n=25)

    vocab = set()
    for m, freq in month_word_counts.items():
        for w, _ in freq.most_common(400):
            vocab.add(w)

    months = [label for (label, _, _) in ranges]  # newest -> older

    series = {}
    for w in vocab:
        series[w] = [
            (month_word_counts[m].get(w, 0) / month_total_tokens[m]) * 10000.0
            for m in months
        ]

    deltas = []
    for w, vals in series.items():
        if len(vals) >= 2:
            deltas.append((w, vals[0] - vals[-1]))
    deltas.sort(key=lambda x: x[1], reverse=True)

    top_increasing = [{"word": w, "delta_per_10k": float(d)} for w, d in deltas[:10]]
    top_decreasing = [{"word": w, "delta_per_10k": float(d)} for w, d in deltas[-10:]][::-1]

    movers = [x["word"] for x in top_increasing] + [x["word"] for x in top_decreasing]
    movers_unique = []
    seen = set()
    for w in movers:
        if w not in seen:
            seen.add(w)
            movers_unique.append(w)

    mover_series = [{"word": w, "rates_per_10k": series[w]} for w in movers_unique]

    return jsonify({
        "keyword": keyword,
        "months": months,
        "top_increasing": top_increasing,
        "top_decreasing": top_decreasing,
        "mover_series": mover_series,
        "month_top_words": month_top,
    })


@app.post("/export/csv")
def export_csv():
    if pd is None:
        return jsonify({"error": "pandas not installed. Run: pip install pandas"}), 500

    payload = request.get_json(force=True) or {}
    rows = payload.get("rows")
    filename = payload.get("filename") or "zipflow_export.csv"

    if not isinstance(rows, list) or not rows:
        return jsonify({"error": "Missing 'rows' (list of objects)"}), 400

    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    data = io.BytesIO(buf.getvalue().encode("utf-8"))

    return send_file(data, mimetype="text/csv", as_attachment=True, download_name=filename)


@app.post("/export/excel")
def export_excel():
    if pd is None:
        return jsonify({"error": "pandas not installed. Run: pip install pandas openpyxl"}), 500

    payload = request.get_json(force=True) or {}
    sheets = payload.get("sheets")
    filename = payload.get("filename") or "zipflow_export.xlsx"

    if not isinstance(sheets, dict) or not sheets:
        return jsonify({"error": "Missing 'sheets' (object: sheetName -> rows list)"}), 400

    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        for sheet_name, rows in sheets.items():
            if not isinstance(rows, list):
                continue
            df = pd.DataFrame(rows)
            safe_name = str(sheet_name)[:31]  # Excel limit
            df.to_excel(writer, sheet_name=safe_name, index=False)

    out.seek(0)
    return send_file(
        out,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=filename
    )


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Running directly: python main.py
    # Backend will be at: http://127.0.0.1:5000
    app.run(host="0.0.0.0", port=5000, debug=True)
