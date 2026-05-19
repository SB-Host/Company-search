import hashlib
import hmac
import os
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()

DB_PATH = "companies.db"
CSV_PATH = "cleaned_companies.csv"
CLAUDE_MODEL = "claude-sonnet-4-6"

# ── Database setup ──────────────────────────────────────────────────────────────

def load_db():
    if not Path(CSV_PATH).exists():
        raise FileNotFoundError(f"{CSV_PATH} not found")
    df = pd.read_csv(CSV_PATH, dtype=str).fillna("")
    seen = {}
    new_cols = []
    for col in df.columns:
        cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", col.strip()).strip("_").lower()
        if cleaned in seen:
            seen[cleaned] += 1
            cleaned = f"{cleaned}_{seen[cleaned]}"
        else:
            seen[cleaned] = 0
        new_cols.append(cleaned)
    df.columns = new_cols
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("companies", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()
    print(f"Loaded {len(df)} companies into database.")
    return df.columns.tolist()

def get_conn():
    return sqlite3.connect(DB_PATH)

def get_schema():
    conn = get_conn()
    cursor = conn.execute("PRAGMA table_info(companies)")
    cols = [{"name": r[1], "type": r[2]} for r in cursor.fetchall()]
    cursor = conn.execute("SELECT COUNT(*) FROM companies")
    count = cursor.fetchone()[0]
    conn.close()
    return cols, count

db_columns = load_db()
schema_cols, total_count = get_schema()

# ── In-memory search history ────────────────────────────────────────────────────

_history_items: dict = {}
_history_order: list = []
_history_counter = 0
_history_lock = threading.Lock()

def save_search(query: str, result_count: int):
    global _history_counter
    with _history_lock:
        _history_counter += 1
        item = {
            "id": _history_counter,
            "query": query,
            "result_count": result_count,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        }
        _history_items[_history_counter] = item
        _history_order.insert(0, _history_counter)
        if len(_history_order) > 50:
            old_id = _history_order.pop()
            _history_items.pop(old_id, None)

def get_recent_searches():
    with _history_lock:
        return [_history_items[i] for i in _history_order if i in _history_items]

def delete_search(item_id: int) -> bool:
    with _history_lock:
        if item_id in _history_items:
            del _history_items[item_id]
            return True
        return False

# ── Auth ────────────────────────────────────────────────────────────────────────

def _session_token() -> str:
    password = os.environ.get("APP_PASSWORD", "changeme")
    return hashlib.sha256(f"{password}:uv-session-v1".encode()).hexdigest()

async def require_auth(request: Request):
    token = request.cookies.get("uv_session")
    expected = _session_token()
    if not (token and hmac.compare_digest(token, expected)):
        raise HTTPException(status_code=401, detail="Not authenticated")

# ── Claude SQL generation ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a SQL expert helping search a database of companies.

The SQLite table is called "companies" with these columns:
{schema}

Rules:
- Return ONLY the SQL query, no explanation, no markdown.
- Always SELECT: company, category, city, state, country, employees, website, linkedin, description, year__nfounded — plus any other columns relevant to the query. Never omit website or linkedin.
- Use LIKE '%value%' for text searches (case-insensitive with LOWER()).
- For employee counts, the Employees column contains numbers as text — cast with CAST(Employees AS INTEGER).
- Always add LIMIT 500 unless the user asks for more.
- For follow-up queries that refine previous results, combine the conditions with AND.
- Common column mappings:
  - "industry" or "sector" or "type" → category column
  - "location" or "headquarters" or "based in" → city and/or state columns
  - "headcount" or "size" or "staff" → employees column
  - "founded" → year__nfounded column
  - "description" or "what they do" → description column"""

def generate_sql(query: str, history: list) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(500, "ANTHROPIC_API_KEY not set")
    schema_str = ", ".join(f"{c['name']} ({c['type']})" for c in schema_cols)
    system = SYSTEM_PROMPT.format(schema=schema_str)
    messages = []
    for h in history[-6:]:
        messages.append({"role": "user", "content": h["query"]})
        messages.append({"role": "assistant", "content": h["sql"]})
    messages.append({"role": "user", "content": query})
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=CLAUDE_MODEL, max_tokens=1024, system=system, messages=messages,
    )
    sql = msg.content[0].text.strip()
    sql = re.sub(r"^```(?:sql)?\s*|\s*```$", "", sql, flags=re.MULTILINE).strip()
    return sql

# ── API routes ──────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    password: str

class QueryRequest(BaseModel):
    query: str
    history: list = []

@app.post("/api/login")
async def login(req: LoginRequest, request: Request, response: Response):
    expected = os.environ.get("APP_PASSWORD", "changeme")
    if not hmac.compare_digest(req.password, expected):
        raise HTTPException(status_code=401, detail="Invalid password")
    response.set_cookie(
        key="uv_session",
        value=_session_token(),
        httponly=True,
        samesite="lax",
        max_age=86400 * 30,
        path="/",
        secure=request.url.scheme == "https",
    )
    return {"ok": True}

@app.post("/api/logout")
async def logout(response: Response):
    response.delete_cookie("uv_session")
    return {"ok": True}

@app.get("/api/stats")
async def stats():
    return {"total": total_count}

@app.get("/api/history")
async def history(_: None = Depends(require_auth)):
    return get_recent_searches()

@app.delete("/api/history/{item_id}")
async def delete_history(item_id: int, _: None = Depends(require_auth)):
    if not delete_search(item_id):
        raise HTTPException(404, "Not found")
    return {"ok": True}

@app.get("/api/analytics")
async def analytics(_: None = Depends(require_auth)):
    conn = get_conn()
    cats = conn.execute("""
        SELECT category, COUNT(*) as cnt FROM companies
        WHERE category != '' GROUP BY category ORDER BY cnt DESC LIMIT 7
    """).fetchall()
    states = conn.execute("""
        SELECT state, COUNT(*) as cnt FROM companies
        WHERE state != '' AND length(state) = 2 GROUP BY state ORDER BY cnt DESC LIMIT 6
    """).fetchall()
    conn.close()
    searches = get_recent_searches()
    total_results = sum(s["result_count"] for s in searches)
    avg_results = round(total_results / len(searches)) if searches else 0
    return {
        "top_categories": [{"name": r[0], "count": r[1]} for r in cats],
        "top_states":     [{"name": r[0], "count": r[1]} for r in states],
        "session_searches": len(searches),
        "avg_results":    avg_results,
        "total_companies": total_count,
    }

@app.post("/api/query")
async def query(req: QueryRequest, _: None = Depends(require_auth)):
    try:
        sql = generate_sql(req.query, req.history)
    except Exception as e:
        raise HTTPException(500, f"Claude error: {e}")
    try:
        conn = get_conn()
        cursor = conn.execute(sql)
        columns = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
        conn.close()
    except Exception as e:
        raise HTTPException(400, f"SQL error: {e} | Generated SQL: {sql}")
    save_search(req.query, len(rows))
    return {"sql": sql, "columns": columns, "rows": rows, "count": len(rows)}

@app.get("/api/reload")
async def reload_db(_: None = Depends(require_auth)):
    global db_columns, schema_cols, total_count
    db_columns = load_db()
    schema_cols, total_count = get_schema()
    return {"total": total_count}

@app.get("/", response_class=HTMLResponse)
async def root():
    return Path("static/index.html").read_text()

app.mount("/static", StaticFiles(directory="static"), name="static")
