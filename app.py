import json
import os
import re
import sqlite3
from pathlib import Path

import anthropic
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()

DB_PATH = "companies.db"
CSV_PATH = "cleaned_companies.csv"
CLAUDE_MODEL = "claude-sonnet-4-6"

DISPLAY_COLUMNS = [
    "company", "category", "city", "state", "country",
    "employees", "website", "linkedin", "description", "notes"
]

# ── Database setup ─────────────────────────────────────────────────────────────

def load_db():
    if not Path(CSV_PATH).exists():
        raise FileNotFoundError(f"{CSV_PATH} not found")
    df = pd.read_csv(CSV_PATH, dtype=str).fillna("")
    # Clean column names for SQL
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

# Load on startup
db_columns = load_db()
schema_cols, total_count = get_schema()

# ── Claude SQL generation ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a SQL expert helping search a database of companies.

The SQLite table is called "companies" with these columns:
{schema}

Rules:
- Return ONLY the SQL query, no explanation, no markdown.
- Use LIKE '%value%' for text searches (case-insensitive with LOWER()).
- For employee counts, the Employees column contains numbers as text — cast with CAST(Employees AS INTEGER).
- For revenue/funding queries use the relevant numeric columns.
- Always add LIMIT 500 unless the user asks for more.
- For follow-up queries that refine previous results, combine the conditions with AND.
- Common column mappings:
  - "industry" or "sector" or "type" → Category column
  - "location" or "headquarters" or "based in" → City and/or State columns
  - "headcount" or "size" or "staff" → Employees column
  - "founded" → year__nfounded column
  - "description" or "what they do" → Description column"""

def generate_sql(query: str, history: list) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(500, "ANTHROPIC_API_KEY not set")

    schema_str = ", ".join(f"{c['name']} ({c['type']})" for c in schema_cols)
    system = SYSTEM_PROMPT.format(schema=schema_str)

    messages = []
    for h in history[-6:]:  # keep last 3 exchanges for context
        messages.append({"role": "user", "content": h["query"]})
        messages.append({"role": "assistant", "content": h["sql"]})
    messages.append({"role": "user", "content": query})

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=system,
        messages=messages,
    )
    sql = msg.content[0].text.strip()
    sql = re.sub(r"^```(?:sql)?\s*|\s*```$", "", sql, flags=re.MULTILINE).strip()
    return sql

# ── API routes ─────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    history: list = []

@app.post("/api/query")
async def query(req: QueryRequest):
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

    return {"sql": sql, "columns": columns, "rows": rows, "count": len(rows)}

@app.get("/api/stats")
async def stats():
    return {"total": total_count}

@app.get("/api/reload")
async def reload_db():
    global db_columns, schema_cols, total_count
    db_columns = load_db()
    schema_cols, total_count = get_schema()
    return {"total": total_count}

# ── Frontend ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return Path("static/index.html").read_text()

app.mount("/static", StaticFiles(directory="static"), name="static")
