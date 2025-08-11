
import os
import re
import json
import math
import sqlite3
from typing import List, Dict, Any, Tuple
from pathlib import Path
from datetime import date, datetime, timedelta

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# Optional OpenAI support
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

DB_DIR = Path("data")
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "app.sqlite"
SEED_PATH = DB_DIR / "food_seed.csv"
BACKUP_DIR = DB_DIR / "backups"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

MEALS = ["Breakfast", "Lunch", "Dinner", "Snacks"]

DEFAULT_TARGETS = {
    "calories": 2300,
    "protein_g": 160,
    "carbs_g": 220,
    "fat_g": 80,
    "fiber_g": 30,
    "sugar_g": 40,
    "sodium_mg": 2300
}

@st.cache_resource
def get_conn():
    conn = sqlite3.connect(DB_PATH.as_posix(), check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        created_at TEXT NOT NULL
    );""")
    cur.execute("""CREATE TABLE IF NOT EXISTS foods(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        serving_size_grams REAL NOT NULL,
        calories REAL NOT NULL,
        protein_g REAL NOT NULL,
        carbs_g REAL NOT NULL,
        fat_g REAL NOT NULL,
        fiber_g REAL NOT NULL,
        sugar_g REAL NOT NULL,
        sodium_mg REAL NOT NULL,
        tags TEXT
    );""")
    cur.execute("""CREATE TABLE IF NOT EXISTS entries(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        date TEXT NOT NULL,
        meal TEXT NOT NULL,
        food_id INTEGER NOT NULL,
        multiplier REAL NOT NULL DEFAULT 1.0,
        notes TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
        FOREIGN KEY(food_id) REFERENCES foods(id) ON DELETE CASCADE
    );""")
    cur.execute("""CREATE TABLE IF NOT EXISTS targets(
        user_id INTEGER PRIMARY KEY,
        calories REAL, protein_g REAL, carbs_g REAL, fat_g REAL, fiber_g REAL, sugar_g REAL, sodium_mg REAL,
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
    );""")
    conn.commit()

def seed_foods(conn: sqlite3.Connection):
    import pandas as pd
    if not SEED_PATH.exists():
        df = pd.DataFrame([
            ["Chicken breast", 100, 165, 31, 0, 3.6, 0, 0, 74, "whole"],
            ["Egg", 50, 72, 6.3, 0.4, 4.8, 0, 0.2, 62, "whole"],
            ["Avocado", 100, 160, 2, 9, 15, 7, 0.7, 7, "whole"],
            ["White rice cooked", 150, 195, 4, 42, 0.4, 0.6, 0.1, 0, "processed"],
            ["Oats dry", 40, 150, 5, 27, 3, 4, 1, 2, "whole"],
            ["Olive oil", 14, 119, 0, 0, 13.5, 0, 0, 0, "whole"],
            ["Greek yogurt 2%", 170, 130, 17, 6, 4, 0, 6, 55, "processed"],
            ["Protein powder whey", 30, 120, 24, 3, 1.5, 0, 2, 60, "processed"],
            ["Soda regular", 355, 140, 0, 39, 0, 0, 39, 20, "ultra"],
            ["Banana", 118, 105, 1.3, 27, 0.4, 3.1, 14, 1, "whole"],
            ["Flat white 250ml", 250, 120, 8, 10, 5, 0, 10, 80, "processed"],
            ["Bread slice white", 30, 80, 2.7, 14, 1, 0.6, 1.5, 140, "processed"],
            ["Bread slice wholegrain", 35, 90, 4, 15, 1.3, 2.5, 1.5, 140, "processed"],
            ["Apple", 182, 95, 0.5, 25, 0.3, 4.4, 19, 2, "whole"],
            ["Beef sirloin", 100, 200, 26, 0, 10, 0, 0, 55, "whole"],
            ["Salmon", 100, 208, 20, 0, 13, 0, 0, 60, "whole"],
            ["Broccoli", 100, 35, 2.4, 7, 0.4, 2.6, 1.4, 33, "whole"],
        ], columns=["name","serving_size_grams","calories","protein_g","carbs_g","fat_g","fiber_g","sugar_g","sodium_mg","tags"])
        df.to_csv(SEED_PATH, index=False)

    df = pd.read_csv(SEED_PATH)
    cur = conn.cursor()
    for _, r in df.iterrows():
        cur.execute("""INSERT OR IGNORE INTO foods(name, serving_size_grams, calories, protein_g, carbs_g, fat_g, fiber_g, sugar_g, sodium_mg, tags)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (r["name"], r["serving_size_grams"], r["calories"], r["protein_g"], r["carbs_g"], r["fat_g"], r["fiber_g"], r["sugar_g"], r["sodium_mg"], r.get("tags","")))
    conn.commit()

def ensure_user(conn: sqlite3.Connection, name: str) -> int:
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE name=?", (name,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute("INSERT INTO users(name, created_at) VALUES (?,?)", (name, datetime.utcnow().isoformat()))
    conn.commit()
    return cur.lastrowid

def get_targets(conn: sqlite3.Connection, user_id: int) -> Dict[str, float]:
    cur = conn.cursor()
    cur.execute("SELECT calories, protein_g, carbs_g, fat_g, fiber_g, sugar_g, sodium_mg FROM targets WHERE user_id=?", (user_id,))
    row = cur.fetchone()
    if not row:
        cur.execute("""INSERT OR REPLACE INTO targets(user_id, calories, protein_g, carbs_g, fat_g, fiber_g, sugar_g, sodium_mg)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (user_id, DEFAULT_TARGETS["calories"], DEFAULT_TARGETS["protein_g"],
                     DEFAULT_TARGETS["carbs_g"], DEFAULT_TARGETS["fat_g"], DEFAULT_TARGETS["fiber_g"],
                     DEFAULT_TARGETS["sugar_g"], DEFAULT_TARGETS["sodium_mg"]))
        conn.commit()
        return DEFAULT_TARGETS.copy()
    keys = ["calories","protein_g","carbs_g","fat_g","fiber_g","sugar_g","sodium_mg"]
    return dict(zip(keys, row))

def update_targets(conn: sqlite3.Connection, user_id: int, values: Dict[str, float]):
    cur = conn.cursor()
    cur.execute("""INSERT OR REPLACE INTO targets(user_id, calories, protein_g, carbs_g, fat_g, fiber_g, sugar_g, sodium_mg)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (user_id, values["calories"], values["protein_g"], values["carbs_g"],
                 values["fat_g"], values["fiber_g"], values["sugar_g"], values["sodium_mg"]))
    conn.commit()

def search_foods(conn: sqlite3.Connection, q: str) -> pd.DataFrame:
    cur = conn.cursor()
    q_like = f"%{q.lower()}%"
    df = pd.read_sql_query("SELECT * FROM foods WHERE lower(name) LIKE ? ORDER BY name LIMIT 100", conn, params=(q_like,))
    return df

def add_food(conn: sqlite3.Connection, food: Dict[str, Any]):
    cur = conn.cursor()
    cur.execute("""INSERT OR REPLACE INTO foods(name, serving_size_grams, calories, protein_g, carbs_g, fat_g, fiber_g, sugar_g, sodium_mg, tags)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (food["name"], food["serving_size_grams"], food["calories"], food["protein_g"], food["carbs_g"],
                 food["fat_g"], food["fiber_g"], food["sugar_g"], food["sodium_mg"], food.get("tags","")))
    conn.commit()

def add_entry(conn: sqlite3.Connection, user_id: int, d: date, meal: str, food_id: int, mult: float, notes: str=""):
    cur = conn.cursor()
    cur.execute("""INSERT INTO entries(user_id, date, meal, food_id, multiplier, notes)
                   VALUES (?,?,?,?,?,?)""", (user_id, d.isoformat(), meal, food_id, mult, notes))
    conn.commit()

def delete_entry(conn: sqlite3.Connection, entry_id: int):
    conn.execute("DELETE FROM entries WHERE id=?", (entry_id,))
    conn.commit()

def day_entries(conn: sqlite3.Connection, user_id: int, d: date) -> pd.DataFrame:
    q = """
    SELECT e.id as entry_id, e.meal, e.multiplier, e.notes, f.*
    FROM entries e
    JOIN foods f ON e.food_id = f.id
    WHERE e.user_id=? AND e.date=?
    ORDER BY CASE e.meal
        WHEN 'Breakfast' THEN 1
        WHEN 'Lunch' THEN 2
        WHEN 'Dinner' THEN 3
        ELSE 4
    END, e.id
    """
    df = pd.read_sql_query(q, get_conn(), params=(user_id, d.isoformat()))
    return df

def compute_totals(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {k: 0.0 for k in ["calories","protein_g","carbs_g","fat_g","fiber_g","sugar_g","sodium_mg"]}
    mult = df["multiplier"].astype(float)
    totals = {}
    for col in ["calories","protein_g","carbs_g","fat_g","fiber_g","sugar_g","sodium_mg"]:
        totals[col] = float((df[col] * mult).sum())
    return totals

def macro_share(totals: Dict[str,float]) -> Tuple[float,float,float]:
    p_kcal = totals["protein_g"] * 4
    c_kcal = totals["carbs_g"] * 4
    f_kcal = totals["fat_g"] * 9
    s = p_kcal + c_kcal + f_kcal
    if s <= 0:
        return (0.0, 0.0, 0.0)
    return (p_kcal/s, c_kcal/s, f_kcal/s)

def cosine_distance(a: Tuple[float,float,float], b: Tuple[float,float,float]) -> float:
    ax = np.array(a); bx = np.array(b)
    if np.linalg.norm(ax) == 0 or np.linalg.norm(bx) == 0:
        return 1.0
    cos_sim = float(ax @ bx) / (np.linalg.norm(ax) * np.linalg.norm(bx))
    return max(0.0, 1.0 - cos_sim)

def health_score(totals: Dict[str,float], targets: Dict[str,float], tags: List[str]) -> Tuple[int, Dict[str,float]]:
    score = 100.0
    breakdown = {}
    if targets["calories"] > 0:
        cal_pen = min(30.0, abs(totals["calories"] - targets["calories"]) / targets["calories"] * 30.0)
    else:
        cal_pen = 0.0
    score -= cal_pen; breakdown["calorie_alignment"] = -cal_pen
    prot = totals["protein_g"]
    p_target = max(1.0, targets["protein_g"])
    p_bonus = min(15.0, 15.0 * prot / p_target)
    score += p_bonus; breakdown["protein"] = p_bonus
    f_bonus = min(10.0, 10.0 * totals["fiber_g"] / max(1.0, targets["fiber_g"]))
    score += f_bonus; breakdown["fiber"] = f_bonus
    s_limit = max(1.0, targets["sugar_g"]); sugar_ratio = totals["sugar_g"] / s_limit
    s_pen = 0.0
    if sugar_ratio > 1.0:
        s_pen = min(15.0, (sugar_ratio - 1.0) * 15.0)
    score -= s_pen; breakdown["sugar"] = -s_pen
    sod_limit = max(1.0, targets["sodium_mg"]); sod_ratio = totals["sodium_mg"] / sod_limit
    sod_pen = 0.0
    if sod_ratio > 1.0:
        sod_pen = min(10.0, (sod_ratio - 1.0) * 10.0)
    score -= sod_pen; breakdown["sodium"] = -sod_pen
    actual_share = macro_share(totals)
    target_share = macro_share({"protein_g": targets["protein_g"], "carbs_g": targets["carbs_g"], "fat_g": targets["fat_g"]})
    dist = cosine_distance(actual_share, target_share)
    mb_pen = min(10.0, 10.0 * dist)
    score -= mb_pen; breakdown["macro_balance"] = -mb_pen
    tag_pen = 0.0
    if "ultra" in tags:
        tag_pen = 10.0
    elif "processed" in tags:
        tag_pen = 5.0
    score -= tag_pen; breakdown["processing"] = -tag_pen
    score = max(0, min(100, round(score)))
    return score, breakdown

def score_label(score: int) -> str:
    if score >= 90: return "Excellent"
    if score >= 75: return "Strong"
    if score >= 60: return "Mixed"
    if score >= 40: return "Needs work"
    return "Off track"

def export_entries(conn: sqlite3.Connection, user_id: int, start: date, end: date) -> pd.DataFrame:
    q = """
    SELECT e.date, e.meal, f.name as food, e.multiplier,
           f.serving_size_grams, f.calories, f.protein_g, f.carbs_g, f.fat_g, f.fiber_g, f.sugar_g, f.sodium_mg, f.tags, e.notes
    FROM entries e JOIN foods f ON e.food_id = f.id
    WHERE e.user_id=? AND e.date BETWEEN ? AND ?
    ORDER BY e.date, e.meal, e.id
    """
    df = pd.read_sql_query(q, conn, params=(user_id, start.isoformat(), end.isoformat()))
    return df

def backup_db():
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    dest = BACKUP_DIR / f"{ts}.sqlite"
    with open(DB_PATH, "rb") as fsrc, open(dest, "wb") as fdst:
        fdst.write(fsrc.read())
    return dest

def import_backup(uploaded_file):
    dest = DB_PATH
    with open(dest, "wb") as fdst:
        fdst.write(uploaded_file.read())
    return dest

# -------- AI Estimation --------

UNIT_GRAM_SYNONYMS = ["g", "gram", "grams"]
UNIT_PIECE_SYNONYMS = ["pc", "piece", "pieces", "x"]
COMMON_ALIASES = {
    "flat white":"Flat white 250ml",
    "coffee":"Flat white 250ml",
    "white bread":"Bread slice white",
    "wholegrain bread":"Bread slice wholegrain",
    "rice":"White rice cooked",
    "oatmeal":"Oats dry",
    "chicken":"Chicken breast",
    "beef":"Beef sirloin",
    "salmon":"Salmon"
}

def _extract_quantity_tokens(text: str):
    # crude quantity parsing: numbers and unit words
    toks = re.findall(r"(\d+(?:\.\d+)?)\s*(g|gram|grams|pcs?|pieces?|x)?", text.lower())
    return toks

def find_food_best_match(conn, name: str) -> Dict[str, Any]:
    # resolve aliases
    base = name.strip().lower()
    if base in COMMON_ALIASES:
        name = COMMON_ALIASES[base]
    df = search_foods(conn, name)
    if df.empty:
        return None
    # prefer exact name match ignoring case
    exact = df[df["name"].str.lower() == name.strip().lower()]
    if not exact.empty:
        return exact.iloc[0].to_dict()
    # else first partial
    return df.iloc[0].to_dict()

def estimate_local(conn, text: str) -> List[Dict[str, Any]]:
    # Split by commas or "and"
    parts = re.split(r",| and ", text.lower())
    results = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # quantity grams
        m = re.search(r"(\d+(?:\.\d+)?)\s*(g|gram|grams)\b", p)
        grams = None
        if m:
            grams = float(m.group(1))
        # pieces like "2 eggs"
        m2 = re.search(r"(\d+(?:\.\d+)?)\s*(pcs?|pieces?|x)?\s*(.+)", p)
        count = None; name_guess = p
        if m2:
            count = float(m2.group(1))
            # if unit missing, treat as pieces only when food is known as piece based, like egg, bread slice
            name_guess = m2.group(3).strip()
        else:
            # fallback name guess
            name_guess = re.sub(r"\d+(?:\.\d+)?\s*(g|gram|grams|pcs?|pieces?|x)", "", p).strip()

        # fetch food template
        food = find_food_best_match(conn, name_guess)
        if not food:
            # unknown, default 100g 200 kcal generic estimate
            food = {
                "id": None, "name": name_guess.title(), "serving_size_grams": 100.0,
                "calories": 200.0, "protein_g": 8.0, "carbs_g": 20.0, "fat_g": 7.0,
                "fiber_g": 2.0, "sugar_g": 4.0, "sodium_mg": 100.0, "tags":"processed"
            }
        # compute multiplier
        mult = 1.0
        if grams is not None:
            mult = grams / float(food["serving_size_grams"])
        elif count is not None:
            # if the base food has serving size like 50g for egg, use count as x
            mult = count
        else:
            mult = 1.0

        results.append({
            "food_id": food.get("id"),
            "name": food["name"],
            "serving_size_grams": float(food["serving_size_grams"]),
            "multiplier": float(mult),
            "calories": float(food["calories"]) * mult,
            "protein_g": float(food["protein_g"]) * mult,
            "carbs_g": float(food["carbs_g"]) * mult,
            "fat_g": float(food["fat_g"]) * mult,
            "fiber_g": float(food["fiber_g"]) * mult,
            "sugar_g": float(food["sugar_g"]) * mult,
            "sodium_mg": float(food["sodium_mg"]) * mult,
            "tags": food.get("tags","")
        })
    return results

def estimate_openai(text: str, api_key: str) -> List[Dict[str, Any]]:
    if OpenAI is None:
        st.error("OpenAI client not installed. Check requirements.")
        return []
    client = OpenAI(api_key=api_key)
    sys = "You are a nutrition estimator. Parse the meal text into items. For each item, estimate serving grams and macros. Return strict JSON list with fields: name, serving_size_grams, grams_used, calories, protein_g, carbs_g, fat_g, fiber_g, sugar_g, sodium_mg, tags."
    user = f"Meal: {text}\nRespond with JSON only."
    resp = client.responses.create(
        model="gpt-4o-mini",
        input=[{"role":"system","content":sys},{"role":"user","content":user}],
        temperature=0.2
    )
    content = resp.output_text
    try:
        data = json.loads(content)
        results = []
        for item in data:
            results.append({
                "food_id": None,
                "name": item.get("name","Unknown"),
                "serving_size_grams": float(item.get("serving_size_grams", 100)),
                "multiplier": float(item.get("grams_used", item.get("serving_size_grams",100))) / float(item.get("serving_size_grams",100)),
                "calories": float(item.get("calories", 0)),
                "protein_g": float(item.get("protein_g", 0)),
                "carbs_g": float(item.get("carbs_g", 0)),
                "fat_g": float(item.get("fat_g", 0)),
                "fiber_g": float(item.get("fiber_g", 0)),
                "sugar_g": float(item.get("sugar_g", 0)),
                "sodium_mg": float(item.get("sodium_mg", 0)),
                "tags": item.get("tags","estimated")
            })
        return results
    except Exception:
        st.error("Failed to parse model JSON. Try again or use Local estimator.")
        return []

# ------------- UI -------------

st.set_page_config(page_title="Daily Fitness Log", page_icon="💪", layout="wide")

conn = get_conn()
init_db(conn)
seed_foods(conn)

with st.sidebar:
    st.title("💪 Fitness Log")
    name = st.text_input("Profile name", value="You")
    user_id = ensure_user(conn, name.strip() or "You")

    today = date.today()
    d = st.date_input("Pick a date", value=today)
    st.session_state.setdefault("current_date", d)
    st.session_state["current_date"] = d

    st.markdown("### Targets")
    t = get_targets(conn, user_id)
    with st.expander("Edit targets"):
        cols = st.columns(2)
        t["calories"] = cols[0].number_input("Calories", value=float(t["calories"]), step=50.0)
        t["protein_g"] = cols[1].number_input("Protein g", value=float(t["protein_g"]), step=5.0)
        t["carbs_g"] = cols[0].number_input("Carbs g", value=float(t["carbs_g"]), step=5.0)
        t["fat_g"] = cols[1].number_input("Fat g", value=float(t["fat_g"]), step=2.0)
        t["fiber_g"] = cols[0].number_input("Fiber g", value=float(t["fiber_g"]), step=2.0)
        t["sugar_g"] = cols[1].number_input("Sugar limit g", value=float(t["sugar_g"]), step=2.0)
        t["sodium_mg"] = cols[0].number_input("Sodium limit mg", value=float(t["sodium_mg"]), step=50.0)
        if st.button("Save targets"):
            update_targets(conn, user_id, t)
            st.success("Targets saved")

    st.markdown("### Data")
    range_start = st.date_input("Export start", value=today - timedelta(days=6), key="exp_start")
    range_end = st.date_input("Export end", value=today, key="exp_end")
    if st.button("Export CSV"):
        df = export_entries(conn, user_id, range_start, range_end)
        csv = df.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, file_name=f"fitness_export_{range_start}_{range_end}.csv", mime="text/csv")
    if st.button("Backup DB"):
        p = backup_db()
        st.success(f"Backed up to {p}")
    up = st.file_uploader("Restore DB from backup file", type=["sqlite","db"])
    if up is not None:
        import_backup(up)
        st.success("Database restored. Reload the app.")

    st.markdown("### AI Estimator")
    ai_mode = st.radio("Mode", ["Off", "Local estimator", "OpenAI API"], index=1)
    api_key = ""
    if ai_mode == "OpenAI API":
        api_key = st.text_input("OpenAI API Key", type="password", help="Stored in memory for this session only.")

st.header(f"{name}'s day: {st.session_state['current_date'].isoformat()}")

df_day = day_entries(conn, user_id, st.session_state["current_date"])
totals = compute_totals(df_day)

c1, c2, c3 = st.columns([1,1,1])
with c1:
    st.metric("Calories", f"{totals['calories']:.0f} / {t['calories']:.0f}")
with c2:
    st.metric("Protein", f"{totals['protein_g']:.0f} g / {t['protein_g']:.0f} g")
with c3:
    st.metric("Fiber", f"{totals['fiber_g']:.0f} g / {t['fiber_g']:.0f} g")

ring = pd.DataFrame({
    "label": ["Consumed","Remaining"],
    "value": [min(totals["calories"], t["calories"]), max(t["calories"]-totals["calories"], 0)]
})
fig = px.pie(ring, values="value", names="label", hole=0.6)
fig.update_traces(textinfo="none")
fig.update_layout(height=250, margin=dict(l=10,r=10,t=10,b=10))
st.plotly_chart(fig, use_container_width=True)

tags = set()
if not df_day.empty:
    for s in df_day["tags"].dropna().tolist():
        for tag in str(s).split(","):
            tags.add(tag.strip().lower())
score, breakdown = health_score(totals, t, list(tags))
st.subheader(f"Health score: {score} ({score_label(score)})")
bd_df = pd.DataFrame([{"factor": k, "points": v} for k, v in breakdown.items()])
st.dataframe(bd_df, hide_index=True, use_container_width=True)

mac = pd.DataFrame({
    "macro": ["Protein","Carbs","Fat"],
    "grams": [totals["protein_g"], totals["carbs_g"], totals["fat_g"]]
})
fig2 = px.bar(mac, x="macro", y="grams", title="Macros g today")
fig2.update_layout(height=300, margin=dict(l=10,r=10,t=30,b=10))
st.plotly_chart(fig2, use_container_width=True)

# Meals UI
st.subheader("Meals")
for meal in MEALS:
    st.markdown(f"### {meal}")
    with st.form(f"add_{meal}"):
        cols = st.columns([3,1,1])
        q = cols[0].text_input("Search or create", key=f"q_{meal}")
        mult = cols[1].number_input("x", value=1.0, step=0.5, min_value=0.1)
        submitted = cols[2].form_submit_button("Add")
        if submitted and q.strip():
            df = search_foods(conn, q.strip())
            if df.empty or q.strip().lower() not in df["name"].str.lower().tolist():
                st.session_state[f"create_{meal}"] = q.strip()
            else:
                exact = df[df["name"].str.lower() == q.strip().lower()]
                if exact.empty:
                    exact = df.head(1)
                food_id = int(exact.iloc[0]["id"])
                add_entry(conn, user_id, st.session_state["current_date"], meal, food_id, mult)
                st.success(f"Added {q.strip()} x{mult}")
                st.experimental_rerun()

    if st.session_state.get(f"create_{meal}"):
        st.info(f"Create new food: {st.session_state[f'create_{meal}']}")
        with st.expander("Create food"):
            fname = st.text_input("Name", value=st.session_state[f"create_{meal}"])
            cols = st.columns(2)
            sgram = cols[0].number_input("Serving size grams", value=100.0, step=5.0, min_value=1.0)
            cal = cols[1].number_input("Calories", value=0.0, step=5.0, min_value=0.0)
            prot = cols[0].number_input("Protein g", value=0.0, step=0.5, min_value=0.0)
            carbs = cols[1].number_input("Carbs g", value=0.0, step=0.5, min_value=0.0)
            fat = cols[0].number_input("Fat g", value=0.0, step=0.5, min_value=0.0)
            fiber = cols[1].number_input("Fiber g", value=0.0, step=0.5, min_value=0.0)
            sugar = cols[0].number_input("Sugar g", value=0.0, step=0.5, min_value=0.0)
            sodium = cols[1].number_input("Sodium mg", value=0.0, step=5.0, min_value=0.0)
            tags_input = st.text_input("Tags comma separated", value="")
            if st.button("Save food"):
                food = {"name": fname.strip(), "serving_size_grams": sgram, "calories": cal, "protein_g": prot,
                        "carbs_g": carbs, "fat_g": fat, "fiber_g": fiber, "sugar_g": sugar, "sodium_mg": sodium, "tags": tags_input}
                if food["name"]:
                    add_food(conn, food)
                    st.success(f"Saved {food['name']}")
                    st.session_state[f"create_{meal}"] = None
                    st.experimental_rerun()

    # list entries for this meal
    subset = df_day[df_day["meal"] == meal] if not df_day.empty else pd.DataFrame()
    if subset.empty:
        st.caption("No items yet.")
    else:
        for _, r in subset.iterrows():
            cols = st.columns([4,1,1,1,1,1])
            label = f"{r['name']} ({r['serving_size_grams']} g per 1x)"
            cols[0].markdown(f"**{label}**")
            cols[1].markdown(f"x {r['multiplier']}")
            kcal = r['calories'] * r['multiplier']
            cols[2].markdown(f"{kcal:.0f} kcal")
            cols[3].markdown(f"P {r['protein_g']*r['multiplier']:.0f} g")
            cols[4].markdown(f"C {r['carbs_g']*r['multiplier']:.0f} g")
            cols[5].markdown(f"F {r['fat_g']*r['multiplier']:.0f} g")
            if st.button("Delete", key=f"del_{r['entry_id']}"):
                delete_entry(conn, int(r["entry_id"]))
                st.experimental_rerun()

# AI panel
st.subheader("AI meal parser")
colA, colB = st.columns([3,1])
with colA:
    free_text = st.text_area("Describe your meal. Example: '2 eggs, 100g avocado, flat white'", height=100)
with colB:
    meal_pick = st.selectbox("Add to meal", MEALS, index=0)
    estimate_btn = st.button("Estimate")

if estimate_btn and free_text.strip():
    rows = []
    if st.session_state.get("ai_mode_cache") != None:
        pass
    if 'ai_mode_cache' not in st.session_state:
        st.session_state['ai_mode_cache'] = None
    st.session_state['ai_mode_cache'] = None

    if ai_mode == "OpenAI API" and api_key:
        rows = estimate_openai(free_text, api_key)
    elif ai_mode == "Local estimator":
        rows = estimate_local(conn, free_text)
    else:
        st.info("AI is Off. Switch to Local estimator or OpenAI API in the sidebar.")
    if rows:
        df_est = pd.DataFrame(rows)
        st.dataframe(df_est[["name","multiplier","calories","protein_g","carbs_g","fat_g","fiber_g","sugar_g","sodium_mg","tags"]], use_container_width=True, hide_index=True)
        if st.button("Add all to selected meal"):
            # ensure foods exist, then add entries
            for r in rows:
                if r["food_id"] is None:
                    # create food using serving_size_grams and per-serving macros
                    add_food(conn, {
                        "name": r["name"],
                        "serving_size_grams": r["serving_size_grams"],
                        "calories": r["calories"] / max(r["multiplier"], 1e-6),
                        "protein_g": r["protein_g"] / max(r["multiplier"], 1e-6),
                        "carbs_g": r["carbs_g"] / max(r["multiplier"], 1e-6),
                        "fat_g": r["fat_g"] / max(r["multiplier"], 1e-6),
                        "fiber_g": r["fiber_g"] / max(r["multiplier"], 1e-6),
                        "sugar_g": r["sugar_g"] / max(r["multiplier"], 1e-6),
                        "sodium_mg": r["sodium_mg"] / max(r["multiplier"], 1e-6),
                        "tags": r.get("tags","estimated")
                    })
                    # fetch the new id
                    ff = search_foods(conn, r["name"])
                    if not ff.empty:
                        r["food_id"] = int(ff.iloc[0]["id"])
                if r["food_id"] is not None:
                    add_entry(conn, user_id, st.session_state["current_date"], meal_pick, int(r["food_id"]), float(r["multiplier"]))
            st.success("Added AI estimated items.")
            st.experimental_rerun()

# tips
st.subheader("Coach tips")
tips = []
if totals["protein_g"] < t["protein_g"] * 0.8:
    tips.append("Increase lean protein to hit your target.")
if totals["fiber_g"] < t["fiber_g"] * 0.8:
    tips.append("Add high fiber foods like oats, legumes, or veggies.")
if totals["sugar_g"] > t["sugar_g"]:
    tips.append("Reduce added sugars. Swap soda for water or zero sugar options.")
if totals["sodium_mg"] > t["sodium_mg"]:
    tips.append("Watch sodium by choosing lower-sodium sauces and processed foods.")
if not tips:
    tips = ["Great balance. Keep it steady."]
st.write("- " + "\n- ".join(tips))
