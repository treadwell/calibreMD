#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import os
import sqlite3
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates


app = FastAPI(title="Calibre Recommendations UI")
templates = Jinja2Templates(directory="/app/ui/templates")

LIBRARY_PARENT = Path(os.getenv("LIBRARY_PARENT", "/libraries")).resolve()
RESULTS_ROOT = Path(os.getenv("RESULTS_ROOT", "/data/results")).resolve()
RECO_SCRIPT = Path(os.getenv("RECO_SCRIPT", "/app/scripts/xml_ridge_pooled.py")).resolve()
PYTHON_BIN = os.getenv("PYTHON_BIN", "python3")

STATE: Dict[str, str] = {
    "library_dir": "",
    "status": "idle",
    "last_refresh": "",
    "last_error": "",
}
LOCK = threading.Lock()


@dataclass
class RecRow:
    book_id: int
    title: str
    tag: str
    gfm_obj: float


def _is_valid_library_dir(path: Path) -> bool:
    return (path / "metadata.db").exists() and (path / "calibregpt.db").exists()


def discover_libraries(parent: Path) -> List[str]:
    out: List[str] = []
    if not parent.exists():
        return out
    for child in sorted(parent.iterdir()):
        if child.is_dir() and _is_valid_library_dir(child):
            out.append(str(child))
    return out


def results_db_for_library(library_dir: str) -> Path:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(library_dir.encode("utf-8")).hexdigest()[:16]
    return RESULTS_ROOT / f"recs_{key}.sqlite"


def run_refresh(library_dir: str) -> None:
    db_path = results_db_for_library(library_dir)
    cmd = [
        PYTHON_BIN,
        str(RECO_SCRIPT),
        "--library-dir",
        library_dir,
        "--model",
        "text-embedding-ada-002",
        "--bo-init",
        "0",
        "--bo-iters",
        "0",
        "--meta-bo-init",
        "0",
        "--meta-bo-iters",
        "0",
        "--results-db",
        str(db_path),
        "--sample-size",
        "5",
        "--k",
        "32",
        "--min-label-freq",
        "1",
    ]

    with LOCK:
        STATE["status"] = "refreshing"
        STATE["last_error"] = ""

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        elapsed = time.time() - t0
        with LOCK:
            STATE["status"] = "idle"
            STATE["last_refresh"] = time.strftime("%Y-%m-%d %H:%M:%S") + f" ({elapsed:.1f}s)"
            STATE["last_error"] = ""
        # Keep output available via logs if needed.
        print(proc.stdout)
    except subprocess.CalledProcessError as e:
        with LOCK:
            STATE["status"] = "error"
            msg = (e.stderr or e.stdout or str(e))[-4000:]
            STATE["last_error"] = msg


def fetch_recommendations(
    library_dir: str,
    mode: str,
    query: Optional[str],
    limit: int,
) -> List[RecRow]:
    db_path = results_db_for_library(library_dir)
    if not db_path.exists():
        return []

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    if mode == "top":
        rows = cur.execute(
            """
            SELECT book_id, title, tag, gfm_obj
            FROM recommendations
            ORDER BY gfm_obj DESC, book_id ASC, tag ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    elif mode == "book":
        if not query:
            con.close()
            return []
        if query.isdigit():
            rows = cur.execute(
                """
                SELECT book_id, title, tag, gfm_obj
                FROM recommendations
                WHERE book_id = ?
                ORDER BY gfm_obj DESC, tag ASC
                LIMIT ?
                """,
                (int(query), limit),
            ).fetchall()
        else:
            rows = cur.execute(
                """
                SELECT book_id, title, tag, gfm_obj
                FROM recommendations
                WHERE LOWER(title) LIKE LOWER(?)
                ORDER BY gfm_obj DESC, tag ASC
                LIMIT ?
                """,
                (f"%{query}%", limit),
            ).fetchall()
    elif mode == "tag":
        if not query:
            con.close()
            return []
        rows = cur.execute(
            """
            SELECT book_id, title, tag, gfm_obj
            FROM recommendations
            WHERE LOWER(tag) LIKE LOWER(?)
            ORDER BY gfm_obj DESC, book_id ASC
            LIMIT ?
            """,
            (f"%{query}%", limit),
        ).fetchall()
    else:
        con.close()
        return []

    con.close()
    return [RecRow(int(r["book_id"]), str(r["title"]), str(r["tag"]), float(r["gfm_obj"])) for r in rows]


def apply_recommendations(library_dir: str, selections: List[str]) -> Tuple[int, int]:
    md = Path(library_dir) / "metadata.db"
    if not md.exists():
        raise FileNotFoundError(f"metadata.db not found at {md}")

    parsed: List[Tuple[int, str]] = []
    for item in selections:
        parts = item.split("|||", 1)
        if len(parts) != 2:
            continue
        book_id_raw, tag = parts
        if not book_id_raw.isdigit() or not tag:
            continue
        parsed.append((int(book_id_raw), tag.strip()))

    if not parsed:
        return 0, 0

    con = sqlite3.connect(str(md))
    cur = con.cursor()
    added = 0
    skipped = 0

    try:
        for book_id, tag_name in parsed:
            tag_row = cur.execute("SELECT id FROM tags WHERE name = ?", (tag_name,)).fetchone()
            if tag_row is None:
                cur.execute("INSERT INTO tags(name) VALUES (?)", (tag_name,))
                tag_id = int(cur.lastrowid)
            else:
                tag_id = int(tag_row[0])

            link_row = cur.execute(
                "SELECT 1 FROM books_tags_link WHERE book = ? AND tag = ?",
                (book_id, tag_id),
            ).fetchone()
            if link_row is not None:
                skipped += 1
                continue

            cur.execute("INSERT INTO books_tags_link(book, tag) VALUES (?, ?)", (book_id, tag_id))
            added += 1

        con.commit()
    finally:
        con.close()

    return added, skipped


@app.get("/", response_class=HTMLResponse)
def index(request: Request, mode: str = "top", q: str = "", limit: int = 50):
    libraries = discover_libraries(LIBRARY_PARENT)
    with LOCK:
        library_dir = STATE.get("library_dir", "")
        status = STATE.get("status", "idle")
        last_refresh = STATE.get("last_refresh", "")
        last_error = STATE.get("last_error", "")

    recs: List[RecRow] = []
    if library_dir and mode in {"top", "book", "tag"}:
        recs = fetch_recommendations(library_dir=library_dir, mode=mode, query=q, limit=max(1, min(limit, 500)))

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "libraries": libraries,
            "library_dir": library_dir,
            "status": status,
            "last_refresh": last_refresh,
            "last_error": last_error,
            "mode": mode,
            "q": q,
            "limit": limit,
            "recs": recs,
        },
    )


@app.post("/set-library")
def set_library(library_dir: str = Form(...)):
    path = Path(library_dir).expanduser().resolve()
    if not _is_valid_library_dir(path):
        with LOCK:
            STATE["last_error"] = f"Invalid library directory: {path}"
        return RedirectResponse(url="/", status_code=303)

    with LOCK:
        STATE["library_dir"] = str(path)
        STATE["last_error"] = ""
    return RedirectResponse(url="/", status_code=303)


@app.post("/refresh")
def refresh_recommendations():
    with LOCK:
        library_dir = STATE.get("library_dir", "")
        status = STATE.get("status", "idle")

    if not library_dir:
        with LOCK:
            STATE["last_error"] = "Select a library first."
        return RedirectResponse(url="/", status_code=303)

    if status == "refreshing":
        return RedirectResponse(url="/", status_code=303)

    thread = threading.Thread(target=run_refresh, args=(library_dir,), daemon=True)
    thread.start()
    return RedirectResponse(url="/", status_code=303)


@app.post("/apply")
def apply_selected(
    selected: List[str] = Form(default=[]),
    mode: str = Form(default="top"),
    q: str = Form(default=""),
    limit: int = Form(default=50),
):
    with LOCK:
        library_dir = STATE.get("library_dir", "")

    if not library_dir:
        with LOCK:
            STATE["last_error"] = "Select a library first."
        return RedirectResponse(url=f"/?mode={mode}&q={q}&limit={limit}", status_code=303)

    try:
        added, skipped = apply_recommendations(library_dir=library_dir, selections=selected)
        with LOCK:
            STATE["last_error"] = f"Applied recommendations: added={added}, skipped_existing={skipped}"
    except Exception as e:
        with LOCK:
            STATE["last_error"] = f"Apply failed: {e}"

    return RedirectResponse(url=f"/?mode={mode}&q={q}&limit={limit}", status_code=303)


@app.get("/status")
def get_status():
    with LOCK:
        return {
            "library_dir": STATE.get("library_dir", ""),
            "status": STATE.get("status", "idle"),
            "last_refresh": STATE.get("last_refresh", ""),
            "last_error": STATE.get("last_error", ""),
        }
