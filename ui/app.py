#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import mimetypes
import os
import sqlite3
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from fastapi import FastAPI, Form, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates


app = FastAPI(title="Calibre Recommendations UI")
templates = Jinja2Templates(directory="/app/ui/templates")

LIBRARY_PARENT = Path(os.getenv("LIBRARY_PARENT", "/libraries")).resolve()
HOST_LIBRARY_PARENT = Path(os.getenv("HOST_LIBRARY_PARENT", "/Users/kbrooks/Dropbox/Books")).resolve()
RESULTS_ROOT = Path(os.getenv("RESULTS_ROOT", "/data/results")).resolve()
RECO_SCRIPT = Path(os.getenv("RECO_SCRIPT", "/app/scripts/xml_ridge_pooled.py")).resolve()
PYTHON_BIN = os.getenv("PYTHON_BIN", "python3")
UI_MIN_LABEL_FREQ = int(os.getenv("UI_MIN_LABEL_FREQ", "5"))
UI_ELM_HIDDEN_NODES = int(os.getenv("UI_ELM_HIDDEN_NODES", "1024"))
UI_SAMPLE_SIZE = int(os.getenv("UI_SAMPLE_SIZE", "1"))
UI_TOP_K = int(os.getenv("UI_TOP_K", "32"))

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
    open_href: str


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
        str(UI_SAMPLE_SIZE),
        "--k",
        str(UI_TOP_K),
        "--min-label-freq",
        str(UI_MIN_LABEL_FREQ),
        "--elm-hidden-nodes",
        str(UI_ELM_HIDDEN_NODES),
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
    model_mode: str,
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

    q = (query or "").strip()
    db_mode = {"embeddings": "embeddings", "metadata": "elm", "combined": "meta"}.get(model_mode, "meta")

    if mode == "top":
        rows = cur.execute(
            """
            SELECT book_id, title, tag, gfm_obj
            FROM recommendations
            WHERE mode = ?
            ORDER BY gfm_obj DESC, book_id ASC, tag ASC
            LIMIT ?
            """,
            (db_mode, limit),
        ).fetchall()
    elif mode == "book":
        if not q:
            con.close()
            return []
        if q.isdigit():
            rows = cur.execute(
                """
                SELECT book_id, title, tag, gfm_obj
                FROM recommendations
                WHERE mode = ? AND book_id = ?
                ORDER BY gfm_obj DESC, tag ASC
                LIMIT ?
                """,
                (db_mode, int(q), limit),
            ).fetchall()
        else:
            rows = cur.execute(
                """
                SELECT book_id, title, tag, gfm_obj
                FROM recommendations
                WHERE mode = ? AND LOWER(title) LIKE LOWER(?)
                ORDER BY gfm_obj DESC, tag ASC
                LIMIT ?
                """,
                (db_mode, f"%{q}%", limit),
            ).fetchall()
    elif mode == "tag":
        if not q:
            con.close()
            return []
        rows = cur.execute(
            """
            SELECT book_id, title, tag, gfm_obj
            FROM recommendations
            WHERE mode = ? AND LOWER(tag) = LOWER(?)
            ORDER BY gfm_obj DESC, book_id ASC
            LIMIT ?
            """,
            (db_mode, q, limit),
        ).fetchall()
    else:
        con.close()
        return []

    con.close()
    recs = [RecRow(int(r["book_id"]), str(r["title"]), str(r["tag"]), float(r["gfm_obj"]), "") for r in rows]
    open_links = resolve_open_links(library_dir=library_dir, book_ids=[r.book_id for r in recs])
    for r in recs:
        r.open_href = open_links.get(r.book_id, "")
    return recs


def resolve_book_targets(library_dir: str, book_ids: List[int]) -> Dict[int, Path]:
    if not book_ids:
        return {}
    md = Path(library_dir) / "metadata.db"
    if not md.exists():
        return {}

    placeholders = ",".join(["?"] * len(book_ids))
    con = sqlite3.connect(str(md))
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    rows = cur.execute(
        f"""
        SELECT b.id AS book_id, b.path AS rel_path, d.name AS name, d.format AS fmt
        FROM books b
        LEFT JOIN data d ON d.book = b.id
        WHERE b.id IN ({placeholders})
        ORDER BY b.id, d.id
        """,
        tuple(int(x) for x in book_ids),
    ).fetchall()
    con.close()

    out: Dict[int, Path] = {}
    for r in rows:
        bid = int(r["book_id"])
        if bid in out:
            continue
        rel_path = str(r["rel_path"] or "")
        if not rel_path:
            continue
        base = Path(library_dir) / rel_path
        name = r["name"]
        fmt = r["fmt"]
        if name and fmt:
            target = base / f"{name}.{str(fmt).lower()}"
            if not target.exists():
                target = base / f"{name}.{fmt}"
        else:
            target = base
        out[bid] = target.resolve()
    return out


def resolve_open_links(library_dir: str, book_ids: List[int]) -> Dict[int, str]:
    targets = resolve_book_targets(library_dir=library_dir, book_ids=book_ids)
    return {bid: f"/book-file/{bid}" for bid in targets.keys()}


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
def index(request: Request, mode: str = "top", q: str = "", limit: int = 50, model_mode: str = "combined"):
    libraries = discover_libraries(LIBRARY_PARENT)
    with LOCK:
        library_dir = STATE.get("library_dir", "")
        status = STATE.get("status", "idle")
        last_refresh = STATE.get("last_refresh", "")
        last_error = STATE.get("last_error", "")

    recs: List[RecRow] = []
    query_note = ""
    if library_dir and mode in {"top", "book", "tag"}:
        recs = fetch_recommendations(
            library_dir=library_dir,
            model_mode=model_mode,
            mode=mode,
            query=q,
            limit=max(1, min(limit, 500)),
        )
        q_trim = (q or "").strip()
        if mode in {"book", "tag"} and q_trim and len(recs) == 0:
            query_note = "None"

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
            "model_mode": model_mode,
            "q": q,
            "limit": limit,
            "recs": recs,
            "query_note": query_note,
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
    model_mode: str = Form(default="combined"),
    q: str = Form(default=""),
    limit: int = Form(default=50),
):
    with LOCK:
        library_dir = STATE.get("library_dir", "")

    if not library_dir:
        with LOCK:
            STATE["last_error"] = "Select a library first."
        return RedirectResponse(url=f"/?model_mode={model_mode}&mode={mode}&q={q}&limit={limit}", status_code=303)

    try:
        added, skipped = apply_recommendations(library_dir=library_dir, selections=selected)
        with LOCK:
            STATE["last_error"] = f"Applied recommendations: added={added}, skipped_existing={skipped}"
    except Exception as e:
        with LOCK:
            STATE["last_error"] = f"Apply failed: {e}"

    return RedirectResponse(url=f"/?model_mode={model_mode}&mode={mode}&q={q}&limit={limit}", status_code=303)


@app.get("/status")
def get_status():
    with LOCK:
        return {
            "library_dir": STATE.get("library_dir", ""),
            "status": STATE.get("status", "idle"),
            "last_refresh": STATE.get("last_refresh", ""),
            "last_error": STATE.get("last_error", ""),
        }


@app.get("/book-file/{book_id}")
def book_file(book_id: int, download: int = Query(default=0)):
    with LOCK:
        library_dir = STATE.get("library_dir", "")
    if not library_dir:
        return RedirectResponse(url="/", status_code=303)
    targets = resolve_book_targets(library_dir=library_dir, book_ids=[int(book_id)])
    path = targets.get(int(book_id))
    if path is None or not path.exists() or not path.is_file():
        with LOCK:
            STATE["last_error"] = f"File not found for book_id={book_id}"
        return RedirectResponse(url="/", status_code=303)
    mime_type, _ = mimetypes.guess_type(str(path))
    disposition = "attachment" if int(download or 0) == 1 else "inline"
    headers = {"Content-Disposition": f'{disposition}; filename=\"{path.name}\"'}
    return FileResponse(path=str(path), media_type=mime_type, filename=path.name, headers=headers)
