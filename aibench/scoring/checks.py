"""Reference-based check implementations for Phase 2 scoring.

Each check takes the task, the request result, and its spec dict, and returns
(score in [0, 1], note). Checks are deterministic and offline. Code checks
actually execute the model's output:
  * `python_func` runs the generated function in an isolated subprocess
    (``python -I``) with a timeout, then compares outputs to known cases.
  * `sql_sqlite` runs the generated query against an in-memory SQLite DB seeded
    with known data and checks the returned rows.

Executing model-generated code is inherent to correctness testing; it is
sandboxed only by process isolation + timeout, so run benchmarks against models
you trust, on tasks you control (as here).
"""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from ..client import RequestResult
from ..tasks import Task
from .extract import best_code


# --- code execution -------------------------------------------------------

# The candidate code goes FIRST so that a leading `from __future__ import ...`
# (which Python requires to be the first statement in the file) stays valid.
# The harness imports/logic follow it — imports are legal anywhere except
# __future__, and reading the LAST stdout line tolerates any prints the
# candidate emits.
_PY_DRIVER = '''\
{code}

import json as _j, sys as _s
_cases = _j.loads(_s.argv[1])
_out = []
for _c in _cases:
    try:
        _r = {func}(*_c["args"])
        _out.append({{"ok": True, "val": _r}})
    except Exception as _e:
        _out.append({{"ok": False, "err": repr(_e)}})
print(_j.dumps(_out))
'''


def check_python_func(task: Task, r: RequestResult, spec: dict) -> tuple[float, str]:
    code = best_code(r.text, "python")
    func = spec["func"]
    cases = spec["cases"]
    timeout = spec.get("timeout", 10)
    if func not in code:
        return 0.0, f"function `{func}` not found in output"

    driver = _PY_DRIVER.format(code=code, func=func)
    with tempfile.TemporaryDirectory() as d:
        script = Path(d) / "cand.py"
        script.write_text(driver, encoding="utf-8")
        # In a normal install sys.executable is Python (run isolated: -I). In a
        # PyInstaller build it's the aibench exe, which runs the script via its
        # own `__pyexec__` self-exec hook (see cli.main). Either way the script
        # runs in a separate process, so the timeout still bounds it.
        if getattr(sys, "frozen", False):
            cmd = [sys.executable, "__pyexec__", str(script), json.dumps(cases)]
        else:
            cmd = [sys.executable, "-I", str(script), json.dumps(cases)]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return 0.0, f"execution timed out (> {timeout}s)"
        if proc.returncode != 0:
            err = (proc.stderr or "").strip().splitlines()
            return 0.0, f"code failed to run: {err[-1] if err else 'error'}"
        try:
            got = json.loads(proc.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            return 0.0, "could not parse execution output"

    passed = 0
    for case, res in zip(cases, got):
        if res.get("ok") and res.get("val") == case["expect"]:
            passed += 1
    return passed / len(cases), f"{passed}/{len(cases)} cases passed"


def _register_sql_dialect_helpers(con: sqlite3.Connection) -> None:
    """Make the SQLite test DB tolerant of common SQL dialects (MySQL/Postgres).

    SQLite lacks YEAR()/MONTH()/DAY(), so a model that writes valid MySQL like
    `WHERE YEAR(created_at) = 2024` would otherwise error and be wrongly failed.
    We register those as UDFs (dates are ISO 'YYYY-MM-DD' strings here). We can't
    add EXTRACT() as a function (it's syntax, not a call), but YEAR/MONTH/DAY
    cover the overwhelmingly common case."""
    def _part(value, start, end):
        try:
            return int(str(value)[start:end])
        except (ValueError, TypeError):
            return None
    con.create_function("YEAR", 1, lambda v: _part(v, 0, 4))
    con.create_function("MONTH", 1, lambda v: _part(v, 5, 7))
    con.create_function("DAY", 1, lambda v: _part(v, 8, 10))


def check_sql_sqlite(task: Task, r: RequestResult, spec: dict) -> tuple[float, str]:
    sql = best_code(r.text, "sql")
    # Keep only the first statement (avoid seed/DDL the model might echo).
    sql = sql.split(";")[0].strip()
    if not re.search(r"\bselect\b", sql, re.I):
        return 0.0, "no SELECT statement found"

    names = set(spec.get("customer_names", []))
    expected = spec["expect_order"]
    con = sqlite3.connect(":memory:")
    _register_sql_dialect_helpers(con)
    try:
        con.executescript(spec["setup"])
        rows = con.execute(sql).fetchall()
    except sqlite3.Error as e:
        return 0.1, f"query error: {e}"
    finally:
        con.close()

    # Pull the customer-name cell out of each row (column order agnostic).
    got = []
    for row in rows:
        for cell in row:
            if isinstance(cell, str) and cell in names:
                got.append(cell)
                break

    exec_score = 0.3
    count_score = 0.2 if len(rows) == len(expected) else 0.0
    order_hits = sum(1 for i, n in enumerate(expected) if i < len(got) and got[i] == n)
    order_score = 0.5 * (order_hits / len(expected))
    total = exec_score + count_score + order_score
    return total, f"top-{len(expected)} order {order_hits}/{len(expected)} correct"


# --- text / format checks -------------------------------------------------

_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+\S")


def check_bullets(task: Task, r: RequestResult, spec: dict) -> tuple[float, str]:
    n = spec["n"]
    max_words = spec.get("max_words")
    bullets = [ln for ln in r.text.splitlines() if _BULLET.match(ln)]
    if not bullets:
        return 0.0, "no bullet points found"
    count_score = 1.0 if len(bullets) == n else max(0.0, 1 - abs(len(bullets) - n) / n)
    if max_words:
        ok = sum(1 for b in bullets if len(b.split()) - 1 <= max_words)
        words_score = ok / len(bullets)
        return 0.5 * count_score + 0.5 * words_score, (
            f"{len(bullets)} bullets, {ok} within {max_words} words"
        )
    return count_score, f"{len(bullets)} bullets (wanted {n})"


def check_keyword_coverage(task: Task, r: RequestResult, spec: dict) -> tuple[float, str]:
    text = r.text.lower()
    keywords = spec["keywords"]
    hits = 0
    for kw in keywords:
        options = kw if isinstance(kw, list) else [kw]
        if any(o.lower() in text for o in options):
            hits += 1
    return hits / len(keywords), f"{hits}/{len(keywords)} key concepts covered"


def check_word_count(task: Task, r: RequestResult, spec: dict) -> tuple[float, str]:
    words = len(re.findall(r"\S+", r.text))
    lo, hi = spec.get("min", 0), spec.get("max", 10**9)
    if lo <= words <= hi:
        return 1.0, f"{words} words (target {lo}-{hi})"
    # Linear falloff: zero once the miss equals the target span.
    span = max(hi - lo, 1)
    miss = (lo - words) if words < lo else (words - hi)
    return max(0.0, 1 - miss / span), f"{words} words (target {lo}-{hi})"


def check_stanza_count(task: Task, r: RequestResult, spec: dict) -> tuple[float, str]:
    n = spec["n"]
    blocks = [b for b in re.split(r"\n\s*\n", r.text.strip()) if b.strip()]
    score = 1.0 if len(blocks) == n else max(0.0, 1 - abs(len(blocks) - n) / n)
    return score, f"{len(blocks)} stanzas (wanted {n})"


def check_sentence_count(task: Task, r: RequestResult, spec: dict) -> tuple[float, str]:
    n = spec["n"]
    sents = [s for s in re.split(r"[.!?]+", r.text) if s.strip()]
    score = 1.0 if len(sents) == n else max(0.0, 1 - abs(len(sents) - n) / max(n, 1))
    return score, f"{len(sents)} sentence(s) (wanted {n})"


def check_acrostic(task: Task, r: RequestResult, spec: dict) -> tuple[float, str]:
    """First letters of the poem's lines should spell `word`. Lines whose first
    character isn't a letter (blank lines, a title with punctuation) are skipped
    leniently by taking the first alphabetic character of each non-empty line."""
    word = spec["word"].upper()
    initials = []
    for ln in r.text.splitlines():
        s = ln.strip()
        if s and s[0].isalpha():
            initials.append(s[0].upper())
    got = "".join(initials)
    hits = sum(1 for i, c in enumerate(word) if i < len(got) and got[i] == c)
    return hits / len(word), f"initials {got[:len(word)]!r} vs {word!r}: {hits}/{len(word)}"


def check_tool_call(task: Task, r: RequestResult, spec: dict) -> tuple[float, str]:
    calls = r.tool_calls or []
    if spec.get("expect_no_call"):
        if not calls:
            return 1.0, "correctly made no tool call"
        return 0.0, f"should not have called a tool, but called `{calls[0].get('name')}`"
    name = spec["name"]
    args_include = spec.get("args_include", {})
    match = next((c for c in calls if c.get("name") == name), None)
    if not match:
        called = ", ".join(c.get("name", "?") for c in calls) or "none"
        return 0.0, f"expected `{name}`, called: {called}"
    if not args_include:
        return 1.0, f"called `{name}`"
    try:
        args = json.loads(match.get("arguments") or "{}")
    except json.JSONDecodeError:
        return 0.5, f"called `{name}` but arguments not valid JSON"
    ok = sum(
        1 for k, v in args_include.items()
        if str(args.get(k, "")).lower() == str(v).lower()
    )
    return 0.5 + 0.5 * (ok / len(args_include)), (
        f"called `{name}`, args {ok}/{len(args_include)} correct"
    )


def check_tool_sequence(task: Task, r: RequestResult, spec: dict) -> tuple[float, str]:
    """Multi-turn agent: did the model call the tools the task requires?
    Order-insensitive — score is the fraction of expected tools that appear
    among the calls made across all turns."""
    expected = spec["names"]
    called = {c.get("name") for c in (r.tool_calls or [])}
    hits = sum(1 for n in expected if n in called)
    return hits / len(expected), (
        f"called {sorted(c for c in called if c)} — {hits}/{len(expected)} "
        f"required tools ({r.turns} turn(s))"
    )


CHECKS: dict[str, Callable[[Task, RequestResult, dict], tuple[float, str]]] = {
    "python_func": check_python_func,
    "tool_sequence": check_tool_sequence,
    "sql_sqlite": check_sql_sqlite,
    "bullets": check_bullets,
    "keyword_coverage": check_keyword_coverage,
    "word_count": check_word_count,
    "stanza_count": check_stanza_count,
    "sentence_count": check_sentence_count,
    "acrostic": check_acrostic,
    "tool_call": check_tool_call,
}


def run_check(task: Task, r: RequestResult, spec: dict) -> tuple[float, str]:
    fn = CHECKS.get(spec.get("type", ""))
    if fn is None:
        return 0.0, f"unknown check type '{spec.get('type')}'"
    try:
        return fn(task, r, spec)
    except Exception as e:  # a bad check shouldn't crash the whole run
        return 0.0, f"check error: {type(e).__name__}: {e}"
