"""Docs consistency linter — the P26/B5 fix for three documented accidents.

The repo has repeatedly shipped stale docs (engine-vs matrix one month behind,
self-referential commit counts, a stale pytest count in README) because five
docs are updated by hand every round.  This script checks what a round can
mechanically get wrong:

  1. self-referential commit counts  — "领先 origin N 个" style hardcoding
     drifts on the very commit that fixes it; the only allowed form is the
     git-log command reference.
  2. dead relative links            — every `(xxx.md...)` / `(assets/...)`
     target in a tracked markdown file must exist on disk.
  3. image targets                  — every README image path must exist.
  4. command paths                  — code spans inside README/benchmark
     fenced blocks that name repo scripts/tests must point at real files.

Exit 0 = clean, 1 = problems listed.  Pure CPU, stdlib only.
Run:  python3 scripts/check_docs.py
"""
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LINK = re.compile(r"(?<!!)\[([^\]]*)\]\(([^)]+)\)")
IMAGE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
COMMAND = re.compile(r"(?:PYTHONPATH=\.\s+)?(?:python3?\s+)?((?:scripts|tests)/[\w/.\-]+\.(?:py|sh))")
STALE_COUNT = re.compile(r"领先\s*origin\s*[0-9一二两三四五六七八九十几]+\s*个\s*commit")


def tracked_md():
    """Git-tracked markdown, minus frozen-history archives (stale links there
    are historical record, kept verbatim)."""
    files = subprocess.run(["git", "ls-files", "*.md"], cwd=REPO,
                           capture_output=True, text=True).stdout.split()
    out = []
    for rel in files:
        if rel.startswith(("benchmarks/archive/", "scripts/archive/", "tests/archive/")):
            continue
        out.append(REPO / rel)
    return out


def strip_target(target: str) -> str:
    target = target.strip()
    if target.startswith(("<", "http://", "https://", "mailto:", "#")):
        return ""
    return target.split("#", 1)[0]


def main() -> int:
    problems = []

    for p in tracked_md():
        rel = p.relative_to(REPO).as_posix()
        text = p.read_text(encoding="utf-8")

        # 1. self-referential commit counts anywhere outside the sanctioned
        #    command-reference form.
        for m in STALE_COUNT.finditer(text):
            problems.append(f"{rel}: hardcoded commit count -> {m.group(0)!r} "
                            f"(use: git log --oneline origin/main..HEAD)")

        # fenced blocks first (commands only count inside them)
        fences = re.findall(r"```[a-z]*\n(.*?)```", text, re.S)
        for fence in fences:
            for m in COMMAND.finditer(fence):
                target = REPO / m.group(1)
                if not target.exists():
                    problems.append(f"{rel}: command names missing file {m.group(1)}")

        # 2/3. links and images (relative targets resolve against the FILE's
        # directory — markdown semantics, not the repo root).  Scan with
        # fenced blocks and inline code stripped: `[x](y)` inside code is
        # code, not a link (e.g. "[0.5, 100, 448](clip)" in a repro snippet).
        prose = re.sub(r"```[a-z]*\n.*?```", "", text, flags=re.S)
        prose = re.sub(r"`[^`]*`", "", prose)
        base = p.parent
        for m in IMAGE.finditer(prose):
            t = strip_target(m.group(1))
            if t and not (base / t).exists():
                problems.append(f"{rel}: image target missing -> {t}")
        for m in LINK.finditer(prose):
            t = strip_target(m.group(2))
            if t and not t.endswith((".png", ".jpg", ".svg")):
                if not (base / t).exists():
                    problems.append(f"{rel}: dead link -> {t}")

    if problems:
        print(f"[check_docs] {len(problems)} problem(s):")
        for line in problems:
            print("  -", line)
        return 1
    print("[check_docs] clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
