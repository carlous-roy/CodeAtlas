"""
Chunk a codebase two ways, so the two can be compared.

Fixed-window chunking is the default in almost every RAG tutorial: slide a window
of N lines with some overlap and embed each window. It is trivial to implement and
it cuts functions in half. Half a function retrieves badly, because the half that
carries the signature and the docstring is a different chunk from the half that
carries the logic.

Structural chunking uses Tree-sitter to split on declaration boundaries instead:
one chunk per function, method or class, with the enclosing scope recorded. For
Markdown it splits on heading boundaries. Nothing is cut mid-definition.

Both strategies are written to disk so the retrieval evaluation can score the same
questions against each and report the difference rather than asserting one.
"""
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from tree_sitter import Language, Parser
import tree_sitter_python
import tree_sitter_java

CORPUS = Path("corpus")
OUT = Path("index")
OUT.mkdir(exist_ok=True)

# Fixed-window baseline settings. 40 lines with 10 lines of overlap is a common
# default and is roughly the size of the structural chunks, so the comparison is
# about *where* the boundaries fall rather than about chunk length.
WINDOW, OVERLAP = 40, 10

LANGS = {
    ".py": Language(tree_sitter_python.language()),
    ".java": Language(tree_sitter_java.language()),
}

# Node types that represent a nameable, self-contained unit of code.
DECL_NODES = {
    ".py": {"function_definition", "class_definition"},
    ".java": {"method_declaration", "class_declaration", "interface_declaration",
              "constructor_declaration"},
}

TEXT_EXT = {".md", ".txt", ".yml", ".yaml", ".sql", ".dax"}
CODE_EXT = {".py", ".java", ".js", ".jsx", ".ts", ".tsx"}
SKIP_DIRS = {"node_modules", ".git", "dist", "build", "target", "__pycache__", ".venv"}
SKIP_FILES = {"package-lock.json", ".DS_Store"}


@dataclass
class Chunk:
    chunk_id: str
    project: str
    path: str
    kind: str          # function | class | method | markdown_section | window
    name: str          # declaration name, or heading text, or line range
    start_line: int
    end_line: int
    text: str


def iter_files():
    for p in sorted(CORPUS.rglob("*")):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts) or p.name in SKIP_FILES:
            continue
        if p.suffix.lower() in TEXT_EXT | CODE_EXT:
            try:
                p.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            yield p


def project_of(p: Path) -> str:
    return p.relative_to(CORPUS).parts[0]


# ---------------------------------------------------------------- structural
def node_name(node, src: bytes) -> str:
    for child in node.children:
        if child.type in ("identifier", "type_identifier"):
            return src[child.start_byte:child.end_byte].decode("utf-8", "replace")
    return "<anonymous>"


def structural_code_chunks(p: Path, src: str):
    """One chunk per declaration. Anything outside a declaration (imports,
    module-level constants, the DAX-style config at the top of a file) is kept as
    a single preamble chunk rather than dropped, because module constants are
    exactly what a question like 'what is the retry limit' needs to hit."""
    ext = p.suffix.lower()
    lang = LANGS.get(ext)
    lines = src.splitlines()
    if lang is None:
        yield from window_chunks(p, src, kind_prefix="window")
        return

    parser = Parser(lang)
    tree = parser.parse(src.encode("utf-8"))
    src_b = src.encode("utf-8")
    wanted = DECL_NODES[ext]
    spans = []

    def walk(node, scope):
        if node.type in wanted:
            name = node_name(node, src_b)
            full = f"{scope}.{name}" if scope else name
            kind = ("class" if "class" in node.type
                    else "method" if scope else "function")
            spans.append((node.start_point[0], node.end_point[0], full, kind))
            # Recurse so methods inside a class also get their own chunk, but
            # keep the class chunk too: a question about the class as a whole
            # should be able to hit the class.
            for c in node.children:
                walk(c, full)
        else:
            for c in node.children:
                walk(c, scope)

    walk(tree.root_node, "")

    covered = set()
    for s, e, name, kind in spans:
        covered.update(range(s, e + 1))
        text = "\n".join(lines[s:e + 1])
        if text.strip():
            yield Chunk(
                chunk_id=f"{p}::{kind}::{name}::{s}",
                project=project_of(p), path=str(p), kind=kind, name=name,
                start_line=s + 1, end_line=e + 1, text=text,
            )

    preamble = [i for i in range(len(lines)) if i not in covered and lines[i].strip()]
    if preamble:
        s, e = preamble[0], preamble[-1]
        text = "\n".join(lines[i] for i in preamble)
        if len(text.strip()) > 40:
            yield Chunk(
                chunk_id=f"{p}::preamble::{s}",
                project=project_of(p), path=str(p), kind="preamble",
                name="module level", start_line=s + 1, end_line=e + 1, text=text,
            )


HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def structural_markdown_chunks(p: Path, src: str):
    """Split on headings, and carry the heading path into the chunk text. A
    section titled 'Retries' under 'Fault Tolerance' is ambiguous on its own;
    prefixing the path is what makes it retrievable by either term."""
    lines = src.splitlines()
    starts = [(i, m.group(1), m.group(2).strip())
              for i, l in enumerate(lines) if (m := HEADING.match(l))]
    if not starts:
        yield from window_chunks(p, src, kind_prefix="window")
        return
    if starts[0][0] > 0:
        starts.insert(0, (0, "#", "(preamble)"))

    stack = {}
    for idx, (ln, hashes, title) in enumerate(starts):
        end = starts[idx + 1][0] - 1 if idx + 1 < len(starts) else len(lines) - 1
        level = len(hashes)
        stack = {k: v for k, v in stack.items() if k < level}
        stack[level] = title
        path_str = " > ".join(stack[k] for k in sorted(stack))
        body = "\n".join(lines[ln:end + 1])
        if body.strip():
            yield Chunk(
                chunk_id=f"{p}::md::{ln}",
                project=project_of(p), path=str(p), kind="markdown_section",
                name=path_str, start_line=ln + 1, end_line=end + 1,
                text=f"{path_str}\n\n{body}",
            )


# ---------------------------------------------------------------- baseline
def window_chunks(p: Path, src: str, kind_prefix="window"):
    lines = src.splitlines()
    step = WINDOW - OVERLAP
    for s in range(0, max(len(lines), 1), step):
        seg = lines[s:s + WINDOW]
        if not any(l.strip() for l in seg):
            continue
        yield Chunk(
            chunk_id=f"{p}::win::{s}",
            project=project_of(p), path=str(p), kind=kind_prefix,
            name=f"lines {s + 1}-{s + len(seg)}",
            start_line=s + 1, end_line=s + len(seg), text="\n".join(seg),
        )
        if s + WINDOW >= len(lines):
            break


def build(strategy: str):
    out = []
    for p in iter_files():
        src = p.read_text(encoding="utf-8", errors="replace")
        if not src.strip():
            continue
        if strategy == "window":
            out.extend(window_chunks(p, src))
        elif p.suffix.lower() == ".md":
            out.extend(structural_markdown_chunks(p, src))
        elif p.suffix.lower() in LANGS:
            out.extend(structural_code_chunks(p, src))
        else:
            out.extend(window_chunks(p, src))
    return out


# Merged structural chunking.
#
# The first evaluation compared structural chunks (median 9 lines, 819 of them)
# against 40-line windows (403 of them) and the windows won. That comparison was
# not fair: chunk size and chunk count both changed at once, so the result could
# not distinguish "boundaries in the wrong place" from "chunks too small to carry
# context". A 9-line method embedded on its own is mostly a signature.
#
# This merges adjacent declarations within a file until each chunk clears a
# minimum size, keeping declaration boundaries intact. It never splits a
# declaration; it only groups neighbours. That holds the boundary policy constant
# and lets size be compared honestly.
# ---------------------------------------------------------------------------
MIN_LINES = 25


def merge_small(chunks, min_lines=MIN_LINES):
    from itertools import groupby
    out = []
    for path, group in groupby(chunks, key=lambda c: c.path):
        group = sorted(group, key=lambda c: c.start_line)
        buf = []
        for c in group:
            buf.append(c)
            span = buf[-1].end_line - buf[0].start_line + 1
            if span >= min_lines:
                out.append(_fuse(buf)); buf = []
        if buf:
            if out and out[-1].path == path and \
               (buf[-1].end_line - buf[0].start_line + 1) < min_lines // 2:
                out[-1] = _fuse([out[-1]] + buf)
            else:
                out.append(_fuse(buf))
    return out


def _fuse(buf):
    if len(buf) == 1:
        return buf[0]
    names = " | ".join(dict.fromkeys(c.name for c in buf))
    return Chunk(
        chunk_id=f"{buf[0].path}::merged::{buf[0].start_line}",
        project=buf[0].project, path=buf[0].path,
        kind="merged" if len({c.kind for c in buf}) > 1 else buf[0].kind,
        name=names[:200], start_line=buf[0].start_line, end_line=buf[-1].end_line,
        text="\n\n".join(c.text for c in buf),
    )


def build_merged():
    """Structural chunks merged up to MIN_LINES, so chunk size matches the
    window baseline and only the boundary policy differs between them."""
    return merge_small(build("structural"))


if __name__ == "__main__":
    from collections import Counter

    builders = {
        "structural": lambda: build("structural"),
        "structural_merged": build_merged,
        "window": lambda: build("window"),
    }
    for strategy, make in builders.items():
        chunks = make()
        path = OUT / f"chunks_{strategy}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(asdict(c)) + "\n")
        sizes = sorted(len(c.text.splitlines()) for c in chunks)
        print(f"{strategy:18s} {len(chunks):5,} chunks   "
              f"median {sizes[len(sizes)//2]:3d} lines   "
              f"p95 {sizes[int(len(sizes)*0.95)]:4d} lines   "
              f"max {sizes[-1]:4d}")
        print(f"{'':18s} kinds: {dict(Counter(c.kind for c in chunks).most_common())}")
