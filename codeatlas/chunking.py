"""Three chunkers over the same files, so they can be compared on the same questions.

``structural``
    One chunk per declaration. Tree-sitter parses Python and Java; a class,
    interface, enum or record is split into a header (its signature, fields and
    docstring) and one chunk per member declaration, so no line is indexed
    twice. Lines outside every declaration (imports, module constants, a
    ``__main__`` block) form ``module level`` chunks; lines between members of a
    class form ``class body`` chunks. Nothing with an alphanumeric character is
    dropped. Markdown splits on headings and carries the heading path into the
    chunk text. Files without a grammar fall back to windows.

``structural_merged``
    The structural units of a file, packed greedily in file order into chunks
    that still fit the embedder window. A declaration is never split by the
    merge; only the size policy changes.

``window``
    Consecutive lines packed until the embedder window is full, with the last
    quarter of each window repeated at the start of the next one.

Every chunk of every chunker fits the embedder's window together with the
``path :: name`` prefix it is embedded with (see :mod:`codeatlas.models`), so
the encoder sees the whole chunk. A structural unit longer than the window is
split into consecutive parts named ``name (2/3)``; a single line longer than
the window is split on whitespace.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import tree_sitter_java
import tree_sitter_python
from tree_sitter import Language, Node, Parser

from codeatlas.models import TokenCounter, payload_budget

CHUNKINGS = ("structural", "structural_merged", "window")

TEXT_EXT = frozenset({".md", ".txt", ".yml", ".yaml", ".sql", ".dax"})
CODE_EXT = frozenset({".py", ".java", ".js", ".jsx", ".ts", ".tsx"})
INDEXED_EXT = TEXT_EXT | CODE_EXT
SKIP_DIRS = frozenset({"node_modules", ".git", "dist", "build", "target", "__pycache__", ".venv"})
SKIP_FILES = frozenset({"package-lock.json", ".DS_Store"})

# Logical prefix of every chunk path, independent of where the corpus directory
# lives on disk. The golden set labels files with this prefix.
CORPUS_PREFIX = "corpus"

# Share of a window's lines repeated at the start of the next window.
WINDOW_OVERLAP = 0.25
NAME_MAX_CHARS = 200
# Reserved for the " (i/n)" suffix a split part gets, in word pieces.
PART_SUFFIX_RESERVE = 8

_ALNUM = re.compile(r"[A-Za-z0-9]")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    chunk_id: str
    project: str
    path: str
    kind: str
    name: str
    start_line: int
    end_line: int
    text: str

    def payload(self) -> str:
        """What the embedder and the prefixed reranker see."""
        return f"{self.path} :: {self.name}\n{self.text}"


@dataclass(frozen=True)
class LanguageSpec:
    language: Language
    kinds: dict[str, str]
    containers: frozenset[str]
    comments: frozenset[str]


LANGUAGES: dict[str, LanguageSpec] = {
    ".py": LanguageSpec(
        language=Language(tree_sitter_python.language()),
        kinds={"function_definition": "function", "class_definition": "class"},
        containers=frozenset({"class_definition"}),
        comments=frozenset({"comment"}),
    ),
    ".java": LanguageSpec(
        language=Language(tree_sitter_java.language()),
        kinds={
            "method_declaration": "method",
            "constructor_declaration": "constructor",
            "class_declaration": "class",
            "interface_declaration": "interface",
            "enum_declaration": "enum",
            "record_declaration": "record",
        },
        containers=frozenset({"class_declaration", "interface_declaration", "enum_declaration", "record_declaration"}),
        comments=frozenset({"line_comment", "block_comment"}),
    ),
}


# ------------------------------------------------------------------ files
def iter_files(corpus_dir: Path) -> Iterator[Path]:
    """Every indexable file under ``corpus_dir`` in a stable order."""
    for p in sorted(corpus_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(corpus_dir)
        if any(part in SKIP_DIRS for part in rel.parts) or p.name in SKIP_FILES:
            continue
        if p.suffix.lower() not in INDEXED_EXT:
            continue
        if read_source(p) is None:
            continue
        yield p


def read_source(p: Path) -> str | None:
    """UTF-8 text of a file, or None when it is not decodable or is empty."""
    try:
        src = p.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None
    return src if src.strip() else None


def logical_path(corpus_dir: Path, p: Path) -> str:
    """``corpus/<project>/<path>`` for a file under the corpus directory."""
    return f"{CORPUS_PREFIX}/{p.relative_to(corpus_dir).as_posix()}"


def project_of(path: str) -> str:
    """The project directory of a logical chunk path (``corpus/<project>/...``)."""
    parts = path.split("/")
    return parts[1] if len(parts) > 1 and parts[0] == CORPUS_PREFIX else "?"


# ------------------------------------------------------------------ units
@dataclass
class Unit:
    """A contiguous run of lines with a name and a kind. Rows are 0-based
    and inclusive. ``header`` is text prepended to the chunk text (the heading
    path of a Markdown section). ``atomic`` marks a piece of a single line
    that must not be merged with its neighbours."""

    start: int
    end: int
    name: str
    kind: str
    header: str = ""
    text: str | None = None
    atomic: bool = False


@dataclass
class Decl:
    start: int
    end: int
    name: str
    kind: str
    children: list[Decl] = field(default_factory=list)


def _node_text(node: Node, src: bytes) -> str:
    return src[node.start_byte : node.end_byte].decode("utf-8", "replace")


def _leading_start(node: Node, spec: LanguageSpec, lines: list[str]) -> int:
    """Start row of a declaration including the comments directly above it."""
    start = node.start_point.row
    prev = node.prev_sibling
    while (
        prev is not None
        and prev.type in spec.comments
        and prev.end_point.row + 1 >= start
        and not lines[prev.start_point.row][: prev.start_point.column].strip()
    ):
        start = prev.start_point.row
        prev = prev.prev_sibling
    return start


def _as_declaration(
    node: Node, scope: str, scope_kind: str, spec: LanguageSpec, src: bytes, lines: list[str]
) -> Decl | None:
    inner = node
    if node.type == "decorated_definition":
        definition = node.child_by_field_name("definition")
        if definition is None:
            return None
        inner = definition
    kind = spec.kinds.get(inner.type)
    if kind is None:
        return None
    name_node = inner.child_by_field_name("name")
    name = _node_text(name_node, src) if name_node is not None else "<anonymous>"
    full = f"{scope}.{name}" if scope else name
    if kind == "function" and scope_kind in ("class", "interface", "enum", "record"):
        kind = "method"
    children: list[Decl] = []
    if inner.type in spec.containers:
        children = _collect(inner, full, kind, spec, src, lines)
    return Decl(_leading_start(node, spec, lines), node.end_point.row, full, kind, children)


def _collect(node: Node, scope: str, scope_kind: str, spec: LanguageSpec, src: bytes, lines: list[str]) -> list[Decl]:
    decls: list[Decl] = []
    for child in node.children:
        decl = _as_declaration(child, scope, scope_kind, spec, src, lines)
        if decl is None:
            decls.extend(_collect(child, scope, scope_kind, spec, src, lines))
        else:
            decls.append(decl)
    return decls


def _trim(lines: list[str], lo: int, hi: int) -> tuple[int, int]:
    while lo <= hi and not _ALNUM.search(lines[lo]):
        lo += 1
    while hi >= lo and not _ALNUM.search(lines[hi]):
        hi -= 1
    return lo, hi


def _gap_units(lines: list[str], lo: int, hi: int, scope: str, scope_kind: str) -> list[Unit]:
    lo, hi = _trim(lines, lo, hi)
    if lo > hi:
        return []
    if scope:
        return [Unit(lo, hi, scope, f"{scope_kind} body")]
    return [Unit(lo, hi, "module level", "preamble")]


def _decl_units(decl: Decl, lines: list[str]) -> list[Unit]:
    if not decl.children:
        return [Unit(decl.start, decl.end, decl.name, decl.kind)]
    units: list[Unit] = []
    first = decl.children[0].start
    if first > decl.start:
        units.append(Unit(decl.start, first - 1, decl.name, decl.kind))
    units.extend(_partition(decl.children, first, decl.end, decl.name, decl.kind, lines))
    return units


def _partition(decls: list[Decl], lo: int, hi: int, scope: str, scope_kind: str, lines: list[str]) -> list[Unit]:
    units: list[Unit] = []
    cursor = lo
    for decl in decls:
        units.extend(_gap_units(lines, cursor, decl.start - 1, scope, scope_kind))
        units.extend(_decl_units(decl, lines))
        cursor = decl.end + 1
    units.extend(_gap_units(lines, cursor, hi, scope, scope_kind))
    return units


def code_units(src: str, spec: LanguageSpec) -> list[Unit]:
    """Partition a source file into declaration, header, body and module-level
    units. Every line with an alphanumeric character belongs to exactly one
    unit and no two units overlap."""
    lines = src.splitlines()
    src_b = src.encode("utf-8")
    parser = _parser_for(spec)
    tree = parser.parse(src_b)
    decls = _collect(tree.root_node, "", "", spec, src_b, lines)
    return _partition(decls, 0, len(lines) - 1, "", "", lines)


_PARSERS: dict[int, Parser] = {}


def _parser_for(spec: LanguageSpec) -> Parser:
    """One long-lived parser per language."""
    parser = _PARSERS.get(id(spec))
    if parser is None:
        parser = Parser(spec.language)
        _PARSERS[id(spec)] = parser
    return parser


def markdown_units(src: str) -> list[Unit]:
    """One unit per heading section; the name is the heading path."""
    lines = src.splitlines()
    starts = [(i, len(m.group(1)), m.group(2).strip()) for i, line in enumerate(lines) if (m := _HEADING.match(line))]
    if not starts:
        return []
    if starts[0][0] > 0:
        starts.insert(0, (0, 1, "(preamble)"))
    units: list[Unit] = []
    stack: dict[int, str] = {}
    for idx, (row, level, title) in enumerate(starts):
        end = starts[idx + 1][0] - 1 if idx + 1 < len(starts) else len(lines) - 1
        stack = {k: v for k, v in stack.items() if k < level}
        stack[level] = title
        heading_path = " > ".join(stack[k] for k in sorted(stack))
        lo, hi = _trim(lines, row, end)
        if lo > hi:
            continue
        units.append(Unit(lo, hi, heading_path, "markdown_section", header=heading_path))
    return units


# ------------------------------------------------------------------ sizing
class _Sizer:
    """Token accounting for one file: cached per-line counts, prefix counts."""

    def __init__(self, path: str, lines: list[str], count: TokenCounter, budget: int):
        self.path = path
        self.lines = lines
        self.count = count
        self.budget = budget
        self.line_tokens = [count(line) for line in lines]

    def span_tokens(self, lo: int, hi: int) -> int:
        return sum(self.line_tokens[lo : hi + 1])

    def prefix_tokens(self, name: str, header: str = "") -> int:
        n = self.count(f"{self.path} :: {name}\n")
        if header:
            n += self.count(f"{header}\n\n")
        return n

    def fits(self, name: str, header: str, lo: int, hi: int) -> bool:
        return self.prefix_tokens(name, header) + self.span_tokens(lo, hi) <= self.budget


def _split_line(line: str, budget: int, count: TokenCounter) -> list[str]:
    """Split one line on whitespace into pieces of at most ``budget`` tokens.
    A single word longer than the budget is cut into 64-character slices."""
    words: list[str] = []
    for word in line.split():
        if count(word) > budget:
            words.extend(word[i : i + 64] for i in range(0, len(word), 64))
        else:
            words.append(word)
    pieces: list[str] = []
    buf: list[str] = []
    used = 0
    for word in words:
        n = count(word)
        if buf and used + n > budget:
            pieces.append(" ".join(buf))
            buf, used = [], 0
        buf.append(word)
        used += n
    if buf:
        pieces.append(" ".join(buf))
    return pieces or [""]


def _fit_unit(unit: Unit, sizer: _Sizer) -> list[Unit]:
    """Split a unit that does not fit the window into consecutive parts."""
    if sizer.fits(unit.name, unit.header, unit.start, unit.end):
        return [unit]
    budget = sizer.budget - sizer.prefix_tokens(unit.name, unit.header) - PART_SUFFIX_RESERVE
    budget = max(budget, 16)
    parts: list[Unit] = []
    lo = unit.start
    while lo <= unit.end:
        hi = lo
        while hi + 1 <= unit.end and sizer.span_tokens(lo, hi + 1) <= budget:
            hi += 1
        if hi == lo and sizer.line_tokens[lo] > budget:
            for piece in _split_line(sizer.lines[lo], budget, sizer.count):
                parts.append(Unit(lo, lo, unit.name, unit.kind, unit.header, text=piece, atomic=True))
        else:
            parts.append(Unit(lo, hi, unit.name, unit.kind, unit.header))
        lo = hi + 1
    total = len(parts)
    for i, part in enumerate(parts, 1):
        part.name = f"{unit.name} ({i}/{total})"
    return parts


def fit_units(units: list[Unit], sizer: _Sizer) -> list[Unit]:
    """Every unit, with those that do not fit the window split into parts."""
    out: list[Unit] = []
    for unit in units:
        out.extend(_fit_unit(unit, sizer))
    return out


def merge_units(units: list[Unit], sizer: _Sizer) -> list[Unit]:
    """Pack consecutive units of one file into the largest runs that still fit
    the window. The merged text is the exact source span from the first unit's
    first line to the last unit's last line."""
    out: list[Unit] = []
    buf: list[Unit] = []

    def flush() -> None:
        if not buf:
            return
        if len(buf) == 1:
            out.append(buf[0])
        else:
            out.append(
                Unit(
                    buf[0].start,
                    buf[-1].end,
                    _joined_name(buf),
                    "merged" if len({u.kind for u in buf}) > 1 else buf[0].kind,
                    buf[0].header,
                )
            )
        buf.clear()

    for unit in units:
        if unit.atomic:
            flush()
            out.append(unit)
            continue
        if buf and sizer.fits(_joined_name([*buf, unit]), buf[0].header, buf[0].start, unit.end):
            buf.append(unit)
            continue
        flush()
        buf.append(unit)
    flush()
    return out


def _joined_name(units: list[Unit]) -> str:
    return " | ".join(dict.fromkeys(u.name for u in units))[:NAME_MAX_CHARS]


def window_units(sizer: _Sizer) -> list[Unit]:
    """Windows of consecutive lines packed to the token budget, overlapping by
    ``WINDOW_OVERLAP`` of the previous window's lines."""
    lines = sizer.lines
    n = len(lines)
    digits = f"lines {n}-{n}"
    budget = sizer.budget - sizer.prefix_tokens(digits)
    out: list[Unit] = []
    lo = 0
    while lo < n:
        hi = lo
        while hi + 1 < n and sizer.span_tokens(lo, hi + 1) <= budget:
            hi += 1
        if hi == lo and sizer.line_tokens[lo] > budget:
            pieces = _split_line(lines[lo], max(budget - PART_SUFFIX_RESERVE, 16), sizer.count)
            for i, piece in enumerate(pieces, 1):
                out.append(
                    Unit(lo, lo, f"lines {lo + 1}-{lo + 1} ({i}/{len(pieces)})", "window", text=piece, atomic=True)
                )
            lo += 1
            continue
        if any(_ALNUM.search(line) for line in lines[lo : hi + 1]):
            out.append(Unit(lo, hi, f"lines {lo + 1}-{hi + 1}", "window"))
        if hi + 1 >= n:
            break
        overlap = int((hi - lo + 1) * WINDOW_OVERLAP)
        lo = max(hi + 1 - overlap, lo + 1)
    return out


# ------------------------------------------------------------------ chunks
def _to_chunk(unit: Unit, path: str, project: str, lines: list[str]) -> Chunk:
    body = unit.text if unit.text is not None else "\n".join(lines[unit.start : unit.end + 1])
    text = f"{unit.header}\n\n{body}" if unit.header else body
    return Chunk(
        chunk_id=f"{path}::{unit.kind}::{unit.name}::{unit.start + 1}",
        project=project,
        path=path,
        kind=unit.kind,
        name=unit.name,
        start_line=unit.start + 1,
        end_line=unit.end + 1,
        text=text,
    )


def chunk_file(
    path: str,
    src: str,
    chunking: str,
    count_tokens: TokenCounter,
    budget: int | None = None,
) -> list[Chunk]:
    """Chunk one file. ``path`` is the logical path (``corpus/<project>/...``)."""
    if chunking not in CHUNKINGS:
        raise ValueError(f"unknown chunking {chunking!r}; expected one of {CHUNKINGS}")
    lines = src.splitlines()
    if not lines:
        return []
    sizer = _Sizer(path, lines, count_tokens, payload_budget() if budget is None else budget)
    ext = Path(path).suffix.lower()
    units: list[Unit] | None = None
    if chunking != "window":
        if ext in LANGUAGES:
            units = code_units(src, LANGUAGES[ext])
        elif ext == ".md":
            units = markdown_units(src) or None
    if units is None:
        units = window_units(sizer)
    else:
        units = fit_units(units, sizer)
        if chunking == "structural_merged":
            units = merge_units(units, sizer)
    project = project_of(path)
    chunks = [_to_chunk(u, path, project, lines) for u in units]
    _dedupe_ids(chunks)
    return chunks


def _dedupe_ids(chunks: list[Chunk]) -> None:
    seen: dict[str, int] = {}
    for c in chunks:
        n = seen.get(c.chunk_id, 0)
        seen[c.chunk_id] = n + 1
        if n:
            c.chunk_id = f"{c.chunk_id}#{n}"


def build_chunks(corpus_dir: Path, chunking: str, count_tokens: TokenCounter) -> list[Chunk]:
    """Chunk every indexable file under ``corpus_dir`` with one chunker."""
    out: list[Chunk] = []
    for p in iter_files(corpus_dir):
        src = read_source(p)
        if src is None:
            continue
        out.extend(chunk_file(logical_path(corpus_dir, p), src, chunking, count_tokens))
    return out


def write_chunks(chunks: list[Chunk], path: Path) -> None:
    """Write chunks as JSON lines, one object per chunk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def read_chunks(path: Path) -> list[Chunk]:
    """Read a chunk file written by :func:`write_chunks`."""
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]


def size_summary(chunks: list[Chunk]) -> dict[str, float | int]:
    """Chunk count, line-count percentiles and file count of one chunking."""
    sizes = sorted(c.end_line - c.start_line + 1 for c in chunks)
    if not sizes:
        return {"n_chunks": 0}
    return {
        "n_chunks": len(sizes),
        "median_lines": sizes[len(sizes) // 2],
        "p95_lines": sizes[min(len(sizes) - 1, int(len(sizes) * 0.95))],
        "max_lines": sizes[-1],
        "n_files": len({c.path for c in chunks}),
    }
