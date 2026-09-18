"""The pinned corpus: fetching the projects at fixed commits, and a manifest
that records what was indexed so that a rerun can prove it indexed the same
files.

The manifest lists, per project, the repository and commit, and per indexed
file its path inside the project, line count and sha256. ``verify`` checks the
directory against the manifest and reports every difference, including
indexable files that the manifest does not know about.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from codeatlas import __version__
from codeatlas.chunking import iter_files, read_source

MANIFEST_VERSION = 1


@dataclass(frozen=True)
class ProjectPin:
    name: str
    repository: str
    commit: str

    @classmethod
    def parse(cls, spec: str) -> ProjectPin:
        """``name=https://host/repo.git@commit``"""
        try:
            name, rest = spec.split("=", 1)
            repository, commit = rest.rsplit("@", 1)
        except ValueError as exc:
            raise ValueError(f"expected name=url@commit, got {spec!r}") from exc
        if not name or not repository or len(commit) < 7:
            raise ValueError(f"expected name=url@commit, got {spec!r}")
        return cls(name, repository, commit)


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(args: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def describe_checkout(project_dir: Path) -> dict[str, str | None]:
    """Repository URL and commit of a project directory, when it is a git
    checkout; ``None`` values otherwise."""
    if not (project_dir / ".git").exists():
        return {"repository": None, "commit": None}
    try:
        url = _git(["config", "--get", "remote.origin.url"], cwd=project_dir)
    except subprocess.CalledProcessError:
        url = None
    commit = _git(["rev-parse", "HEAD"], cwd=project_dir)
    return {"repository": url, "commit": commit}


def file_entries(corpus_dir: Path) -> list[dict[str, str | int]]:
    """One entry per indexable file, sorted by project and path."""
    entries: list[dict[str, str | int]] = []
    for p in iter_files(corpus_dir):
        rel = p.relative_to(corpus_dir)
        src = read_source(p)
        assert src is not None
        entries.append(
            {
                "project": rel.parts[0],
                "path": Path(*rel.parts[1:]).as_posix(),
                "lines": len(src.splitlines()),
                "sha256": sha256_of(p),
            }
        )
    return entries


def build_manifest(corpus_dir: Path, pins: list[ProjectPin] | None = None) -> dict:
    """Manifest of the current corpus directory. Repository and commit come
    from ``pins`` when given, else from each project's git checkout."""
    entries = file_entries(corpus_dir)
    names = sorted({str(e["project"]) for e in entries})
    projects: dict[str, dict] = {}
    by_name = {p.name: p for p in (pins or [])}
    for name in names:
        if name in by_name:
            info: dict[str, str | None] = {
                "repository": by_name[name].repository,
                "commit": by_name[name].commit,
            }
        else:
            info = describe_checkout(corpus_dir / name)
        files = [e for e in entries if e["project"] == name]
        info["files"] = len(files)
        info["lines"] = sum(int(e["lines"]) for e in files)
        projects[name] = info
    return {
        "manifest_version": MANIFEST_VERSION,
        "codeatlas_version": __version__,
        "generated": date.today().isoformat(),
        "projects": projects,
        "totals": {"files": len(entries), "lines": sum(int(e["lines"]) for e in entries)},
        "files": entries,
    }


def write_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def load_manifest(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def manifest_hash(path: Path) -> str:
    return sha256_of(path)


def verify_manifest(corpus_dir: Path, manifest: dict) -> list[str]:
    """Every way the corpus directory differs from the manifest; empty when it
    matches exactly."""
    problems: list[str] = []
    expected = {(str(e["project"]), str(e["path"])): e for e in manifest["files"]}
    actual = {(str(e["project"]), str(e["path"])): e for e in file_entries(corpus_dir)}
    for key in sorted(expected.keys() - actual.keys()):
        problems.append(f"missing: {key[0]}/{key[1]}")
    for key in sorted(actual.keys() - expected.keys()):
        problems.append(f"not in manifest: {key[0]}/{key[1]}")
    for key in sorted(expected.keys() & actual.keys()):
        exp, act = expected[key], actual[key]
        if exp["sha256"] != act["sha256"]:
            problems.append(f"content differs: {key[0]}/{key[1]}")
        elif exp["lines"] != act["lines"]:
            problems.append(f"line count differs: {key[0]}/{key[1]} ({exp['lines']} vs {act['lines']})")
    for name, info in manifest["projects"].items():
        commit = info.get("commit")
        checkout = describe_checkout(corpus_dir / name)
        if commit and checkout["commit"] and checkout["commit"] != commit:
            problems.append(f"{name}: checked out {checkout['commit'][:10]}, manifest pins {commit[:10]}")
    return problems


def pins_from_manifest(manifest: dict) -> list[ProjectPin]:
    pins: list[ProjectPin] = []
    for name, info in manifest["projects"].items():
        if not info.get("repository") or not info.get("commit"):
            raise ValueError(f"manifest has no repository or commit for {name}")
        pins.append(ProjectPin(name, info["repository"], info["commit"]))
    return pins


def fetch(pins: list[ProjectPin], corpus_dir: Path) -> None:
    """Clone each project and check out its pinned commit. An existing
    checkout is reused: it is fetched and reset to the pinned commit."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for pin in pins:
        dest = corpus_dir / pin.name
        if (dest / ".git").exists():
            _git(["-C", str(dest), "fetch", "--quiet", "origin"])
        else:
            if dest.exists():
                raise FileExistsError(f"{dest} exists and is not a git checkout")
            _git(["clone", "--quiet", pin.repository, str(dest)])
        _git(["-C", str(dest), "checkout", "--quiet", "--detach", pin.commit])
        head = _git(["-C", str(dest), "rev-parse", "HEAD"])
        if not head.startswith(pin.commit):
            raise RuntimeError(f"{pin.name}: checked out {head}, expected {pin.commit}")
