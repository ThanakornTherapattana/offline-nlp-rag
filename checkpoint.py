"""Save local source snapshots without duplicating models or course data."""
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    excluded = {"checkpoints", "data", "index", "output", ".git", ".venv", "venv", "__pycache__", ".pytest_cache"}
    suffixes = {".py", ".md", ".txt", ".toml", ".yaml", ".yml", ".json", ".bat", ".ps1", ".ini", ".cfg"}
    files = [p for p in root.rglob("*") if p.is_file()
             and not any(part.lower() in excluded for part in p.relative_to(root).parts[:-1])
             and (p.suffix.lower() in suffixes or p.name == ".gitignore")]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = root / "checkpoints" / stamp
    target.mkdir(parents=True)
    manifest = {"reason": " ".join(sys.argv[1:]) or "manual checkpoint", "files": {}}
    for path in files:
        rel = path.relative_to(root)
        destination = target / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        manifest["files"][rel.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
