"""Check that the deliverable wheel contains the tested code and database."""
from pathlib import Path
from hashlib import sha256
import json
import zipfile
from email.parser import BytesParser

ROOT = Path(__file__).resolve().parents[1]


def main():
    wheels = list((ROOT / "dist").glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected one deliverable wheel; found {len(wheels)}")
    wheel = wheels[0]
    expected = sorted(
        p for p in (ROOT / "development/CellChatPy").rglob("*")
        if p.is_file() and p.suffix in {".py", ".csv", ".mtx"}
    )
    checks = []
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        for path in expected:
            relative = path.relative_to(ROOT / "development").as_posix()
            payload = archive.read(relative) if relative in names else None
            checks.append({"path": relative,
                           "present": payload is not None,
                           "matches_source": payload == path.read_bytes()})
        metadata_name = next(n for n in names if n.endswith(".dist-info/METADATA"))
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
    passed = all(c["present"] and c["matches_source"] for c in checks)
    passed = passed and metadata["Version"] == "1.1.0+streaming.20261020"
    result = {"passed": passed, "wheel": str(wheel.relative_to(ROOT)),
              "sha256": sha256(wheel.read_bytes()).hexdigest(),
              "version": metadata["Version"], "checked_files": len(checks),
              "requirements": metadata.get_all("Requires-Dist"), "files": checks}
    (ROOT / "validation/wheel_integrity.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"files", "requirements"}}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
