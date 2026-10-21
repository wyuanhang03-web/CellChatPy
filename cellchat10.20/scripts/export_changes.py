"""Export a reviewable source diff and fingerprints without touching the baseline."""
from pathlib import Path
import difflib
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old, new = ROOT / 'upstream', ROOT / 'development'
    paths = {Path('README.md'), Path('pyproject.toml')}
    for base in (old, new):
        for directory, suffix in [('CellChatPy', '*.py'), ('tests', '*.py'), ('docs', '*.md')]:
            paths.update(p.relative_to(base) for p in (base / directory).rglob(suffix)
                         if '__pycache__' not in p.parts)
    changes, patch = [], []
    for rel in sorted(paths):
        left, right = old / rel, new / rel
        a = left.read_text(encoding='utf-8') if left.exists() else ''
        b = right.read_text(encoding='utf-8') if right.exists() else ''
        if a == b:
            continue
        changes.append({'path': rel.as_posix(), 'kind': 'modified' if left.exists() else 'added',
                        'upstream_sha256': sha(left) if left.exists() else None,
                        'development_sha256': sha(right) if right.exists() else None})
        patch.extend(difflib.unified_diff(a.splitlines(True), b.splitlines(True),
                                         'a/' + rel.as_posix(), 'b/' + rel.as_posix()))
    notebook_changes = []
    for right in new.rglob('*.ipynb'):
        rel = right.relative_to(new)
        left = old / rel
        if not left.exists():
            continue
        a, b = json.loads(left.read_text(encoding='utf-8')), json.loads(right.read_text(encoding='utf-8'))
        for i, (x, y) in enumerate(zip(a['cells'], b['cells'])):
            sx, sy = ''.join(x.get('source', '')), ''.join(y.get('source', ''))
            if sx != sy:
                notebook_changes.append({'path': rel.as_posix(), 'cell_index': i,
                                         'old_source': sx, 'new_source': sy})
    manifest = {'base_commit': '986bf7ef909f1e15059aad8a9a9ee31ac675ffc3',
                'source_changes': changes, 'notebook_source_changes': notebook_changes}
    (ROOT / 'provenance/development_changes.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (ROOT / 'provenance/development.patch').write_text(''.join(patch), encoding='utf-8')
    print(json.dumps({'changed_text_files': len(changes), 'changed_notebook_cells': len(notebook_changes)}))


if __name__ == '__main__':
    main()
