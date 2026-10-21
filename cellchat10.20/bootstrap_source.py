"""Verify the pinned baseline archive without overwriting development files.

The archive was produced by git archive from the local object database, with
commit/tree checked against the saved GitHub API response.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parent
COMMIT = '986bf7ef909f1e15059aad8a9a9ee31ac675ffc3'
TREE = 'f3c44d7cc552bbf6ac14bec25bf0d89d98c1f65d'
ARCHIVE_SHA256 = '46ef4c548cd97365ba61e0335fc58424425efa5dd4e88b0211177181548ef894'
metadata = json.loads((ROOT / 'provenance/github_commit.json').read_text(encoding='utf-8'))
assert metadata['sha'] == COMMIT
assert metadata['commit']['tree']['sha'] == TREE

# The recorded archive hash also permits verification after moving this whole
# research directory to another machine, without needing the original sibling
# repository or a live GitHub request.
repository = ROOT.parent / 'CellChatPy_GitHub'
local_git_checked = (repository / '.git').exists()
if local_git_checked:
    tree = subprocess.check_output(
        ['git', '-C', str(repository), 'rev-parse', COMMIT + '^{tree}'], text=True).strip()
    assert tree == TREE
archive = ROOT / 'provenance/upstream.zip'
archive_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
assert archive_digest == ARCHIVE_SHA256, 'Pinned source archive changed'
with zipfile.ZipFile(archive) as z:
    files = [x for x in z.infolist() if not x.is_dir()]
    for item in files:
        assert (ROOT / 'upstream' / item.filename).read_bytes() == z.read(item), 'Baseline modified: ' + item.filename
print(json.dumps({'commit': COMMIT, 'tree': TREE, 'verified_files': len(files),
                  'archive_sha256': archive_digest, 'local_git_checked': local_git_checked}, indent=2))
