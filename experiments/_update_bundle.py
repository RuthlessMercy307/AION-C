"""
_update_bundle.py
Re-encodes updated source files into the base64 bundle in Cell 2.
Files updated: decoder/hybrid_layer.py, decoder/model.py,
               router/pipeline.py, experiments/train_cora_50m.py
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import json, base64, re, pathlib, textwrap

NB_PATH   = pathlib.Path('benchmark_cora_vs_transformer.ipynb')
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent   # AION-C/

FILES_TO_UPDATE = [
    'decoder/hybrid_layer.py',
    'decoder/model.py',
    'router/pipeline.py',
    'experiments/train_cora_50m.py',
]

# ── Load notebook ─────────────────────────────────────────────────────────────
with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell2_lines = nb['cells'][2]['source']
cell2 = ''.join(cell2_lines)

# ── Helper: re-encode one file into the bundle string ────────────────────────
def update_file_in_bundle(cell_src: str, bundle_key: str, file_path: pathlib.Path) -> str:
    """
    Replaces the base64 block for `bundle_key` in `cell_src` with a freshly
    encoded version of `file_path`.

    The bundle format is:
        'key': (
            'base64line...'
            'base64line...'
        ),
    """
    raw   = file_path.read_bytes()
    b64   = base64.b64encode(raw).decode('ascii')
    lines = textwrap.wrap(b64, 76)
    new_block = "(\n        '" + "'\n        '".join(lines) + "'\n    )"

    # Pattern: match from the key to the closing ),
    # The key is a dict key followed by ': (' ... '),'
    escaped_key = re.escape(bundle_key)
    pattern = rf"('{escaped_key}':\s*)\([^)]*?\)"
    replacement = r'\g<1>' + new_block
    new_src, n = re.subn(pattern, replacement, cell_src, count=1, flags=re.DOTALL)
    if n == 0:
        raise ValueError(f"Key '{bundle_key}' not found in bundle")
    return new_src

# ── Apply updates ─────────────────────────────────────────────────────────────
for bundle_key in FILES_TO_UPDATE:
    file_path = REPO_ROOT / bundle_key
    if not file_path.exists():
        raise FileNotFoundError(f"Source file not found: {file_path}")
    cell2 = update_file_in_bundle(cell2, bundle_key, file_path)
    print(f'  updated: {bundle_key}')

# ── Verify updated content round-trips correctly ─────────────────────────────
print('\nVerifying decoded content...')
for bundle_key in FILES_TO_UPDATE:
    escaped_key = re.escape(bundle_key)
    m = re.search(rf"'{escaped_key}':\s*\((.*?)\)", cell2, re.DOTALL)
    if not m:
        raise ValueError(f"Key '{bundle_key}' missing after update")
    b64_raw = re.sub(r"[\s']", '', m.group(1))
    decoded = base64.b64decode(b64_raw).decode('utf-8')
    disk    = (REPO_ROOT / bundle_key).read_text(encoding='utf-8')
    if decoded != disk:
        raise ValueError(f"Round-trip mismatch for {bundle_key}")
    print(f'  OK: {bundle_key}')

# ── Write back to notebook ────────────────────────────────────────────────────
nb['cells'][2]['source'] = cell2
with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'\nBundle updated and notebook saved ({NB_PATH})')
