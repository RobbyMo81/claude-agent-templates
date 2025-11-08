import os
import shutil
import json
from datetime import datetime
from pathlib import Path
import argparse

# === CLI ARGUMENTS ===
parser = argparse.ArgumentParser(description="Archive non-backend files from Helios project.")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--dry-run', action='store_true', help='Simulate the archive without moving files')
group.add_argument('--run', action='store_true', help='Execute the archive and move files')
args = parser.parse_args()

DRY_RUN = args.dry_run
# === CONFIGURATION ===
PROJECT_ROOT = Path(r"C:\Users\RobMo\OneDrive\Documents\helios")
ARCHIVE_DIR = Path(r"C:\Users\RobMo\OneDrive\Documents\helios\archive")
GENERATE_JSON = True  # Set to False to skip JSON audit trail

# === BACKEND FILES TO PRESERVE ===
PRESERVE = {
    "server.py",
    "agent.py",
    "trainer.py",
    "memory_store.py",
    "metacognition.py",
    "decision_engine.py",
    "cross_model_analytics.py",
    "requirements.txt",
    "environment.yml",
    ".venv",
    ".git",
    ".gitignore"
}

# === INITIALIZE LOGGING ===
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
manifest_lines = [
    "# Helios Archive Manifest",
    "",
    f"Archived on: {timestamp}",
    "",
    "## Moved Files and Folders",
    ""
]
audit_log = []

# === CREATE ARCHIVE FOLDER IF NEEDED ===
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

# === UTILITY FUNCTIONS ===
def is_preserved(path: Path) -> bool:
    rel = path.relative_to(PROJECT_ROOT)
    return any(str(rel).startswith(p) for p in PRESERVE)

def log_action(action, path, extra=None):
    line = f"- {action}: {path}"
    if extra:
        line += f" ({extra})"
    manifest_lines.append(line)
    audit_log.append({
        "action": action,
        "path": str(path),
        "timestamp": datetime.now().isoformat(),
        "details": extra
    })

# === MAIN EXECUTION ===
for item in PROJECT_ROOT.iterdir():
    if is_preserved(item):
        log_action("KEPT", item)
        continue

    dest = ARCHIVE_DIR / item.name
    try:
        size = item.stat().st_size
        mtime = datetime.fromtimestamp(item.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        extra = f"Size: {size} bytes, Modified: {mtime}"

        if DRY_RUN:
            log_action("DRY RUN - Would move", item, extra)
        else:
            shutil.move(str(item), str(dest))
            log_action("MOVED", item, extra)

    except Exception as e:
        log_action("ERROR", item, str(e))

# === WRITE MANIFEST ===
readme_path = ARCHIVE_DIR / "README.md"
if not DRY_RUN:
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("\n".join(manifest_lines))
    print(f"[INFO] Manifest written to: {readme_path}")
else:
    print("[DRY RUN] Manifest would be written to:", readme_path)

# === WRITE JSON AUDIT TRAIL ===
if GENERATE_JSON and not DRY_RUN:
    json_path = ARCHIVE_DIR / "audit_log.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit_log, f, indent=2)
    print(f"[INFO] JSON audit log written to: {json_path}")
elif GENERATE_JSON:
    print("[DRY RUN] JSON audit log would be written to:", ARCHIVE_DIR / "audit_log.json")