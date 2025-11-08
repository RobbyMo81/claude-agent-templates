import os
import shutil
from pathlib import Path
from datetime import datetime
import traceback

# === CONFIGURATION ===
ARCHIVE_DIR = Path(r"C:\Users\RobMo\OneDrive\Documents\helios\archive\backend")
RESTORE_DIR = Path(r"C:\Users\RobMo\OneDrive\Documents\helios")
DRY_RUN = False  # Set to False to execute moves
GENERATE_MANIFEST = True
ALLOW_OVERWRITE = False  # Set to True to overwrite existing files/folders

# === BACKEND FILES & FOLDERS TO RESTORE ===
BACKEND_ITEMS = [
    "server.py", "requirements.txt", "agent.py", "trainer.py", "memory_store.py",
    "metacognition.py", "decision_engine.py", "cross_model_analytics.py", "data_verification.py",
    "db.py", "logger.py", "full_training_suite.py", "run_optimization_experiments.py",
    "check_schema.py", "optimization_configs", "models", "checkpoints", "helios_memory.db",
    "logs", "venv", "venv_py311", "Dockerfile", ".env.example"
]

# === LOGGING ===
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
manifest = [
    "# Helios Backend Recovery Manifest",
    "",
    f"Restored on: {timestamp}",
    "",
    "## Restored Files and Folders",
    ""
]
skipped = []
errors = []

# === MAIN EXECUTION ===
for item_name in BACKEND_ITEMS:
    source = ARCHIVE_DIR / item_name
    destination = RESTORE_DIR / item_name

    if not source.exists():
        print(f"[SKIPPED] Not found in archive: {item_name}")
        skipped.append(item_name)
        continue

    if destination.exists() and not ALLOW_OVERWRITE:
        print(f"[SKIPPED] Destination exists: {destination}")
        skipped.append(item_name)
        continue

    try:
        # Move file or directory
        if source.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.move(str(source), str(destination))
        else:
            shutil.move(str(source), str(destination))
        print(f"[MOVED] {source} → {destination}")
        manifest.append(f"- {item_name}")
    except Exception as e:
        print(f"[ERROR] Failed to move {item_name}: {e}")
        errors.append(f"{item_name}: {e}\n{traceback.format_exc()}")

# === WRITE MANIFEST ===
if GENERATE_MANIFEST:
    manifest_path = RESTORE_DIR / "backend_recovery_manifest.md"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("\n".join(manifest))
        if skipped:
            f.write("\n\n## Skipped Items\n")
            f.write("\n".join(f"- {item}" for item in skipped))
        if errors:
            f.write("\n\n## Errors\n")
            f.write("\n".join(errors))
    print(f"[INFO] Recovery manifest written to: {manifest_path}")

# === SUMMARY ===
print(f"\nSummary: {len(manifest)-6} moved, {len(skipped)} skipped, {len(errors)} errors.")
