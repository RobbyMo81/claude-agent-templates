# core/storage.py
from pathlib import Path
import pandas as pd
from pandas.errors import EmptyDataError, ParserError
from typing import Dict, Any

DATA_PATH = Path("data")
DATASET_PATHS = [DATA_PATH / "powerball_complete_dataset.csv", DATA_PATH / "powerball_history_corrected.csv", DATA_PATH / "powerball_history.csv"]


class _Store:
    """
    Simple data store that reads directly from the CSV file.

    • `latest()`        → read the data from the CSV file.
    • `set_latest(df)`  → overwrite the CSV file with new data.
    """

    def __init__(self) -> None:
        DATA_PATH.mkdir(exist_ok=True)
        self.df = self._load_default_csv()

    def _load_default_csv(self) -> pd.DataFrame:
        """Return the best available dataset, prioritizing complete authentic data."""
        for dataset_path in DATASET_PATHS:
            if dataset_path.exists():
                try:
                    # Read the CSV without restricting columns so we can detect and map
                    # alternate header names (for example: wb1..wb5, pb) to the internal
                    # canonical names (N1..N5, Powerball).
                    df = pd.read_csv(dataset_path, engine="python", encoding="utf-8-sig")
                    if df.empty or len(df.columns) == 0:
                        continue

                    # Normalize column names to lowercase for easier matching
                    lower_map = {col: col.lower() for col in df.columns}
                    df.rename(columns=lower_map, inplace=True)

                    # Map common alternate names to canonical names
                    rename_map: Dict[str, str] = {}
                    for col in list(df.columns):
                        # wb1, wb2 ... -> n1, n2 ...
                        if col.startswith("wb") and col[2:].isdigit():
                            idx = int(col[2:])
                            if 1 <= idx <= 5:
                                rename_map[col] = f"n{idx}"
                        # pb -> powerball
                        if col == "pb":
                            rename_map[col] = "powerball"

                    if rename_map:
                        df.rename(columns=rename_map, inplace=True)

                    required_cols = ["draw_date", "n1", "n2", "n3", "n4", "n5", "powerball"]
                    if all(col in df.columns for col in required_cols):
                        # Optional: Validate draw_date is parseable
                        try:
                            pd.to_datetime(df["draw_date"], errors="raise")
                        except Exception as e:
                            print(f"Date parsing failed for {dataset_path}: {e}")
                            continue

                        # Keep only the required columns in canonical order
                        df = df[required_cols]

                        print(f"Loaded data from {dataset_path}")
                        return df

                except (EmptyDataError, ParserError, ValueError) as e:
                    print(f"Failed to load {dataset_path}: {e}")
                    continue

        # No valid dataset found
        print("No valid dataset found. Initializing with an empty DataFrame.")
        return pd.DataFrame()

    def reload_data(self) -> None:
        """Force a reload of the CSV data from disk into the in-memory DataFrame."""
        print("Reloading data from primary CSV source...")
        self.df = self._load_default_csv()

    # -----------------------------------------------------------
    # WRITE METHODS
    # -----------------------------------------------------------
    def set_latest(self, df: pd.DataFrame) -> None:
        """
        Replace the data in the primary CSV file with `df`.
        This method ONLY writes to disk and does not update the in-memory store.
        Call reload_data() to update the in-memory state.
        """
        # Use the primary dataset path for saving
        target_csv_file = DATASET_PATHS[0]
        df.to_csv(target_csv_file, index=False)

    # -----------------------------------------------------------
    # READ METHODS
    # -----------------------------------------------------------
    def latest(self) -> pd.DataFrame:
        """Return the most recently saved DataFrame from memory."""
        # Return the in-memory DataFrame directly
        return self.df


# ------------------------------------------------------------------
# Singleton accessor
# ------------------------------------------------------------------
_STORE = _Store()


def get_store() -> _Store:
    return _STORE


def load_and_normalize_csv(source) -> pd.DataFrame:
    """
    Load a CSV from a file path or file-like object and normalize column names
    into the canonical schema expected by the system.

    Args:
        source: Path/str or file-like object accepted by pandas.read_csv

    Returns:
        DataFrame with canonical columns ['draw_date','n1','n2','n3','n4','n5','powerball']
        or an empty DataFrame if the file cannot be normalized.
    """
    try:
        df = pd.read_csv(source, engine="python", encoding="utf-8-sig")
    except Exception as e:
        print(f"Failed to read {source}: {e}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # Normalize column names to lowercase for easier matching
    lower_map = {col: col.lower() for col in df.columns}
    df.rename(columns=lower_map, inplace=True)

    # Map common alternate names to canonical names
    rename_map: Dict[str, str] = {}
    for col in list(df.columns):
        # wb1, wb2 ... -> n1, n2 ...
        if col.startswith("wb") and col[2:].isdigit():
            idx = int(col[2:])
            if 1 <= idx <= 5:
                rename_map[col] = f"n{idx}"
        # pb -> powerball
        if col == "pb":
            rename_map[col] = "powerball"

    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    required_cols = ["draw_date", "n1", "n2", "n3", "n4", "n5", "powerball"]
    if not all(col in df.columns for col in required_cols):
        return pd.DataFrame()

    # Validate draw_date parseability
    try:
        df["draw_date"] = pd.to_datetime(df["draw_date"], errors="raise")
    except Exception as e:
        print(f"Date parsing failed for {source}: {e}")
        return pd.DataFrame()

    # Keep only required columns in canonical order
    df = df[required_cols]
    # Normalize draw_date to YYYY-MM-DD string for consistency
    df["draw_date"] = df["draw_date"].dt.strftime('%Y-%m-%d')
    return df


if __name__ == "__main__":
    print("\n=== LottoDataAnalyzer Data Ingestion Diagnostic ===")
    for path in DATASET_PATHS:
        print(f"Checking: {path} ... exists: {path.exists()}")
    store = get_store()
    df = store.latest()
    print(f"\nLoaded DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nIf DataFrame is empty or columns are missing, check your CSV files and headers.")
