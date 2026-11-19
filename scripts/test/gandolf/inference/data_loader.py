"""
**Supported Input Data Formats**

1. A single JSONL file (input: file path).
2. Multiple JSONL files in a directory, all sharing the same data schema (input: directory path).
3. A single Parquet file (input: file path).
4. Multiple Parquet files in a directory, all sharing the same data schema (input: directory path).
"""

from pathlib import Path
import datasets
import pyarrow.parquet as pq


def is_valid_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return True
    return False


def is_valid_parquet(path):
    try:
        parquet_file = pq.ParquetFile(path)

        _ = parquet_file.schema

        if parquet_file.num_row_groups < 1:
            return False

        metadata = parquet_file.metadata

        return (
            metadata is not None
            and metadata.num_rows > 0
        )

    except Exception:
        return False


class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def _resolve_data_files(self):
        """
        Return:
            dataset_type: 'json' or 'parquet'
            data_files: str or list[str]
        """

        path = Path(self.data_path)

        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        # =========================
        # Case: single file
        # =========================
        if path.is_file():

            if path.suffix == ".jsonl":

                if not is_valid_jsonl(path):
                    raise ValueError(f"Empty jsonl file: {path}")

                return "json", str(path)

            elif path.suffix == ".parquet":

                if not is_valid_parquet(path):
                    raise ValueError(f"Empty parquet file: {path}")

                return "parquet", str(path)

            else:
                raise ValueError(
                    f"Unsupported file type: {path.suffix}"
                )

        # =========================
        # Case: directory
        # =========================
        elif path.is_dir():

            jsonl_files = [
                f for f in sorted(path.glob("*.jsonl"))
                if is_valid_jsonl(f)
            ]
            parquet_files = [
                f for f in sorted(path.glob("*.parquet"))
                if is_valid_parquet(f)
            ]

            if jsonl_files and parquet_files:
                raise ValueError(
                    "Directory contains both jsonl and parquet files."
                )

            if jsonl_files:
                return "json", [str(f) for f in jsonl_files]

            if parquet_files:
                return "parquet", [str(f) for f in parquet_files]

            raise ValueError(
                "No jsonl or parquet files found in directory."
            )

        else:
            raise ValueError(f"Invalid path: {path}")

    def load_dataset(self):

        dataset_type, data_files = self._resolve_data_files()

        dataset = datasets.load_dataset(
            dataset_type,
            data_files=data_files,
            streaming=True
        )["train"]

        return dataset

