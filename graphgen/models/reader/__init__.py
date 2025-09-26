from .csv_reader import CsvReader
from .json_reader import JsonReader
from .jsonl_reader import JsonlReader
from .txt_reader import TxtReader

_MAPPING = {
    "jsonl": JsonlReader,
    "json": JsonReader,
    "txt": TxtReader,
    "csv": CsvReader,
}


def read_file(file_path: str):
    suffix = file_path.split(".")[-1]
    if suffix in _MAPPING:
        reader = _MAPPING[suffix]()
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. Supported formats are: {list(_MAPPING.keys())}"
        )
    return reader.read(file_path)
