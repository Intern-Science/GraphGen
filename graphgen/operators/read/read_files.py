from graphgen.models import CsvReader, JsonlReader, JsonReader, TxtReader

_MAPPING = {
    "jsonl": JsonlReader,
    "json": JsonReader,
    "txt": TxtReader,
    "csv": CsvReader,
}


def read_files(file_path: str):
    suffix = file_path.split(".")[-1]
    if suffix in _MAPPING:
        reader = _MAPPING[suffix]()
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. Supported formats are: {list(_MAPPING.keys())}"
        )
    return reader.read(file_path)
