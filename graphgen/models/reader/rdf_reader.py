from pathlib import Path
from typing import Any, Dict, List, Union

import ray
import rdflib
from ray.data import Dataset
from rdflib import Literal
from rdflib.util import guess_format

from graphgen.bases.base_reader import BaseReader
from graphgen.utils import logger


class RDFReader(BaseReader):
    """
    Reader for RDF files that extracts triples and represents them as dictionaries.

    Uses Ray Data for distributed processing of multiple RDF files.
    """

    def __init__(self, *, text_column: str = "content", **kwargs):
        """
        Initialize RDFReader.

        :param text_column: The column name for text content (default: "content").
        """
        super().__init__(**kwargs)
        self.text_column = text_column

    def read(
        self,
        input_path: Union[str, List[str]],
    ) -> Dataset:
        """
        Read RDF file(s) using Ray Data.

        :param input_path: Path to RDF file or list of RDF files.
        :return: Ray Dataset containing extracted documents.
        """
        if not ray.is_initialized():
            ray.init()

        # Ensure input_path is a list to prevent Ray from splitting string into characters
        if isinstance(input_path, str):
            input_path = [input_path]

        # Create dataset from file paths
        paths_ds = ray.data.from_items(input_path)

        def process_rdf(row: Dict[str, Any]) -> List[Dict[str, Any]]:
            """Process a single RDF file and return list of documents."""
            try:
                file_path = row["item"]
                return self._parse_rdf_file(Path(file_path))
            except Exception as e:
                logger.error(
                    "Failed to process RDF file %s: %s", row.get("item", "unknown"), e
                )
                return []

        # Process files in parallel and flatten results
        docs_ds = paths_ds.flat_map(process_rdf)

        # Filter valid documents
        docs_ds = docs_ds.filter(self._should_keep_item)

        return docs_ds

    def _parse_rdf_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Parse a single RDF file and extract documents.

        :param file_path: Path to RDF file.
        :return: List of document dictionaries.
        """
        if not file_path.is_file():
            raise FileNotFoundError(f"RDF file not found: {file_path}")

        g = rdflib.Graph()
        fmt = guess_format(str(file_path))

        try:
            g.parse(str(file_path), format=fmt)
        except Exception as e:
            raise ValueError(f"Cannot parse RDF file {file_path}: {e}") from e

        docs: List[Dict[str, Any]] = []

        # Process each unique subject in the RDF graph
        for subj in set(g.subjects()):
            literals = []
            props = {}

            # Extract all triples for this subject
            for _, pred, obj in g.triples((subj, None, None)):
                pred_str = str(pred)
                obj_str = str(obj)

                # Collect literal values as text content
                if isinstance(obj, Literal):
                    literals.append(obj_str)

                # Store all properties (including non-literals)
                props.setdefault(pred_str, []).append(obj_str)

            # Join all literal values as the text content
            text = " ".join(literals).strip()
            if not text:
                logger.warning(
                    "Subject %s in %s has no literal values; document will have empty '%s' field.",
                    subj,
                    file_path,
                    self.text_column,
                )

            # Create document dictionary
            doc = {
                "id": str(subj),
                self.text_column: text,
                "properties": props,
                "source_file": str(file_path),
            }
            docs.append(doc)

        if not docs:
            logger.warning("RDF file %s contains no valid documents.", file_path)

        return docs
