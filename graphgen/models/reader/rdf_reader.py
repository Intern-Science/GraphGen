from typing import Any, Dict, List

import rdflib
from rdflib import Literal
from rdflib.util import guess_format

from graphgen.bases.base_reader import BaseReader


class RDFReader(BaseReader):
    """
    Reader for RDF files that extracts triples and represents them as dictionaries.
    """

    def read(self, file_path: str) -> List[Dict[str, Any]]:
        g = rdflib.Graph()
        fmt = guess_format(file_path)
        try:
            g.parse(file_path, format=fmt)
        except Exception as e:
            raise ValueError(f"Cannot parse RDF file {file_path}: {e}") from e

        docs: List[Dict[str, Any]] = []
        text_col = self.text_column

        for subj in set(g.subjects()):
            literals = []
            props = {}
            for _, pred, obj in g.triples((subj, None, None)):
                pred_str = str(pred)
                if isinstance(obj, Literal):
                    literals.append(str(obj))
                props.setdefault(pred_str, []).append(str(obj))

            text = " ".join(literals).strip()
            if not text:
                raise ValueError(
                    f"Subject {subj} has no literal values; "
                    f"missing '{text_col}' for text column."
                )

            doc = {"id": str(subj), text_col: text, "properties": props}
            docs.append(doc)

        if not docs:
            raise ValueError("RDF file contains no valid documents.")

        return self.filter(docs)
