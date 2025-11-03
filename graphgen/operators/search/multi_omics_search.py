import re
from typing import Dict, Optional

from graphgen.models import UniProtSearch


def _fetch_uniprot(entry: str) -> Optional[Dict]:
    entry = entry.strip()
    client = UniProtSearch()

    # 1. first try accession search
    if re.fullmatch(
        r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}", entry
    ):
        return client.get_by_accession(entry)

    # 2. then try keyword search
    return client.get_best_hit(entry)


def multi_omics_search(entry: str) -> Dict:
    """
    Multi-omics search function that tries to fetch protein/gene information.
    """
    # TODO: Extend this function to include more omics databases as needed.
    result = _fetch_uniprot(entry)
    if result:
        return {"input": entry, "uniprot": result}
    return {"input": entry, "uniprot": None}
