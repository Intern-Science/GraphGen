import requests
from fastapi import HTTPException

from graphgen.utils import logger

UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb/search"


class UniProtSearch:
    """
    UniProt Search client to search with UniProt.
    1) Get the protein by accession number.
    2) Search with keywords or protein names.
    """

    def get_entry(self, accession: str) -> dict:
        """
        Get the UniProt entry by accession number(e.g., P04637).
        """
        url = f"{UNIPROT_BASE}/{accession}.json"
        return self._safe_get(url).json()

    def search(
        self,
        query: str,
        *,
        size: int = 10,
        cursor: str = None,
        fields: list[str] = None,
    ) -> dict:
        """
        Search UniProt with a query string.
        :param query: The search query.
        :param size: The number of results to return.
        :param cursor: The cursor for pagination.
        :param fields: The fields to return in the response.
        :return: A dictionary containing the search results.
        """
        params = {
            "query": query,
            "size": size,
        }
        if cursor:
            params["cursor"] = cursor
        if fields:
            params["fields"] = ",".join(fields)
        url = UNIPROT_BASE
        return self._safe_get(url, params=params).json()

    @staticmethod
    def _safe_get(url: str, params: dict = None) -> requests.Response:
        r = requests.get(
            url,
            params=params,
            headers={"Accept": "application/json"},
            timeout=10,
        )
        if not r.ok:
            logger.error("Search engine error: %s", r.text)
            raise HTTPException(r.status_code, "Search engine error.")
        return r
