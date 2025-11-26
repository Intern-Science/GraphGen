import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Dict, Optional

import aiohttp
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from graphgen.bases import BaseSearcher
from graphgen.utils import logger


@lru_cache(maxsize=None)
def _get_pool():
    return ThreadPoolExecutor(max_workers=10)


class RNACentralSearch(BaseSearcher):
    """
    RNAcentral Search client to search RNA databases.
    1) Get RNA by RNAcentral ID.
    2) Search with keywords or RNA names (fuzzy search).
    3) Search with RNA sequence.
    
    API Documentation: https://rnacentral.org/api/v1
    """

    def __init__(self):
        super().__init__()
        self.base_url = "https://rnacentral.org/api/v1"
        self.headers = {"Accept": "application/json"}

    async def get_by_rna_id(self, rna_id: str) -> Optional[dict]:
        """
        Get RNA information by RNAcentral ID.
        :param rna_id: RNAcentral ID (e.g., URS0000000001).
        :return: A dictionary containing RNA information or None if not found.
        """
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/rna/{rna_id}"
                async with session.get(
                    url, headers=self.headers, timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        rna_data = await resp.json()
                        return {
                            "molecule_type": "RNA",
                            "database": "RNAcentral",
                            "id": rna_id,
                            "rnacentral_id": rna_data.get("rnacentral_id", "N/A"),
                            "sequence": rna_data.get("sequence", ""),
                            "sequence_length": len(rna_data.get("sequence", "")),
                            "rna_type": rna_data.get("rna_type", "N/A"),
                            "description": rna_data.get("description", "N/A"),
                            "url": f"https://rnacentral.org/rna/{rna_id}",
                        }
                    if resp.status == 404:
                        logger.error("RNA ID %s not found", rna_id)
                        return None
                    raise Exception(f"HTTP {resp.status}: {await resp.text()}")
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("RNA ID %s not found: %s", rna_id, exc)
            return None

    async def search_by_keyword(self, keyword: str) -> Optional[dict]:
        """
        Search RNAcentral with a keyword and return the best hit.
        :param keyword: The search keyword (e.g., miRNA name, RNA name).
        :return: A dictionary containing the best hit information or None if not found.
        """
        if not keyword.strip():
            return None

        try:
            async with aiohttp.ClientSession() as session:
                search_url = f"{self.base_url}/rna"
                params = {"search": keyword, "format": "json"}
                async with session.get(
                    search_url,
                    params=params,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 200:
                        search_results = await resp.json()
                        if search_results.get("results"):
                            rna_id = search_results["results"][0].get("rnacentral_id")
                            if rna_id:
                                return await self.get_by_rna_id(rna_id)
                        logger.info("No results found for keyword: %s", keyword)
                        return None
                    raise Exception(f"HTTP {resp.status}: {await resp.text()}")
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Keyword %s not found: %s", keyword, e)
            return None

    async def search_by_sequence(self, sequence: str) -> Optional[dict]:
        """
        Search RNAcentral with an RNA sequence.
        :param sequence: RNA sequence (FASTA format or raw sequence).
        :return: A dictionary containing the best hit information or None if not found.
        """
        try:
            # Extract sequence (if in FASTA format)
            if sequence.startswith(">"):
                seq_lines = sequence.strip().split("\n")
                seq = "".join(seq_lines[1:])
            else:
                seq = sequence.strip().replace(" ", "").replace("\n", "")

            # Validate if it's an RNA sequence (contains U instead of T)
            if not re.fullmatch(r"[AUCGN\s]+", seq, re.I):
                logger.error("Invalid RNA sequence provided.")
                return None

            if not seq:
                logger.error("Empty RNA sequence provided.")
                return None

            # RNAcentral API supports sequence search
            async with aiohttp.ClientSession() as session:
                search_url = f"{self.base_url}/rna"
                params = {"sequence": seq, "format": "json"}
                async with session.get(
                    search_url,
                    params=params,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=60),  # Sequence search may take longer
                ) as resp:
                    if resp.status == 200:
                        search_results = await resp.json()
                        if search_results.get("results"):
                            rna_id = search_results["results"][0].get("rnacentral_id")
                            if rna_id:
                                return await self.get_by_rna_id(rna_id)
                        logger.info("No results found for sequence.")
                        return None
                    raise Exception(f"HTTP {resp.status}: {await resp.text()}")
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Sequence search failed: %s", e)
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        reraise=True,
    )
    async def search(
        self, query: str, **kwargs
    ) -> Optional[Dict]:
        """
        Search RNAcentral with either an RNAcentral ID, keyword, or RNA sequence.
        :param query: The search query (RNAcentral ID, keyword, or RNA sequence).
        :param kwargs: Additional keyword arguments (not used currently).
        :return: A dictionary containing the search results or None if not found.
        """
        # auto detect query type
        if not query or not isinstance(query, str):
            logger.error("Empty or non-string input.")
            return None
        query = query.strip()

        logger.debug("RNAcentral search query: %s", query)

        # check if RNA sequence (AUCG characters, contains U)
        if query.startswith(">") or (
            re.fullmatch(r"[AUCGN\s]+", query, re.I) and "U" in query.upper()
        ):
            result = await self.search_by_sequence(query)
        # check if RNAcentral ID (typically starts with URS)
        elif re.fullmatch(r"URS\d+", query, re.I):
            result = await self.get_by_rna_id(query)
        else:
            # otherwise treat as keyword
            result = await self.search_by_keyword(query)

        if result:
            result["_search_query"] = query
        return result
