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

    @staticmethod
    def _rna_data_to_dict(rna_id: str, rna_data: dict) -> dict:
        """
        Convert RNAcentral API response to a dictionary.
        :param rna_id: RNAcentral ID.
        :param rna_data: API response data (dict or dict-like from search results).
        :return: A dictionary containing RNA information.
        """
        sequence = rna_data.get("sequence", "")
        return {
            "molecule_type": "RNA",
            "database": "RNAcentral",
            "id": rna_id,
            "rnacentral_id": rna_data.get("rnacentral_id", rna_id),
            "sequence": sequence,
            "sequence_length": rna_data.get("length", len(sequence)),
            "rna_type": rna_data.get("rna_type", "N/A"),
            "description": rna_data.get("description", "N/A"),
            "url": f"https://rnacentral.org/rna/{rna_id}",
        }

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
                        return self._rna_data_to_dict(rna_id, rna_data)
                    if resp.status == 404:
                        logger.error("RNA ID %s not found", rna_id)
                        return None
                    raise Exception(f"HTTP {resp.status}: {await resp.text()}")
        except aiohttp.ClientError as e:
            logger.error("Network error getting RNA ID %s: %s", rna_id, e)
            return None
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("RNA ID %s not found: %s", rna_id, exc)
            return None

    async def get_best_hit(self, keyword: str) -> Optional[dict]:
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
                        results = search_results.get("results", [])
                        if results:
                            # Use the first result directly (search API already returns enough info)
                            first_result = results[0]
                            rna_id = first_result.get("rnacentral_id")
                            if rna_id:
                                # Try to get detailed info, but fall back to search result if it fails
                                detailed_info = await self.get_by_rna_id(rna_id)
                                if detailed_info:
                                    return detailed_info
                                # Fall back to using search result data
                                return self._rna_data_to_dict(rna_id, first_result)
                        logger.info("No results found for keyword: %s", keyword)
                        return None
                    error_text = await resp.text()
                    logger.error("HTTP %d error for keyword %s: %s", resp.status, keyword, error_text[:200])
                    raise Exception(f"HTTP {resp.status}: {error_text}")
        except aiohttp.ClientError as e:
            logger.error("Network error searching for keyword %s: %s", keyword, e)
            return None
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
                        results = search_results.get("results", [])
                        if results:
                            # First, try to find an exact sequence match
                            exact_match = None
                            for result in results:
                                result_seq = result.get("sequence", "")
                                if result_seq == seq:
                                    exact_match = result
                                    break
                            
                            # Use exact match if found, otherwise use first result
                            target_result = exact_match if exact_match else results[0]
                            rna_id = target_result.get("rnacentral_id")
                            
                            if rna_id:
                                # Try to get detailed info, but fall back to search result if it fails
                                try:
                                    detailed_info = await self.get_by_rna_id(rna_id)
                                    if detailed_info:
                                        return detailed_info
                                except Exception as e:
                                    logger.debug("Failed to get detailed info for %s: %s, using search result", rna_id, e)
                                
                                # Fall back to using search result data
                                return self._rna_data_to_dict(rna_id, target_result)
                        logger.info("No results found for sequence.")
                        return None
                    error_text = await resp.text()
                    logger.error("HTTP %d error for sequence search: %s", resp.status, error_text[:200])
                    raise Exception(f"HTTP {resp.status}: {error_text}")
        except aiohttp.ClientError as e:
            logger.error("Network error searching for sequence: %s", e)
            return None
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
        self, query: str, threshold: float = 0.7, **kwargs
    ) -> Optional[Dict]:
        """
        Search RNAcentral with either an RNAcentral ID, keyword, or RNA sequence.
        :param query: The search query (RNAcentral ID, keyword, or RNA sequence).
        :param threshold: E-value threshold for sequence search.
        Note: RNAcentral API uses its own similarity matching, this parameter is for interface consistency.
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
            result = await self.get_best_hit(query)

        if result:
            result["_search_query"] = query
        return result
