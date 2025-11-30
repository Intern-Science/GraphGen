import asyncio
import os
import re
import subprocess
import tempfile
from typing import Dict, Optional, List, Any

import aiohttp
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from graphgen.bases import BaseSearcher
from graphgen.utils import logger

class RNACentralSearch(BaseSearcher):
    """
    RNAcentral Search client to search RNA databases.
    1) Get RNA by RNAcentral ID.
    2) Search with keywords or RNA names (fuzzy search).
    3) Search with RNA sequence.

    API Documentation: https://rnacentral.org/api/v1
    """

    def __init__(self, use_local_blast: bool = False, local_blast_db: str = "rna_db"):
        super().__init__()
        self.base_url = "https://rnacentral.org/api/v1"
        self.headers = {"Accept": "application/json"}
        self.use_local_blast = use_local_blast
        self.local_blast_db = local_blast_db
        if self.use_local_blast and not os.path.isfile(f"{self.local_blast_db}.nhr"):
            logger.error("Local BLAST database files not found. Please check the path.")
            self.use_local_blast = False

    async def _fetch_all_xrefs(self, xrefs_url: str, session: aiohttp.ClientSession) -> List[Dict]:
        """
        Fetch all xrefs from the xrefs endpoint, handling pagination.
        :param xrefs_url: URL to the xrefs endpoint.
        :param session: aiohttp ClientSession to use for requests.
        :return: List of all xref entries.
        """
        all_xrefs = []
        current_url = xrefs_url

        while current_url:
            try:
                async with session.get(
                    current_url, headers=self.headers, timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results = data.get("results", [])
                        all_xrefs.extend(results)

                        # Check if there's a next page
                        current_url = data.get("next")
                        if not current_url:
                            break

                        # Small delay to avoid rate limiting
                        await asyncio.sleep(0.2)
                    else:
                        logger.warning("Failed to fetch xrefs from %s: HTTP %d", current_url, resp.status)
                        break
            except Exception as e:
                logger.warning("Error fetching xrefs from %s: %s", current_url, e)
                break

        return all_xrefs

    @staticmethod
    def _extract_info_from_xrefs(xrefs: List[Dict]) -> Dict[str, Any]:
        """Extract information from xrefs data."""
        organisms = set()
        gene_names = set()
        modifications = []
        so_terms = set()
        xrefs_list = []

        for xref in xrefs:
            accession = xref.get("accession", {})
            species = accession.get("species")
            if species:
                organisms.add(species)

            gene = accession.get("gene")
            if gene and gene.strip():
                gene_names.add(gene.strip())

            if mods := xref.get("modifications", []):
                modifications.extend(mods)

            if biotype := accession.get("biotype"):
                so_terms.add(biotype)

            xrefs_list.append({
                "database": xref.get("database"),
                "accession_id": accession.get("id"),
                "external_id": accession.get("external_id"),
                "description": accession.get("description"),
                "species": species,
                "gene": gene,
            })

        def _format_set(s):
            """Format set to single value or comma-separated string."""
            if not s:
                return None
            return list(s)[0] if len(s) == 1 else ", ".join(s)

        return {
            "organism": _format_set(organisms),
            "gene_name": _format_set(gene_names),
            "related_genes": list(gene_names) if gene_names else None,
            "modifications": modifications if modifications else None,
            "so_term": _format_set(so_terms),
            "xrefs": xrefs_list if xrefs_list else None,
        }

    @staticmethod
    def _rna_data_to_dict(rna_id: str, rna_data: dict, xrefs_data: Optional[List[Dict]] = None) -> dict:
        """Convert RNAcentral API response to a dictionary."""
        sequence = rna_data.get("sequence", "")
        extracted_info = RNACentralSearch._extract_info_from_xrefs(xrefs_data) if xrefs_data else {}

        # Helper to get value with fallbacks
        def _get_with_fallbacks(key, *fallback_keys):
            if key in extracted_info and extracted_info[key]:
                return extracted_info[key]
            for fk in fallback_keys:
                if value := rna_data.get(fk):
                    return value
            return None

        # Extract related genes with special handling
        related_genes = extracted_info.get("related_genes")
        if not related_genes:
            related_genes = rna_data.get("related_genes") or rna_data.get("genes", [])
            if not related_genes:
                if gene_name_temp := rna_data.get("gene_name"):
                    related_genes = [gene_name_temp]

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
            "organism": _get_with_fallbacks("organism", "organism", "species"),
            "related_genes": related_genes if related_genes else None,
            "gene_name": _get_with_fallbacks("gene_name", "gene_name", "gene"),
            "so_term": _get_with_fallbacks("so_term", "so_term"),
            "modifications": _get_with_fallbacks("modifications", "modifications"),
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

                        # Check if xrefs is a URL and fetch the actual xrefs data
                        xrefs_data = None
                        xrefs_url = rna_data.get("xrefs")
                        if xrefs_url and isinstance(xrefs_url, str) and xrefs_url.startswith("http"):
                            try:
                                xrefs_data = await self._fetch_all_xrefs(xrefs_url, session)
                                logger.debug("Fetched %d xrefs for RNA ID %s", len(xrefs_data), rna_id)
                            except Exception as e:
                                logger.warning("Failed to fetch xrefs for RNA ID %s: %s", rna_id, e)
                                # Continue without xrefs data

                        return self._rna_data_to_dict(rna_id, rna_data, xrefs_data)
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
        """Search RNAcentral with a keyword and return the best hit."""
        if not keyword.strip():
            return None

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/rna",
                    params={"search": keyword, "format": "json"},
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error("HTTP %d error for keyword %s: %s", resp.status, keyword, error_text[:200])
                        raise Exception(f"HTTP {resp.status}: {error_text}")

                    search_results = await resp.json()
                    if not (results := search_results.get("results", [])):
                        logger.info("No results found for keyword: %s", keyword)
                        return None

                    first_result = results[0]
                    if not (rna_id := first_result.get("rnacentral_id")):
                        return None

                    result = await self.get_by_rna_id(rna_id)
                    if not result:
                        logger.debug("get_by_rna_id() failed for %s, using search result data", rna_id)
                        result = self._rna_data_to_dict(rna_id, first_result)
                    return result
        except aiohttp.ClientError as e:
            logger.error("Network error searching for keyword %s: %s", keyword, e)
            return None
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Keyword %s not found: %s", keyword, e)
            return None

    def _local_blast(self, seq: str, threshold: float) -> Optional[str]:
        """Perform local BLAST search using local BLAST database."""
        try:
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".fa", delete=False) as tmp:
                tmp.write(f">query\n{seq}\n")
                tmp_name = tmp.name

            cmd = [
                "blastn", "-db", self.local_blast_db, "-query", tmp_name,
                "-evalue", str(threshold), "-max_target_seqs", "1", "-outfmt", "6 sacc"
            ]
            logger.debug("Running local blastn for RNA: %s", " ".join(cmd))
            out = subprocess.check_output(cmd, text=True).strip()
            os.remove(tmp_name)
            return out.split("\n", maxsplit=1)[0] if out else None
        except Exception as exc:
            logger.error("Local blastn failed: %s", exc)
            return None

    async def search_by_sequence(self, sequence: str, threshold: float = 0.01) -> Optional[dict]:
        """Search RNAcentral with an RNA sequence using BLAST or API."""

        def _extract_and_normalize_sequence(sequence: str) -> Optional[str]:
            """Extract and normalize RNA sequence from input."""
            if sequence.startswith(">"):
                seq = "".join(sequence.strip().split("\n")[1:])
            else:
                seq = sequence.strip().replace(" ", "").replace("\n", "")
            return seq if seq and re.fullmatch(r"[AUCGN\s]+", seq, re.I) else None

        def _find_best_match(results: List[Dict], seq: str) -> Optional[Dict]:
            """Find best match from search results, preferring exact match."""
            for result_item in results:
                if result_item.get("sequence", "") == seq:
                    return result_item
            return results[0] if results else None

        result = None
        try:
            if not (seq := _extract_and_normalize_sequence(sequence)):
                logger.error("Empty or invalid RNA sequence provided.")
            elif self.use_local_blast and (accession := self._local_blast(seq, threshold)):
                logger.debug("Local BLAST found accession: %s", accession)
                result = await self.get_by_rna_id(accession) or await self.get_best_hit(accession)
            else:
                # Fall back to RNAcentral API
                logger.debug("Falling back to RNAcentral API")
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self.base_url}/rna",
                        params={"sequence": seq, "format": "json"},
                        headers=self.headers,
                        timeout=aiohttp.ClientTimeout(total=60),
                    ) as resp:
                        if resp.status != 200:
                            error_text = await resp.text()
                            logger.error("HTTP %d error for sequence search: %s", resp.status, error_text[:200])
                            raise Exception(f"HTTP {resp.status}: {error_text}")

                        search_results = await resp.json()
                        if results := search_results.get("results", []):
                            target_result = _find_best_match(results, seq)
                            if rna_id := target_result.get("rnacentral_id"):
                                result = await self.get_by_rna_id(rna_id)
                                if not result:
                                    logger.debug("get_by_rna_id() failed for %s, using search result data", rna_id)
                                    result = self._rna_data_to_dict(rna_id, target_result)
                        else:
                            logger.info("No results found for sequence.")
        except aiohttp.ClientError as e:
            logger.error("Network error searching for sequence: %s", e)
        except Exception as e:
            logger.error("Sequence search failed: %s", e)
        return result

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        reraise=True,
    )
    async def search(self, query: str, threshold: float = 0.1, **kwargs) -> Optional[Dict]:
        """Search RNAcentral with either an RNAcentral ID, keyword, or RNA sequence."""
        if not query or not isinstance(query, str):
            logger.error("Empty or non-string input.")
            return None

        query = query.strip()
        logger.debug("RNAcentral search query: %s", query)

        # Auto-detect query type
        if query.startswith(">") or (re.fullmatch(r"[AUCGN\s]+", query, re.I) and "U" in query.upper()):
            result = await self.search_by_sequence(query, threshold)
        elif re.fullmatch(r"URS\d+", query, re.I):
            result = await self.get_by_rna_id(query)
        else:
            result = await self.get_best_hit(query)

        if result:
            result["_search_query"] = query
        return result
