import asyncio
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import tempfile
from typing import Dict, Optional, List, Any, Set

import hashlib
import requests
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

    def __init__(self, use_local_blast: bool = False, local_blast_db: str = "rna_db"):
        super().__init__()
        self.base_url = "https://rnacentral.org/api/v1"
        self.headers = {"Accept": "application/json"}
        self.use_local_blast = use_local_blast
        self.local_blast_db = local_blast_db
        if self.use_local_blast and not os.path.isfile(f"{self.local_blast_db}.nhr"):
            logger.error("Local BLAST database files not found. Please check the path.")
            self.use_local_blast = False

    @staticmethod
    def _extract_info_from_xrefs(xrefs: List[Dict[str, Any]]) -> Dict[str, Any]:
        organisms: Set[str] = set()
        gene_names: Set[str] = set()
        modifications: List[Any] = []
        so_terms: Set[str] = set()
        xrefs_list: List[Dict[str, Any]] = []

        def format_unique_values(values: Set[str]) -> Optional[str]:
            if not values:
                return None
            if len(values) == 1:
                return next(iter(values))
            return ", ".join(sorted(values))

        for xref in xrefs:
            accession = xref.get("accession", {})
            species = accession.get("species")
            gene = accession.get("gene")
            stripped_gene = gene.strip() if gene else None
            if species:
                organisms.add(species)
            if stripped_gene:
                gene_names.add(stripped_gene)
            if mods := xref.get("modifications"):
                modifications.extend(mods)
            if biotype := accession.get("biotype"):
                so_terms.add(biotype)

            xrefs_list.append({
                "database": xref.get("database"),
                "accession_id": accession.get("id"),
                "external_id": accession.get("external_id"),
                "description": accession.get("description"),
                "species": species,
                "gene": stripped_gene,
            })

        return {
            "organism": format_unique_values(organisms),
            "gene_name": format_unique_values(gene_names),
            "related_genes": list(gene_names) if gene_names else None,
            "modifications": modifications or None,
            "so_term": format_unique_values(so_terms),
            "xrefs": xrefs_list or None,
        }

    @staticmethod
    def _rna_data_to_dict(
        rna_id: str, 
        rna_data: Dict[str, Any], 
        xrefs_data: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        fallback_rules = {
            "organism": ["organism", "species"],
            "related_genes": ["related_genes", "genes"],
            "gene_name": ["gene_name", "gene"],
            "so_term": ["so_term"],
            "modifications": ["modifications"],
        }

        xrefs_info = RNACentralSearch._extract_info_from_xrefs(xrefs_data) if xrefs_data else {}

        def resolve_field(field_name: str) -> Any:
            if value := xrefs_info.get(field_name):
                return value

            for key in fallback_rules[field_name]:
                if (value := rna_data.get(key)) is not None:
                    return value

            return None

        organism = resolve_field("organism")
        gene_name = resolve_field("gene_name")
        so_term = resolve_field("so_term")
        modifications = resolve_field("modifications")

        related_genes = resolve_field("related_genes")
        if not related_genes and (single_gene := rna_data.get("gene_name")):
            related_genes = [single_gene]

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
            "organism": organism,
            "related_genes": related_genes or None,
            "gene_name": gene_name,
            "so_term": so_term,
            "modifications": modifications,
        }

    @staticmethod
    def _calculate_md5(sequence: str) -> str:
        """
        Calculate MD5 hash for RNA sequence as per RNAcentral spec.
        - Replace U with T
        - Convert to uppercase
        - Encode as ASCII
        """
        # Normalize sequence
        normalized_seq = sequence.replace("U", "T").replace("u", "t").upper()
        if not re.fullmatch(r"[ATCGN]+", normalized_seq):
            raise ValueError(f"Invalid sequence characters after normalization: {normalized_seq[:50]}...")

        return hashlib.md5(normalized_seq.encode("ascii")).hexdigest()

    def get_by_rna_id(self, rna_id: str) -> Optional[dict]:
        """
        Get RNA information by RNAcentral ID.
        :param rna_id: RNAcentral ID (e.g., URS0000000001).
        :return: A dictionary containing RNA information or None if not found.
        """
        try:
            url = f"{self.base_url}/rna/{rna_id}"
            url += "?flat=true"

            resp = requests.get(url, headers=self.headers, timeout=30)
            if resp.status_code == 200:
                rna_data = resp.json()
                xrefs_data = rna_data.get("xrefs", [])
                return self._rna_data_to_dict(rna_id, rna_data, xrefs_data)
            logger.error("Failed to fetch RNA ID %s: HTTP %s", rna_id, resp.status_code)
            return None
        except requests.RequestException as e:
            logger.error("Network error getting RNA ID %s: %s", rna_id, e)
            return None
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Unexpected error getting RNA ID %s: %s", rna_id, e)
            return None

    def get_best_hit(self, keyword: str) -> Optional[dict]:
        """
        Search RNAcentral with a keyword and return the best hit.
        Unified approach: Find RNA ID from search, then call get_by_rna_id() for complete information.
        :param keyword: The search keyword (e.g., miRNA name, RNA name).
        :return: A dictionary containing complete RNA information or None if not found.
        """
        if not keyword.strip():
            return None

        try:
            search_url = f"{self.base_url}/rna"
            params = {"search": keyword, "format": "json"}

            resp = requests.get(
                search_url,
                params=params,
                headers=self.headers,
                timeout=30,
            )
            if resp.status_code == 200:
                search_results = resp.json()
                results = search_results.get("results", [])
                if results:
                    # Step 1: Get RNA ID from search results
                    first_result = results[0]
                    rna_id = first_result.get("rnacentral_id")

                    if rna_id:
                        # Step 2: Unified call to get_by_rna_id() for complete information
                        return self.get_by_rna_id(rna_id)
                # Step 3: If get_by_rna_id() failed, use search result data as fallback
                logger.debug("get_by_rna_id() failed for %s, using search result data", rna_id)
                return self._rna_data_to_dict(rna_id, first_result)
            logger.error("No RNA ID found for keyword %s", keyword)
            return None
        except aiohttp.ClientError as e:
            logger.error("Network error searching for keyword %s: %s", keyword, e)
            return None
        except Exception as e:
            logger.error("Keyword %s not found: %s", keyword, e)
            return None

    def _local_blast(self, seq: str, threshold: float) -> Optional[str]:
        """
        Perform local BLAST search using local BLAST database.
        :param seq: The RNA sequence.
        :param threshold: E-value threshold for BLAST search.
        :return: The accession/ID of the best hit or None if not found.
        """
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+", suffix=".fa", delete=False
            ) as tmp:
                tmp.write(f">query\n{seq}\n")
                tmp_name = tmp.name

            cmd = [
                "blastn",
                "-db",
                self.local_blast_db,
                "-query",
                tmp_name,
                "-evalue",
                str(threshold),
                "-max_target_seqs",
                "1",
                "-outfmt",
                "6 sacc",  # only return accession
            ]
            logger.debug("Running local blastn for RNA: %s", " ".join(cmd))
            out = subprocess.check_output(cmd, text=True).strip()
            os.remove(tmp_name)
            if out:
                return out.split("\n", maxsplit=1)[0]
            return None
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Local blastn failed: %s", exc)
            return None

    def _find_best_match_from_results(self, results: List[Dict], seq: str) -> Optional[Dict]:
        """Find best match from search results, preferring exact match."""
        exact_match = None
        for result_item in results:
            result_seq = result_item.get("sequence", "")
            if result_seq == seq:
                exact_match = result_item
                break
        return exact_match if exact_match else (results[0] if results else None)

    async def _process_api_search_results(
        self, results: List[Dict], seq: str
    ) -> Optional[dict]:
        """Process API search results and return dictionary or None."""
        if not results:
            logger.info("No results found for sequence.")
            return None

        target_result = self._find_best_match_from_results(results, seq)
        if not target_result:
            return None

        rna_id = target_result.get("rnacentral_id")
        if not rna_id:
            return None

        # Try to get complete information
        result = await self.get_by_rna_id(rna_id)
        if not result:
            logger.debug("get_by_rna_id() failed for %s, using search result data", rna_id)
            result = self._rna_data_to_dict(rna_id, target_result)
        return result

    def get_by_fasta(self, sequence: str, threshold: float = 0.01) -> Optional[dict]:
        """
        Search RNAcentral with an RNA sequence.
        Tries local BLAST first if enabled, falls back to RNAcentral API.
        Unified approach: Find RNA ID from sequence search, then call get_by_rna_id() for complete information.
        :param sequence: RNA sequence (FASTA format or raw sequence).
        :param threshold: E-value threshold for BLAST search.
        :return: A dictionary containing complete RNA information or None if not found.
        """
        def _extract_and_normalize_sequence(sequence: str) -> Optional[str]:
            """Extract and normalize RNA sequence from input."""
            if sequence.startswith(">"):
                seq_lines = sequence.strip().split("\n")
                seq = "".join(seq_lines[1:])
            else:
                seq = sequence.strip().replace(" ", "").replace("\n", "")
            return seq if seq and re.fullmatch(r"[AUCGN\s]+", seq, re.I) else None

        try:
            seq = _extract_and_normalize_sequence(sequence)
            if not seq:
                logger.error("Empty or invalid RNA sequence provided.")
                return None

            # Try local BLAST first if enabled
            if self.use_local_blast:
                accession = self._local_blast(seq, threshold)
                if accession:
                    logger.debug("Local BLAST found accession: %s", accession)
                    return self.get_by_rna_id(accession)

            # Fall back to RNAcentral API if local BLAST didn't find result
            logger.debug("Falling back to RNAcentral API.")

            md5_hash = self._calculate_md5(seq)
            search_url = f"{self.base_url}/rna"
            params = {"md5": md5_hash, "format": "json"}

            resp = requests.get(search_url, params=params, headers=self.headers, timeout=60)  # Sequence search may take longer
            if resp.status_code == 200:
                search_results = resp.json()
                results = search_results.get("results", [])
                return self._process_api_search_results(results, seq)
            error_text = resp.text()
            logger.error("HTTP %d error for sequence search: %s", resp.status, error_text[:200])
            raise Exception(f"HTTP {resp.status}: {error_text}")
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
        self, query: str, threshold: float = 0.1, **kwargs
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

        loop = asyncio.get_running_loop()

        # check if RNA sequence (AUCG characters, contains U)
        if query.startswith(">") or (
            re.fullmatch(r"[AUCGN\s]+", query, re.I) and "U" in query.upper()
        ):
            result = await loop.run_in_executor(_get_pool(), self.get_by_fasta, query, threshold)
        # check if RNAcentral ID (typically starts with URS)
        elif re.fullmatch(r"URS\d+", query, re.I):
            result = await loop.run_in_executor(_get_pool(), self.get_by_rna_id, query)
        else:
            # otherwise treat as keyword
            result = await loop.run_in_executor(_get_pool(), self.get_best_hit, query)

        if result:
            result["_search_query"] = query
        return result
