import asyncio
import re
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

    def __init__(self):
        super().__init__()
        self.base_url = "https://rnacentral.org/api/v1"
        self.headers = {"Accept": "application/json"}

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
        """
        Extract information from xrefs data.
        :param xrefs: List of xref entries.
        :return: Dictionary with extracted information.
        """
        extracted = {
            "organisms": set(),
            "gene_names": set(),
            "modifications": [],
            "so_terms": set(),
            "xrefs_list": [],
        }
        
        for xref in xrefs:
            # Extract accession information
            accession = xref.get("accession", {})
            
            # Extract species information
            species = accession.get("species")
            if species:
                extracted["organisms"].add(species)
            
            # Extract gene name
            gene = accession.get("gene")
            if gene and gene.strip():  # Only add non-empty genes
                extracted["gene_names"].add(gene.strip())
            
            # Extract modifications
            modifications = xref.get("modifications", [])
            if modifications:
                extracted["modifications"].extend(modifications)
            
            # Extract SO term (biotype)
            biotype = accession.get("biotype")
            if biotype:
                extracted["so_terms"].add(biotype)
            
            # Build xrefs list
            xref_info = {
                "database": xref.get("database"),
                "accession_id": accession.get("id"),
                "external_id": accession.get("external_id"),
                "description": accession.get("description"),
                "species": species,
                "gene": gene,
            }
            extracted["xrefs_list"].append(xref_info)
        
        # Convert sets to appropriate formats
        return {
            "organism": (
                list(extracted["organisms"])[0] 
                if len(extracted["organisms"]) == 1 
                else (", ".join(extracted["organisms"]) if extracted["organisms"] else None)
            ),
            "gene_name": (
                list(extracted["gene_names"])[0] 
                if len(extracted["gene_names"]) == 1 
                else (", ".join(extracted["gene_names"]) if extracted["gene_names"] else None)
            ),
            "related_genes": list(extracted["gene_names"]) if extracted["gene_names"] else None,
            "modifications": extracted["modifications"] if extracted["modifications"] else None,
            "so_term": (
                list(extracted["so_terms"])[0] 
                if len(extracted["so_terms"]) == 1 
                else (", ".join(extracted["so_terms"]) if extracted["so_terms"] else None)
            ),
            "xrefs": extracted["xrefs_list"] if extracted["xrefs_list"] else None,
        }

    @staticmethod
    def _rna_data_to_dict(rna_id: str, rna_data: dict, xrefs_data: Optional[List[Dict]] = None) -> dict:
        """
        Convert RNAcentral API response to a dictionary.
        :param rna_id: RNAcentral ID.
        :param rna_data: API response data (dict or dict-like from search results).
        :param xrefs_data: Optional list of xref entries fetched from xrefs endpoint.
        :return: A dictionary containing RNA information.
        """
        sequence = rna_data.get("sequence", "")
        
        # Initialize extracted info from xrefs if available
        extracted_info = {}
        if xrefs_data:
            extracted_info = RNACentralSearch._extract_info_from_xrefs(xrefs_data)
        
        # Extract organism information (prefer from xrefs, fallback to main data)
        organism = extracted_info.get("organism")
        if not organism:
            organism = rna_data.get("organism", None)
        if not organism:
            organism = rna_data.get("species", None)

        # Extract related genes (prefer from xrefs, fallback to main data)
        related_genes = extracted_info.get("related_genes")
        if not related_genes:
            related_genes = rna_data.get("related_genes", [])
        if not related_genes:
            related_genes = rna_data.get("genes", [])
        if not related_genes:
            gene_name_temp = rna_data.get("gene_name", None)
            if gene_name_temp:
                related_genes = [gene_name_temp]

        # Extract gene name (prefer from xrefs, fallback to main data)
        gene_name = extracted_info.get("gene_name")
        if not gene_name:
            gene_name = rna_data.get("gene_name", None)
        if not gene_name:
            gene_name = rna_data.get("gene", None)

        # Extract so_term (prefer from xrefs, fallback to main data)
        so_term = extracted_info.get("so_term")
        if not so_term:
            so_term = rna_data.get("so_term", None)

        # Extract modifications (prefer from xrefs, fallback to main data)
        modifications = extracted_info.get("modifications")
        if not modifications:
            modifications = rna_data.get("modifications", None)

        # Build result dictionary (xrefs information is already extracted into other fields)
        # information is extracted into organism, gene_name, so_term, modifications, etc.
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
            "related_genes": related_genes if related_genes else None,
            "gene_name": gene_name,
            "so_term": so_term,
            "modifications": modifications,
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
        """
        Search RNAcentral with a keyword and return the best hit.
        Unified approach: Find RNA ID from search, then call get_by_rna_id() for complete information.
        :param keyword: The search keyword (e.g., miRNA name, RNA name).
        :return: A dictionary containing complete RNA information or None if not found.
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
                            # Step 1: Get RNA ID from search results
                            first_result = results[0]
                            rna_id = first_result.get("rnacentral_id")
                            
                            if rna_id:
                                # Step 2: Unified call to get_by_rna_id() for complete information
                                result = await self.get_by_rna_id(rna_id)
                                
                                # Step 3: If get_by_rna_id() failed, use search result data as fallback
                                if not result:
                                    logger.debug("get_by_rna_id() failed for %s, using search result data", rna_id)
                                    result = self._rna_data_to_dict(rna_id, first_result)
                                
                                return result
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
        Unified approach: Find RNA ID from sequence search, then call get_by_rna_id() for complete information.
        :param sequence: RNA sequence (FASTA format or raw sequence).
        :return: A dictionary containing complete RNA information or None if not found.
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
                            # Step 1: Find best match (prefer exact match)
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
                                # Step 2: Unified call to get_by_rna_id() for complete information
                                result = await self.get_by_rna_id(rna_id)
                                
                                # Step 3: If get_by_rna_id() failed, use search result data as fallback
                                if not result:
                                    logger.debug("get_by_rna_id() failed for %s, using search result data", rna_id)
                                    result = self._rna_data_to_dict(rna_id, target_result)
                                
                                return result
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
