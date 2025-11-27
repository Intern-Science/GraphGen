import asyncio
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from http.client import IncompleteRead
from typing import Dict, Optional

from Bio import Entrez
from Bio.Blast import NCBIWWW, NCBIXML
from requests.exceptions import RequestException
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from graphgen.bases import BaseSearcher
from graphgen.utils import logger


@lru_cache(maxsize=None)
def _get_pool():
    return ThreadPoolExecutor(max_workers=10)


class NCBISearch(BaseSearcher):
    """
    NCBI Search client to search DNA/GenBank/Entrez databases.
    1) Get the gene/DNA by accession number or gene ID.
    2) Search with keywords or gene names (fuzzy search).
    3) Search with FASTA sequence (BLAST search for DNA sequences).

    API Documentation: https://www.ncbi.nlm.nih.gov/home/develop/api/
    Note: NCBI has rate limits (max 3 requests per second), delays are required between requests.
    """

    def __init__(self, email: str = "test@example.com", tool: str = "GraphGen"):
        super().__init__()
        Entrez.email = email
        Entrez.tool = tool
        Entrez.timeout = 60  # 60 seconds timeout

    @staticmethod
    def _safe_get(obj, key, default=None):
        """Safely get value from dict or StringElement-like object."""
        if isinstance(obj, dict):
            return obj.get(key, default)
        if hasattr(obj, "get"):
            return obj.get(key, default)
        if hasattr(obj, key):
            return getattr(obj, key, default)
        return default

    @staticmethod
    def _gene_record_to_dict(gene_record, gene_id: str) -> dict:
        """
        Convert an Entrez gene record to a dictionary.
        :param gene_record: The Entrez gene record (list from Entrez.read).
        :param gene_id: The gene ID.
        :return: A dictionary containing gene information.
        """
        if not gene_record:
            raise ValueError("Empty gene record")

        gene_data = gene_record[0]

        # Safely extract gene_ref, handling both dict and StringElement types
        gene_ref = {}
        entrezgene_gene = gene_data.get("Entrezgene_gene")
        if isinstance(entrezgene_gene, dict):
            gene_ref = entrezgene_gene.get("Gene-ref", {})
        elif hasattr(entrezgene_gene, "get"):
            gene_ref = entrezgene_gene.get("Gene-ref", {})
        else:
            # If it's a StringElement or other type, try to access as dict
            try:
                if hasattr(entrezgene_gene, "Gene-ref"):
                    gene_ref = getattr(entrezgene_gene, "Gene-ref", {})
            except Exception:
                pass

        # Safely extract organism
        organism = "N/A"
        try:
            entrezgene_source = gene_data.get("Entrezgene_source")
            if isinstance(entrezgene_source, dict):
                biosource = entrezgene_source.get("BioSource", {})
                if isinstance(biosource, dict):
                    biosource_org = biosource.get("BioSource_org", {})
                    if isinstance(biosource_org, dict):
                        org_ref = biosource_org.get("Org-ref", {})
                        if isinstance(org_ref, dict):
                            organism = org_ref.get("Org-ref_taxname", "N/A")
                        elif hasattr(org_ref, "Org-ref_taxname"):
                            organism = getattr(org_ref, "Org-ref_taxname", "N/A")
        except Exception as e:
            logger.debug("Error extracting organism: %s", e)

        # Extract gene synonyms - safely handle StringElement types
        gene_synonyms = []
        try:
            gene_syn = gene_ref.get("Gene-ref_syn", []) if isinstance(gene_ref, dict) else []
            if isinstance(gene_syn, list):
                for syn in gene_syn:
                    if isinstance(syn, dict):
                        gene_synonyms.append(syn.get("Gene-ref_syn_E", "N/A"))
                    elif isinstance(syn, str):
                        gene_synonyms.append(syn)
                    else:
                        # Handle StringElement or other types
                        gene_synonyms.append(str(syn))
            elif isinstance(gene_syn, str):
                gene_synonyms.append(gene_syn)
            elif gene_syn:  # Handle StringElement
                gene_synonyms.append(str(gene_syn))
        except Exception as e:
            logger.debug("Error extracting gene synonyms: %s", e)

        # Extract gene type - safely handle StringElement types
        # Note: Entrezgene_type is a StringElement with numeric value (e.g., "6" for ncRNA)
        gene_type = None
        try:
            gene_type_data = gene_data.get("Entrezgene_type")
            if gene_type_data:
                type_value = str(gene_type_data)
                # Map numeric values to type names (NCBI gene type codes)
                type_mapping = {
                    "1": "protein-coding",
                    "2": "pseudo",
                    "3": "rRNA",
                    "4": "tRNA",
                    "5": "snRNA",
                    "6": "ncRNA",
                    "7": "other",
                }
                gene_type = type_mapping.get(type_value, f"type_{type_value}")
        except Exception as e:
            logger.debug("Error extracting gene type: %s", e)

        # Extract chromosome and genomic location from Entrezgene_locus
        # Note: Entrezgene_location doesn't exist, but Entrezgene_locus contains location info
        chromosome = None
        genomic_location = None

        try:
            locus_data = gene_data.get("Entrezgene_locus")
            if locus_data and isinstance(locus_data, list) and locus_data:
                first_locus = locus_data[0]
                if isinstance(first_locus, dict):
                    # Extract chromosome from Gene-commentary_label
                    # Example: "Chromosome 13 Reference RoL_Sarg_1.0" -> "13"
                    label = NCBISearch._safe_get(first_locus, "Gene-commentary_label", "")
                    if label and "Chromosome" in str(label):
                        match = re.search(r'Chromosome\s+(\S+)', str(label))
                        if match:
                            chromosome = match.group(1)

                    # Extract genomic location from Gene-commentary_seqs
                    seqs = NCBISearch._safe_get(first_locus, "Gene-commentary_seqs", [])
                    if seqs and isinstance(seqs, list) and seqs:
                        first_seq = seqs[0]
                        if isinstance(first_seq, dict):
                            seq_loc_int = NCBISearch._safe_get(first_seq, "Seq-loc_int", {})
                            if seq_loc_int:
                                seq_interval = NCBISearch._safe_get(seq_loc_int, "Seq-interval", {})
                                if seq_interval:
                                    seq_from = NCBISearch._safe_get(seq_interval, "Seq-interval_from", "")
                                    seq_to = NCBISearch._safe_get(seq_interval, "Seq-interval_to", "")
                                    if seq_from and seq_to:
                                        genomic_location = f"{seq_from}-{seq_to}"
        except Exception as e:
            logger.debug("Error extracting chromosome/location from gene record: %s", e)

        # Extract gene functional description
        # Note: Entrezgene_summary doesn't exist for most genes
        # Try to extract from Entrezgene_comments if available
        function = None
        try:
            # First try Entrezgene_summary (if exists)
            summary = gene_data.get("Entrezgene_summary")
            if summary:
                function = str(summary)
            else:
                # Try to extract from Entrezgene_comments
                comments_data = gene_data.get("Entrezgene_comments")
                if comments_data and isinstance(comments_data, list):
                    for comment in comments_data:
                        if isinstance(comment, dict):
                            heading = NCBISearch._safe_get(comment, "Gene-commentary_heading", "")
                            # Look for function-related comments
                            if "function" in str(heading).lower() or "summary" in str(heading).lower():
                                comment_text = NCBISearch._safe_get(comment, "Gene-commentary_comment", "")
                                if comment_text:
                                    function = str(comment_text)
                                    break
        except Exception as e:
            logger.debug("Error extracting function: %s", e)

        # Try to extract representative mRNA accession from Entrezgene_locus for sequence retrieval
        representative_accession = None
        try:
            if locus_data and isinstance(locus_data, list) and locus_data:
                first_locus = locus_data[0]
                if isinstance(first_locus, dict):
                    products = NCBISearch._safe_get(first_locus, "Gene-commentary_products", [])
                    if products and isinstance(products, list):
                        # Look for mRNA (type 3) or the first product
                        for product in products:
                            if isinstance(product, dict):
                                product_type = NCBISearch._safe_get(product, "Gene-commentary_type", "")
                                product_type_str = str(product_type)
                                # Type 3 is mRNA, prefer mRNA over other types
                                if product_type_str == "3" or (not representative_accession and product_type_str):
                                    accession = NCBISearch._safe_get(product, "Gene-commentary_accession", "")
                                    if accession:
                                        representative_accession = str(accession)
                                        if product_type_str == "3":  # Found mRNA, use it
                                            break
        except Exception as e:
            logger.debug("Error extracting representative accession: %s", e)

        # Build result dictionary with all fields
        # Include all fields that might be present in accession-based queries
        return {
            "molecule_type": "DNA",
            "database": "NCBI",
            "id": gene_id,
            "gene_name": NCBISearch._safe_get(gene_ref, "Gene-ref_locus", "N/A"),
            "gene_description": NCBISearch._safe_get(gene_ref, "Gene-ref_desc", "N/A"),
            "organism": organism,
            "url": f"https://www.ncbi.nlm.nih.gov/gene/{gene_id}",
            "gene_synonyms": gene_synonyms if gene_synonyms else None,
            "gene_type": gene_type,
            "chromosome": chromosome,
            "genomic_location": genomic_location,
            "function": function,
            # Fields from accession-based queries (set to None initially, may be filled later)
            "title": None,
            "sequence": None,
            "sequence_length": None,
            "gene_id": gene_id,  # For consistency with accession queries
            "molecule_type_detail": None,
            "_representative_accession": representative_accession,
        }

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((RequestException, IncompleteRead)),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )

    def get_by_gene_id(self, gene_id: str, preferred_accession: Optional[str] = None) -> Optional[dict]:
        """
        Get gene information by Gene ID.
        This is the unified data source - all search methods eventually call this.
        :param gene_id: NCBI Gene ID.
        :param preferred_accession: Optional accession to use for sequence retrieval if representative mRNA is not available.
        :return: A dictionary containing gene information or None if not found.
        """
        try:
            time.sleep(0.35)  # Comply with rate limit (max 3 requests per second)
            handle = Entrez.efetch(db="gene", id=gene_id, retmode="xml")
            try:
                gene_record = Entrez.read(handle)
                if not gene_record:
                    return None
                result = self._gene_record_to_dict(gene_record, gene_id)

                # Try to get sequence from accession
                # Priority: 1) preferred_accession (if provided), 2) representative mRNA accession
                accession_to_use = preferred_accession or result.get("_representative_accession")
                if accession_to_use:
                    try:
                        # Get sequence info directly from nuccore database
                        time.sleep(0.35)
                        seq_handle = Entrez.efetch(
                            db="nuccore",
                            id=accession_to_use,
                            rettype="fasta",
                            retmode="text",
                        )
                        try:
                            sequence_data = seq_handle.read()
                            if sequence_data:
                                seq_lines = sequence_data.strip().split("\n")
                                header = seq_lines[0] if seq_lines else ""
                                sequence = "".join(seq_lines[1:])

                                # Get summary for additional info
                                time.sleep(0.35)
                                summary_handle = Entrez.esummary(db="nuccore", id=accession_to_use)
                                try:
                                    summary = Entrez.read(summary_handle)
                                    if summary:
                                        summary_data = summary[0]
                                        title = summary_data.get("Title", header)

                                        # Determine molecule type detail
                                        molecule_type_detail = "N/A"
                                        if accession_to_use.startswith("NM_") or accession_to_use.startswith("XM_"):
                                            molecule_type_detail = "mRNA"
                                        elif accession_to_use.startswith("NC_") or accession_to_use.startswith("NT_"):
                                            molecule_type_detail = "genomic DNA"
                                        elif accession_to_use.startswith("NR_") or accession_to_use.startswith("XR_"):
                                            molecule_type_detail = "RNA"
                                        elif accession_to_use.startswith("NG_"):
                                            molecule_type_detail = "genomic region"

                                        # Merge sequence information into result
                                        result["sequence"] = sequence
                                        result["sequence_length"] = len(sequence)
                                        result["title"] = title
                                        result["molecule_type_detail"] = molecule_type_detail

                                        # Update chromosome and genomic_location if not already set
                                        if not result.get("chromosome"):
                                            chromosome = summary_data.get("ChrLoc") or summary_data.get("ChrAccVer")
                                            if chromosome:
                                                result["chromosome"] = chromosome
                                        if not result.get("genomic_location"):
                                            chr_start = summary_data.get("ChrStart")
                                            chr_stop = summary_data.get("ChrStop")
                                            if chr_start and chr_stop:
                                                result["genomic_location"] = f"{chr_start}-{chr_stop}"
                                finally:
                                    summary_handle.close()
                        finally:
                            seq_handle.close()
                    except (RequestException, IncompleteRead):
                        # Re-raise to allow retry mechanism
                        raise
                    except Exception as e:
                        logger.debug("Failed to get sequence for accession %s: %s",
                                   accession_to_use, e)

                # Remove internal field
                result.pop("_representative_accession", None)
                return result
            finally:
                handle.close()
        except RequestException:
            raise
        except IncompleteRead:
            raise
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Gene ID %s not found: %s", gene_id, exc)
            return None

    def get_by_accession(self, accession: str) -> Optional[dict]:
        """
        Get sequence information by accession number.
        Unified approach: Get GeneID from accession, then call get_by_gene_id() for complete information.
        :param accession: NCBI accession number (e.g., NM_000546).
        :return: A dictionary containing complete gene information or None if not found.
        """
        try:
            # Step 1: Get GeneID from elink (nuccore -> gene)
            # Note: esummary for nuccore doesn't include GeneID, so we use elink instead
            time.sleep(0.35)
            link_handle = Entrez.elink(dbfrom="nuccore", db="gene", id=accession)
            gene_id = None
            try:
                links = Entrez.read(link_handle)

                # Extract GeneID from elink results
                # Structure: links[0]["LinkSetDb"][0]["Link"][0]["Id"]
                if links and len(links) > 0:
                    first_link = links[0]
                    if "LinkSetDb" in first_link:
                        for link_set in first_link["LinkSetDb"]:
                            if link_set.get("DbTo") == "gene":
                                # Try Link structure first (most common)
                                links_in_set = link_set.get("Link", [])
                                if links_in_set and len(links_in_set) > 0:
                                    first_link_item = links_in_set[0]
                                    if isinstance(first_link_item, dict):
                                        gene_id = str(first_link_item.get("Id", ""))
                                    elif hasattr(first_link_item, "Id"):
                                        gene_id = str(getattr(first_link_item, "Id", ""))
                                    else:
                                        # Handle StringElement or other types
                                        gene_id = str(first_link_item)
                                    if gene_id:
                                        break
                                # Fallback: Try IdList (if Link is not available)
                                id_list = link_set.get("IdList", [])
                                if id_list and not gene_id:
                                    gene_id = str(id_list[0])
                                    break
            except Exception as e:
                logger.error("Error parsing elink result for accession %s: %s", accession, e)
                import traceback
                logger.debug(traceback.format_exc())
                # Continue to check if we got gene_id before the error
            finally:
                link_handle.close()

            # Step 2: If we have a GeneID, get complete information from Gene database
            # Pass the accession as preferred_accession so get_by_gene_id can use it for sequence
            if gene_id:
                result = self.get_by_gene_id(gene_id, preferred_accession=accession)

                # Update id to accession for consistency (user searched by accession)
                if result:
                    result["id"] = accession
                    result["url"] = f"https://www.ncbi.nlm.nih.gov/nuccore/{accession}"

                return result

            # Step 3: If no GeneID, this is a rare case (accession without associated gene)
            # Return None - we can't provide complete information without Gene ID
            logger.warning("Accession %s has no associated GeneID, cannot provide complete information", accession)
            return None
        except (RequestException, IncompleteRead):
            raise
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Accession %s not found: %s", accession, exc)
            return None

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((RequestException, IncompleteRead)),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def get_best_hit(self, keyword: str) -> Optional[dict]:
        """
        Search NCBI Gene database with a keyword and return the best hit.
        :param keyword: The search keyword (e.g., gene name).
        :return: A dictionary containing the best hit information or None if not found.
        """
        if not keyword.strip():
            return None

        try:
            time.sleep(0.35)  # Comply with rate limit
            # Search gene database
            search_handle = Entrez.esearch(
                db="gene",
                term=f"{keyword}[Gene Name] OR {keyword}[All Fields]",
                retmax=1,
            )
            try:
                search_results = Entrez.read(search_handle)
                if not search_results.get("IdList"):
                    # If not found, try a broader search
                    time.sleep(0.35)
                    search_handle2 = Entrez.esearch(
                        db="gene",
                        term=keyword,
                        retmax=1,
                    )
                    try:
                        search_results = Entrez.read(search_handle2)
                    finally:
                        search_handle2.close()

                if search_results.get("IdList"):
                    gene_id = search_results["IdList"][0]
                    return self.get_by_gene_id(gene_id)
            finally:
                search_handle.close()
        except RequestException:
            raise
        except IncompleteRead:
            raise
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Keyword %s not found: %s", keyword, e)
        return None

    def search_by_sequence(self, sequence: str, threshold: float = 0.01) -> Optional[dict]:
        """
        Search NCBI with a DNA sequence using BLAST.
        Note: This is a simplified version. For production, consider using local BLAST.
        :param sequence: DNA sequence (FASTA format or raw sequence).
        :param threshold: E-value threshold for BLAST search.
        :return: A dictionary containing the best hit information or None if not found.
        """
        try:
            # Extract sequence (if in FASTA format)
            if sequence.startswith(">"):
                seq_lines = sequence.strip().split("\n")
                seq = "".join(seq_lines[1:])
            else:
                seq = sequence.strip().replace(" ", "").replace("\n", "")

            # Validate if it's a DNA sequence
            if not re.fullmatch(r"[ATCGN\s]+", seq, re.I):
                logger.error("Invalid DNA sequence provided.")
                return None

            if not seq:
                logger.error("Empty DNA sequence provided.")
                return None

            # Use BLAST search (Note: requires network connection, may be slow)
            logger.debug("Performing BLAST search for DNA sequence...")
            time.sleep(0.35)

            result_handle = NCBIWWW.qblast(
                program="blastn",
                database="nr",
                sequence=seq,
                hitlist_size=1,
                expect=threshold,
            )
            blast_record = NCBIXML.read(result_handle)

            if not blast_record.alignments:
                logger.info("No BLAST hits found for the given sequence.")
                return None

            best_alignment = blast_record.alignments[0]
            best_hsp = best_alignment.hsps[0]
            if best_hsp.expect > threshold:
                logger.info("No BLAST hits below the threshold E-value.")
                return None
            hit_id = best_alignment.hit_id

            # Extract accession number
            # Format may be: gi|123456|ref|NM_000546.5|
            accession_match = re.search(r"ref\|([^|]+)", hit_id)
            if accession_match:
                accession = accession_match.group(1).split(".")[0]
                return self.get_by_accession(accession)
            # If unable to extract accession, return basic information
            return {
                "molecule_type": "DNA",
                "database": "NCBI",
                "id": hit_id,
                "title": best_alignment.title,
                "sequence_length": len(seq),
                "e_value": best_hsp.expect,
                "identity": best_hsp.identities / best_hsp.align_length if best_hsp.align_length > 0 else 0,
                "url": f"https://www.ncbi.nlm.nih.gov/nuccore/{hit_id}",
            }
        except RequestException:
            raise
        except Exception as e:  # pylint: disable=broad-except
            logger.error("BLAST search failed: %s", e)
            return None

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((RequestException, IncompleteRead)),
        reraise=True,
    )
    async def search(
        self, query: str, threshold: float = 0.01, **kwargs
    ) -> Optional[Dict]:
        """
        Search NCBI with either a gene ID, accession number, keyword, or DNA sequence.
        :param query: The search query (gene ID, accession, keyword, or DNA sequence).
        :param threshold: E-value threshold for BLAST search.
        :param kwargs: Additional keyword arguments (not used currently).
        :return: A dictionary containing the search results or None if not found.
        """
        # auto detect query type
        if not query or not isinstance(query, str):
            logger.error("Empty or non-string input.")
            return None
        query = query.strip()

        logger.debug("NCBI search query: %s", query)

        loop = asyncio.get_running_loop()

        # check if DNA sequence (ATCG characters)
        if query.startswith(">") or re.fullmatch(r"[ATCGN\s]+", query, re.I):
            result = await loop.run_in_executor(
                _get_pool(), self.search_by_sequence, query, threshold
            )
        # check if gene ID (numeric)
        elif re.fullmatch(r"^\d+$", query):
            result = await loop.run_in_executor(
                _get_pool(), self.get_by_gene_id, query
            )
        # check if accession number (e.g., NM_000546, NC_000001)
        elif re.fullmatch(r"[A-Z]{2}_\d+\.?\d*", query, re.I):
            result = await loop.run_in_executor(
                _get_pool(), self.get_by_accession, query
            )
        else:
            # otherwise treat as keyword
            result = await loop.run_in_executor(
                _get_pool(), self.get_best_hit, query
            )

        if result:
            result["_search_query"] = query
        return result
