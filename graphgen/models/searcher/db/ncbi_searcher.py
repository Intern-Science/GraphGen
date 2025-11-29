import asyncio
import logging
import os
import re
import subprocess
import tempfile
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

    def __init__(
        self,
        email: str = "test@example.com",
        tool: str = "GraphGen",
        use_local_blast: bool = False,
        local_blast_db: str = "nt_db",
    ):
        super().__init__()
        Entrez.email = email
        Entrez.tool = tool
        Entrez.timeout = 60  # 60 seconds timeout
        self.use_local_blast = use_local_blast
        self.local_blast_db = local_blast_db
        if self.use_local_blast and not os.path.isfile(f"{self.local_blast_db}.nhr"):
            logger.error("Local BLAST database files not found. Please check the path.")
            self.use_local_blast = False

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
    def _extract_gene_ref(entrezgene_gene):
        """Extract gene_ref from entrezgene_gene."""
        if isinstance(entrezgene_gene, dict):
            return entrezgene_gene.get("Gene-ref", {})
        if hasattr(entrezgene_gene, "get"):
            return entrezgene_gene.get("Gene-ref", {})
        try:
            if hasattr(entrezgene_gene, "Gene-ref"):
                return getattr(entrezgene_gene, "Gene-ref", {})
        except Exception:
            pass
        return {}

    @staticmethod
    def _extract_organism(entrezgene_source):
        """Extract organism from entrezgene_source."""
        try:
            biosource = NCBISearch._safe_get(entrezgene_source, "BioSource", {})
            biosource_org = NCBISearch._safe_get(biosource, "BioSource_org", {})
            org_ref = NCBISearch._safe_get(biosource_org, "Org-ref", {})
            return NCBISearch._safe_get(org_ref, "Org-ref_taxname", "N/A")
        except Exception as e:
            logger.debug("Error extracting organism: %s", e)
            return "N/A"

    @staticmethod
    def _extract_synonyms(gene_ref):
        """Extract gene synonyms from gene_ref."""
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
                        gene_synonyms.append(str(syn))
            elif isinstance(gene_syn, str):
                gene_synonyms.append(gene_syn)
            elif gene_syn:
                gene_synonyms.append(str(gene_syn))
        except Exception as e:
            logger.debug("Error extracting gene synonyms: %s", e)
        return gene_synonyms

    @staticmethod
    def _extract_gene_type(gene_data):
        """Extract gene type from gene_data."""
        try:
            gene_type_data = gene_data.get("Entrezgene_type")
            if not gene_type_data:
                return None
            type_value = str(gene_type_data)
            type_mapping = {
                "1": "protein-coding",
                "2": "pseudo",
                "3": "rRNA",
                "4": "tRNA",
                "5": "snRNA",
                "6": "ncRNA",
                "7": "other",
            }
            return type_mapping.get(type_value, f"type_{type_value}")
        except Exception as e:
            logger.debug("Error extracting gene type: %s", e)
            return None

    @staticmethod
    def _extract_chromosome(first_locus):
        """Extract chromosome from first_locus."""
        label = NCBISearch._safe_get(first_locus, "Gene-commentary_label", "")
        if not label or "Chromosome" not in str(label):
            return None
        match = re.search(r'Chromosome\s+(\S+)', str(label))
        return match.group(1) if match else None

    @staticmethod
    def _extract_genomic_location(first_locus):
        """Extract genomic location from first_locus."""
        seqs = NCBISearch._safe_get(first_locus, "Gene-commentary_seqs", [])
        if not seqs or not isinstance(seqs, list) or not seqs:
            return None
        first_seq = seqs[0]
        if not isinstance(first_seq, dict):
            return None
        seq_loc_int = NCBISearch._safe_get(first_seq, "Seq-loc_int", {})
        if not seq_loc_int:
            return None
        seq_interval = NCBISearch._safe_get(seq_loc_int, "Seq-interval", {})
        if not seq_interval:
            return None
        seq_from = NCBISearch._safe_get(seq_interval, "Seq-interval_from", "")
        seq_to = NCBISearch._safe_get(seq_interval, "Seq-interval_to", "")
        if seq_from and seq_to:
            return f"{seq_from}-{seq_to}"
        return None

    @staticmethod
    def _extract_location_info(locus_data):
        """Extract chromosome and genomic location from locus data."""
        if not locus_data or not isinstance(locus_data, list) or not locus_data:
            return None, None
        first_locus = locus_data[0]
        if not isinstance(first_locus, dict):
            return None, None
        chromosome = NCBISearch._extract_chromosome(first_locus)
        genomic_location = NCBISearch._extract_genomic_location(first_locus)
        return chromosome, genomic_location

    @staticmethod
    def _extract_function_info(gene_data):
        """Extract gene functional description."""
        try:
            summary = gene_data.get("Entrezgene_summary")
            if summary:
                return str(summary)
            comments_data = gene_data.get("Entrezgene_comments")
            if not comments_data or not isinstance(comments_data, list):
                return None
            for comment in comments_data:
                if not isinstance(comment, dict):
                    continue
                heading = NCBISearch._safe_get(comment, "Gene-commentary_heading", "")
                heading_lower = str(heading).lower()
                if "function" not in heading_lower and "summary" not in heading_lower:
                    continue
                comment_text = NCBISearch._safe_get(comment, "Gene-commentary_comment", "")
                if comment_text:
                    return str(comment_text)
            return None
        except Exception as e:
            logger.debug("Error extracting function: %s", e)
            return None

    @staticmethod
    def _extract_accession(locus_data):
        """Extract representative mRNA accession from locus data."""
        if not locus_data or not isinstance(locus_data, list) or not locus_data:
            return None
        first_locus = locus_data[0]
        if not isinstance(first_locus, dict):
            return None
        products = NCBISearch._safe_get(first_locus, "Gene-commentary_products", [])
        if not products or not isinstance(products, list):
            return None
        representative_accession = None
        for product in products:
            if not isinstance(product, dict):
                continue
            product_type = NCBISearch._safe_get(product, "Gene-commentary_type", "")
            product_type_str = str(product_type)
            if product_type_str == "3" or (not representative_accession and product_type_str):
                accession = NCBISearch._safe_get(product, "Gene-commentary_accession", "")
                if accession:
                    representative_accession = str(accession)
                    if product_type_str == "3":
                        break
        return representative_accession

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
        locus_data = gene_data.get("Entrezgene_locus")

        # Extract information using helper methods
        entrezgene_gene = gene_data.get("Entrezgene_gene")
        gene_ref = NCBISearch._extract_gene_ref(entrezgene_gene)
        organism = NCBISearch._extract_organism(gene_data.get("Entrezgene_source"))
        gene_synonyms = NCBISearch._extract_synonyms(gene_ref)
        gene_type = NCBISearch._extract_gene_type(gene_data)
        chromosome, genomic_location = NCBISearch._extract_location_info(locus_data)
        function = NCBISearch._extract_function_info(gene_data)
        representative_accession = NCBISearch._extract_accession(locus_data)

        # Build result dictionary with all fields
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

    def _fetch_sequence(self, accession: str):
        """Fetch sequence from nuccore database using efetch."""
        time.sleep(0.35)  # Comply with rate limit
        seq_handle = Entrez.efetch(
            db="nuccore",
            id=accession,
            rettype="fasta",
            retmode="text",
        )
        try:
            sequence_data = seq_handle.read()
            if not sequence_data:
                return None, None
            seq_lines = sequence_data.strip().split("\n")
            header = seq_lines[0] if seq_lines else ""
            sequence = "".join(seq_lines[1:])
            return sequence, header
        finally:
            seq_handle.close()

    def _fetch_summary(self, accession: str, default_header: str = ""):
        """Fetch summary from nuccore database using esummary."""
        time.sleep(0.35)  # Comply with rate limit
        summary_handle = Entrez.esummary(db="nuccore", id=accession)
        try:
            summary = Entrez.read(summary_handle)
            if not summary:
                return None
            summary_data = summary[0]

            # Determine molecule type detail
            molecule_type_detail = "N/A"
            if accession.startswith("NM_") or accession.startswith("XM_"):
                molecule_type_detail = "mRNA"
            elif accession.startswith("NC_") or accession.startswith("NT_"):
                molecule_type_detail = "genomic DNA"
            elif accession.startswith("NR_") or accession.startswith("XR_"):
                molecule_type_detail = "RNA"
            elif accession.startswith("NG_"):
                molecule_type_detail = "genomic region"

            title = summary_data.get("Title", default_header)
            chromosome = summary_data.get("ChrLoc") or summary_data.get("ChrAccVer")
            chr_start = summary_data.get("ChrStart")
            chr_stop = summary_data.get("ChrStop")
            genomic_location = None
            if chr_start and chr_stop:
                genomic_location = f"{chr_start}-{chr_stop}"

            return {
                "title": title,
                "molecule_type_detail": molecule_type_detail,
                "chromosome": chromosome,
                "genomic_location": genomic_location,
            }
        finally:
            summary_handle.close()

    def _extract_gene_id(self, link_handle):
        """Extract GeneID from elink results."""
        try:
            links = Entrez.read(link_handle)
            if not links or len(links) == 0:
                return None

            first_link = links[0]
            if "LinkSetDb" not in first_link:
                return None

            for link_set in first_link["LinkSetDb"]:
                if link_set.get("DbTo") != "gene":
                    continue

                # Try Link structure first (most common)
                links_in_set = link_set.get("Link", [])
                if links_in_set and len(links_in_set) > 0:
                    first_link_item = links_in_set[0]
                    if isinstance(first_link_item, dict):
                        gene_id = str(first_link_item.get("Id", ""))
                    elif hasattr(first_link_item, "Id"):
                        gene_id = str(getattr(first_link_item, "Id", ""))
                    else:
                        gene_id = str(first_link_item)
                    if gene_id:
                        return gene_id

                # Fallback: Try IdList (if Link is not available)
                id_list = link_set.get("IdList", [])
                if id_list:
                    return str(id_list[0])

            return None
        except Exception as e:
            logger.error("Error parsing elink result: %s", e)
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def _extract_sequence(self, result: dict, accession: str):
        """Enrich result dictionary with sequence and summary information from accession."""
        try:
            sequence, header = self._fetch_sequence(accession)
            if sequence:
                result["sequence"] = sequence
                result["sequence_length"] = len(sequence)

            summary_info = self._fetch_summary(accession, header or "")
            if not summary_info:
                return

            result["title"] = summary_info.get("title")
            result["molecule_type_detail"] = summary_info.get("molecule_type_detail")
            # Update chromosome and genomic_location if not already set
            if not result.get("chromosome") and summary_info.get("chromosome"):
                result["chromosome"] = summary_info["chromosome"]
            if not result.get("genomic_location") and summary_info.get("genomic_location"):
                result["genomic_location"] = summary_info["genomic_location"]
        except (RequestException, IncompleteRead):
            raise
        except Exception as e:
            logger.debug("Failed to get sequence for accession %s: %s", accession, e)

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
        :param preferred_accession: Optional accession to use for sequence retrieval.
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
                accession_to_use = preferred_accession or result.get("_representative_accession")
                if accession_to_use:
                    self._extract_sequence(result, accession_to_use)

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
            try:
                gene_id = self._extract_gene_id(link_handle)
            finally:
                link_handle.close()

            # Step 2: If we have a GeneID, get complete information from Gene database
            if gene_id:
                result = self.get_by_gene_id(gene_id, preferred_accession=accession)
                if result:
                    result["id"] = accession
                    result["url"] = f"https://www.ncbi.nlm.nih.gov/nuccore/{accession}"
                return result

            # Step 3: If no GeneID, this is a rare case (accession without associated gene)
            logger.warning(
                "Accession %s has no associated GeneID, cannot provide complete information",
                accession
            )
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

    def _local_blast(self, seq: str, threshold: float) -> Optional[str]:
        """
        Perform local BLAST search using local BLAST database.
        :param seq: The DNA sequence.
        :param threshold: E-value threshold for BLAST search.
        :return: The accession number of the best hit or None if not found.
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
            logger.debug("Running local blastn: %s", " ".join(cmd))
            out = subprocess.check_output(cmd, text=True).strip()
            os.remove(tmp_name)
            if out:
                return out.split("\n", maxsplit=1)[0]
            return None
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Local blastn failed: %s", exc)
            return None

    def search_by_sequence(self, sequence: str, threshold: float = 0.01) -> Optional[dict]:
        """
        Search NCBI with a DNA sequence using BLAST.
        Tries local BLAST first if enabled, falls back to network BLAST.
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

            # Validate sequence
            if not seq or not re.fullmatch(r"[ATCGN\s]+", seq, re.I):
                if not seq:
                    logger.error("Empty DNA sequence provided.")
                else:
                    logger.error("Invalid DNA sequence provided.")
                return None

            # Try local BLAST first if enabled
            accession = None
            if self.use_local_blast:
                accession = self._local_blast(seq, threshold)
                if accession:
                    logger.debug("Local BLAST found accession: %s", accession)
                    return self.get_by_accession(accession)

            # Fall back to network BLAST
            logger.debug("Falling back to NCBIWWW.qblast.")
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
