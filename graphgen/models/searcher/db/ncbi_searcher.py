import asyncio
import re
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from http.client import IncompleteRead
from typing import Dict, Optional

from Bio import Entrez
from requests.exceptions import RequestException
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
        gene_ref = gene_data.get("Entrezgene_gene", {}).get("Gene-ref", {})

        organism = (
            gene_data.get("Entrezgene_source", {})
            .get("BioSource", {})
            .get("BioSource_org", {})
            .get("Org-ref", {})
            .get("Org-ref_taxname", "N/A")
        )

        return {
            "molecule_type": "DNA",
            "database": "NCBI",
            "id": gene_id,
            "gene_name": gene_ref.get("Gene-ref_locus", "N/A"),
            "gene_description": gene_ref.get("Gene-ref_desc", "N/A"),
            "organism": organism,
            "url": f"https://www.ncbi.nlm.nih.gov/gene/{gene_id}",
        }

    def get_by_gene_id(self, gene_id: str) -> Optional[dict]:
        """
        Get gene information by Gene ID.
        :param gene_id: NCBI Gene ID.
        :return: A dictionary containing gene information or None if not found.
        """
        try:
            time.sleep(0.35)  # Comply with rate limit (max 3 requests per second)
            handle = Entrez.efetch(db="gene", id=gene_id, retmode="xml")
            try:
                gene_record = Entrez.read(handle)
                if not gene_record:
                    return None
                return self._gene_record_to_dict(gene_record, gene_id)
            finally:
                handle.close()
        except RequestException:
            raise
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Gene ID %s not found: %s", gene_id, exc)
            return None

    @staticmethod
    def _accession_to_dict(accession: str, sequence: str, header: str, title: str, organism: str) -> dict:
        """
        Convert accession information to a dictionary.
        :param accession: NCBI accession number.
        :param sequence: DNA sequence.
        :param header: FASTA header.
        :param title: Sequence title.
        :param organism: Organism name.
        :return: A dictionary containing sequence information.
        """
        return {
            "molecule_type": "DNA",
            "database": "NCBI",
            "id": accession,
            "title": title,
            "organism": organism,
            "sequence": sequence,
            "sequence_length": len(sequence),
            "url": f"https://www.ncbi.nlm.nih.gov/nuccore/{accession}",
        }

    def get_by_accession(self, accession: str) -> Optional[dict]:
        """
        Get sequence information by accession number.
        :param accession: NCBI accession number (e.g., NM_000546).
        :return: A dictionary containing sequence information or None if not found.
        """
        try:
            time.sleep(0.35)  # Comply with rate limit
            handle = Entrez.efetch(
                db="nuccore",
                id=accession,
                rettype="fasta",
                retmode="text",
            )
            try:
                sequence_data = handle.read()
                if not sequence_data:
                    return None

                seq_lines = sequence_data.strip().split("\n")
                header = seq_lines[0] if seq_lines else ""
                sequence = "".join(seq_lines[1:])

                # Try to get more information
                time.sleep(0.35)
                summary_handle = Entrez.esummary(db="nuccore", id=accession)
                try:
                    summary = Entrez.read(summary_handle)
                    if summary:
                        summary_data = summary[0]
                        title = summary_data.get("Title", header)
                        organism = summary_data.get("Organism", "N/A")
                    else:
                        title = header
                        organism = "N/A"
                finally:
                    summary_handle.close()

                return self._accession_to_dict(accession, sequence, header, title, organism)
            finally:
                handle.close()
        except RequestException:
            raise
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Accession %s not found: %s", accession, exc)
            return None

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
            from Bio.Blast import NCBIWWW, NCBIXML

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
