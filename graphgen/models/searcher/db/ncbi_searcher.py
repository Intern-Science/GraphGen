import asyncio
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from http.client import IncompleteRead
from typing import Dict, Optional
from graphgen.models.searcher.limitter import RateLimiter

from Bio import Entrez, SeqIO
from Bio.Blast import NCBIWWW, NCBIXML
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

    def __init__(
        self,
        use_local_blast: bool = False,
        local_blast_db: str = "nt_db",
        email: str = "email@example.com",
        api_key: str = "",
    ):
        """
        Initialize the NCBI Search client.

        Args:
            use_local_blast (bool): Whether to use local BLAST database.
            local_blast_db (str): Path to the local BLAST database.
            email (str): Email address for NCBI API requests.
            api_key (str): API key for NCBI API requests, see https://account.ncbi.nlm.nih.gov/settings/.
        """
        super().__init__()
        Entrez.timeout = 60  # 60 seconds timeout
        Entrez.email = email
        if api_key:
            Entrez.api_key = api_key
        Entrez.max_tries = 10 if api_key else 3
        Entrez.sleep_between_tries = 5
        self.use_local_blast = use_local_blast
        self.local_blast_db = local_blast_db
        if self.use_local_blast and not os.path.isfile(f"{self.local_blast_db}.nhr"):
            logger.error("Local BLAST database files not found. Please check the path.")
            self.use_local_blast = False
        self.rate_limiter = RateLimiter()

    @staticmethod
    def _nested_get(data: dict, *keys, default=None):
        """Safely traverse nested dictionaries."""
        for key in keys:
            if not isinstance(data, dict):
                return default
            data = data.get(key, default)
        return data

    @staticmethod
    def _gene_record_to_dict(gene_record, gene_id: str) -> dict:
        """
        Convert an Entrez gene record to a dictionary.
        All extraction logic is inlined for maximum clarity and performance.
        """
        if not gene_record:
            raise ValueError("Empty gene record")

        data = gene_record[0]
        locus = (data.get("Entrezgene_locus") or [{}])[0]

        # Extract common nested paths once
        gene_ref = NCBISearch._nested_get(data, "Entrezgene_gene", "Gene-ref", default={})
        biosource = NCBISearch._nested_get(data, "Entrezgene_source", "BioSource", default={})

        # Process synonyms
        synonyms_raw = gene_ref.get("Gene-ref_syn", [])
        gene_synonyms = []
        if isinstance(synonyms_raw, list):
            for syn in synonyms_raw:
                gene_synonyms.append(syn.get("Gene-ref_syn_E") if isinstance(syn, dict) else str(syn))
        elif synonyms_raw:
            gene_synonyms.append(str(synonyms_raw))

        # Extract location info
        label = locus.get("Gene-commentary_label", "")
        chromosome_match = re.search(r"Chromosome\s+(\S+)", str(label)) if label else None

        seq_interval = NCBISearch._nested_get(
            locus, "Gene-commentary_seqs", 0, "Seq-loc_int", "Seq-interval", default={}
        )
        genomic_location = (
            f"{seq_interval.get('Seq-interval_from')}-{seq_interval.get('Seq-interval_to')}"
            if seq_interval.get('Seq-interval_from') and seq_interval.get('Seq-interval_to')
            else None
        )

        # Extract representative accession
        representative_accession = next(
            (
                product.get("Gene-commentary_accession")
                for product in locus.get("Gene-commentary_products", [])
                if product.get("Gene-commentary_type") == "3"
            ),
            None,
        )

        # Extract function
        function = data.get("Entrezgene_summary") or next(
            (
                comment.get("Gene-commentary_comment")
                for comment in data.get("Entrezgene_comments", [])
                if isinstance(comment, dict)
                and "function" in str(comment.get("Gene-commentary_heading", "")).lower()
            ),
            None,
        )

        return {
            "molecule_type": "DNA",
            "database": "NCBI",
            "id": gene_id,
            "gene_name": gene_ref.get("Gene-ref_locus", "N/A"),
            "gene_description": gene_ref.get("Gene-ref_desc", "N/A"),
            "organism": NCBISearch._nested_get(
                biosource, "BioSource_org", "Org-ref", "Org-ref_taxname", default="N/A"
            ),
            "url": f"https://www.ncbi.nlm.nih.gov/gene/{gene_id}",
            "gene_synonyms": gene_synonyms or None,
            "gene_type": {
                "1": "protein-coding",
                "2": "pseudo",
                "3": "rRNA",
                "4": "tRNA",
                "5": "snRNA",
                "6": "ncRNA",
                "7": "other",
            }.get(str(data.get("Entrezgene_type")), f"type_{data.get('Entrezgene_type')}"),
            "chromosome": chromosome_match.group(1) if chromosome_match else None,
            "genomic_location": genomic_location,
            "function": function,
            # Fields from accession-based queries
            "title": None,
            "sequence": None,
            "sequence_length": None,
            "gene_id": gene_id,
            "molecule_type_detail": None,
            "_representative_accession": representative_accession,
        }

    def get_by_gene_id(self, gene_id: str, preferred_accession: Optional[str] = None) -> Optional[dict]:
        """Get gene information by Gene ID."""
        def _extract_from_genbank(result: dict, accession: str):
            """Enrich result dictionary with sequence and summary information from accession."""
            with Entrez.efetch(db="nuccore", id=accession, rettype="gb", retmode="text") as handle:
                record = SeqIO.read(handle, "genbank")
                result["sequence"] = str(record.seq)
                result["sequence_length"] = len(record.seq)
                result["title"] = record.description
                result["molecule_type_detail"] = (
                    "mRNA" if accession.startswith(("NM_", "XM_")) else
                    "genomic DNA" if accession.startswith(("NC_", "NT_")) else
                    "RNA" if accession.startswith(("NR_", "XR_")) else
                    "genomic region" if accession.startswith("NG_") else "N/A"
                )

                for feature in record.features:
                    if feature.type == "source":
                        if 'chromosome' in feature.qualifiers:
                            result["chromosome"] = feature.qualifiers['chromosome'][0]

                        if feature.location:
                            start = int(feature.location.start) + 1
                            end = int(feature.location.end)
                            result["genomic_location"] = f"{start}-{end}"

                        break

                if not result.get("organism") and 'organism' in record.annotations:
                    result["organism"] = record.annotations['organism']

            return result


        try:
            with Entrez.efetch(db="gene", id=gene_id, retmode="xml") as handle:
                gene_record = Entrez.read(handle)
                if not gene_record:
                    return None

                result = self._gene_record_to_dict(gene_record, gene_id)
                if accession := (preferred_accession or result.get("_representative_accession")):
                    result = _extract_from_genbank(result, accession)

                result.pop("_representative_accession", None)
                return result
        except (RequestException, IncompleteRead):
            raise
        except Exception as exc:
            logger.error("Gene ID %s not found: %s", gene_id, exc)
            return None

    def get_by_accession(self, accession: str) -> Optional[dict]:
        """Get sequence information by accession number."""
        def _extract_gene_id(link_handle):
            """Extract GeneID from elink results."""
            links = Entrez.read(link_handle)
            if not links or "LinkSetDb" not in links[0]:
                return None

            for link_set in links[0]["LinkSetDb"]:
                if link_set.get("DbTo") != "gene":
                    continue

                link = (link_set.get("Link") or link_set.get("IdList", [{}]))[0]
                return str(link.get("Id") if isinstance(link, dict) else link)

        try:
            with Entrez.elink(dbfrom="nuccore", db="gene", id=accession) as link_handle:
                gene_id = _extract_gene_id(link_handle)

            if not gene_id:
                logger.warning("Accession %s has no associated GeneID", accession)
                return None

            result = self.get_by_gene_id(gene_id, preferred_accession=accession)
            if result:
                result["id"] = accession
                result["url"] = f"https://www.ncbi.nlm.nih.gov/nuccore/{accession}"
            return result
        except (RequestException, IncompleteRead):
            raise
        except Exception as exc:
            logger.error("Accession %s not found: %s", accession, exc)
            return None

    def get_best_hit(self, keyword: str) -> Optional[dict]:
        """Search NCBI Gene database with a keyword and return the best hit."""
        if not keyword.strip():
            return None

        try:
            for search_term in [f"{keyword}[Gene Name] OR {keyword}[All Fields]", keyword]:
                with Entrez.esearch(db="gene", term=search_term, retmax=1) as search_handle:
                    if search_results := Entrez.read(search_handle):
                        if gene_id := search_results["IdList"][0]:
                            return self.get_by_gene_id(gene_id)
                self.rate_limiter.wait()
        except (RequestException, IncompleteRead):
            raise
        except Exception as e:
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
            logger.debug("Running local blastn: %s", " ".join(cmd))
            out = subprocess.check_output(cmd, text=True).strip()
            os.remove(tmp_name)
            return out.split("\n", maxsplit=1)[0] if out else None
        except Exception as exc:
            logger.error("Local blastn failed: %s", exc)
            return None

    def search_by_sequence(self, sequence: str, threshold: float = 0.01) -> Optional[dict]:
        """Search NCBI with a DNA sequence using BLAST."""

        def _extract_and_normalize_sequence(sequence: str) -> Optional[str]:
            """Extract and normalize DNA sequence from input."""
            if sequence.startswith(">"):
                seq = "".join(sequence.strip().split("\n")[1:])
            else:
                seq = sequence.strip().replace(" ", "").replace("\n", "")
            return seq if re.fullmatch(r"[ATCGN]+", seq, re.I) else None


        def _process_network_blast_result(blast_record, seq: str, threshold: float) -> Optional[dict]:
            """Process network BLAST result and return dictionary or None."""
            if not blast_record.alignments:
                logger.info("No BLAST hits found for the given sequence.")
                return None

            best_alignment = blast_record.alignments[0]
            best_hsp = best_alignment.hsps[0]
            if best_hsp.expect > threshold:
                logger.info("No BLAST hits below the threshold E-value.")
                return None

            hit_id = best_alignment.hit_id
            if accession_match := re.search(r"ref\|([^|]+)", hit_id):
                return self.get_by_accession(accession_match.group(1).split(".")[0])

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

        try:
            if not (seq := _extract_and_normalize_sequence(sequence)):
                logger.error("Empty or invalid DNA sequence provided.")
                return None

            # Try local BLAST first if enabled
            if self.use_local_blast and (accession := self._local_blast(seq, threshold)):
                logger.debug("Local BLAST found accession: %s", accession)
                return self.get_by_accession(accession)

            # Fall back to network BLAST
            logger.debug("Falling back to NCBIWWW.qblast")

            with NCBIWWW.qblast("blastn", "nr", seq, hitlist_size=1, expect=threshold) as result_handle:
                return _process_network_blast_result(NCBIXML.read(result_handle), seq, threshold)
        except (RequestException, IncompleteRead):
            raise
        except Exception as e:
            logger.error("BLAST search failed: %s", e)
            return None

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((RequestException, IncompleteRead)),
        reraise=True,
    )
    async def search(self, query: str, threshold: float = 0.01, **kwargs) -> Optional[Dict]:
        """Search NCBI with either a gene ID, accession number, keyword, or DNA sequence."""
        if not query or not isinstance(query, str):
            logger.error("Empty or non-string input.")
            return None

        query = query.strip()
        logger.debug("NCBI search query: %s", query)

        loop = asyncio.get_running_loop()

        # Auto-detect query type and execute in thread pool
        if query.startswith(">") or re.fullmatch(r"[ATCGN\s]+", query, re.I):
            result = await loop.run_in_executor(_get_pool(), self.search_by_sequence, query, threshold)
        elif re.fullmatch(r"^\d+$", query):
            result = await loop.run_in_executor(_get_pool(), self.get_by_gene_id, query)
        elif re.fullmatch(r"[A-Z]{2}_\d+\.?\d*", query, re.I):
            result = await loop.run_in_executor(_get_pool(), self.get_by_accession, query)
        else:
            result = await loop.run_in_executor(_get_pool(), self.get_best_hit, query)

        if result:
            result["_search_query"] = query
        return result
