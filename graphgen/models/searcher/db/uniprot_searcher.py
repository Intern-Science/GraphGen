import asyncio
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from io import StringIO
from typing import Dict, Optional

from Bio import ExPASy, SeqIO, SwissProt, UniProt
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


class UniProtSearch(BaseSearcher):
    """
    UniProt Search client to searcher with UniProt.
    1) Get the protein by accession number.
    2) Search with keywords or protein names (fuzzy searcher).
    3) Search with FASTA sequence (BLAST searcher).
    """

    def __init__(self, use_local_blast: bool = False, local_blast_db: str = "sp_db"):
        super().__init__()
        self.use_local_blast = use_local_blast
        self.local_blast_db = local_blast_db
        if self.use_local_blast and not os.path.isfile(f"{self.local_blast_db}.phr"):
            logger.error("Local BLAST database files not found. Please check the path.")
            self.use_local_blast = False

    def get_by_accession(self, accession: str) -> Optional[dict]:
        try:
            handle = ExPASy.get_sprot_raw(accession)
            record = SwissProt.read(handle)
            handle.close()
            return self._swissprot_to_dict(record)
        except RequestException:  # network-related errors
            raise
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Accession %s not found: %s", accession, exc)
            return None

    @staticmethod
    def _swissprot_to_dict(record: SwissProt.Record) -> dict:
        """error
        Convert a SwissProt.Record to a dictionary.
        """
        functions = []
        for line in record.comments:
            if line.startswith("FUNCTION:"):
                functions.append(line[9:].strip())

        return {
            "molecule_type": "protein",
            "database": "UniProt",
            "id": record.accessions[0],
            "entry_name": record.entry_name,
            "gene_names": record.gene_name,
            "protein_name": record.description.split(";")[0].split("=")[-1],
            "organism": record.organism.split(" (")[0],
            "sequence": str(record.sequence),
            "function": functions,
            "url": f"https://www.uniprot.org/uniprot/{record.accessions[0]}",
        }

    def get_best_hit(self, keyword: str) -> Optional[Dict]:
        """
        Search UniProt with a keyword and return the best hit.
        :param keyword: The searcher keyword.
        :return: A dictionary containing the best hit information or None if not found.
        """
        if not keyword.strip():
            return None

        try:
            iterator = UniProt.search(keyword, fields=None, batch_size=1)
            hit = next(iterator, None)
            if hit is None:
                return None
            return self.get_by_accession(hit["primaryAccession"])

        except RequestException:
            raise
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Keyword %s not found: %s", keyword, e)
        return None

    def get_by_fasta(self, fasta_sequence: str, threshold: float) -> Optional[Dict]:
        """
        Search UniProt with a FASTA sequence and return the best hit.
        :param fasta_sequence: The FASTA sequence.
        :param threshold: E-value threshold for BLAST searcher.
        :return: A dictionary containing the best hit information or None if not found.
        """
        try:
            if fasta_sequence.startswith(">"):
                seq = str(list(SeqIO.parse(StringIO(fasta_sequence), "fasta"))[0].seq)
            else:
                seq = fasta_sequence.strip()
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Invalid FASTA sequence: %s", e)
            return None

        if not seq:
            logger.error("Empty FASTA sequence provided.")
            return None

        accession = None
        if self.use_local_blast:
            accession = self._local_blast(seq, threshold)
            if accession:
                logger.debug("Local BLAST found accession: %s", accession)

        if not accession:
            logger.debug("Falling back to NCBIWWW.qblast.")

            # UniProtKB/Swiss-Prot BLAST API
            try:
                logger.debug(
                    "Performing BLAST searcher for the given sequence: %s", seq
                )
                result_handle = NCBIWWW.qblast(
                    program="blastp",
                    database="swissprot",
                    sequence=seq,
                    hitlist_size=1,
                    expect=threshold,
                )
                blast_record = NCBIXML.read(result_handle)
            except RequestException:
                raise
            except Exception as e:  # pylint: disable=broad-except
                logger.error("BLAST searcher failed: %s", e)
                return None

            if not blast_record.alignments:
                logger.info("No BLAST hits found for the given sequence.")
                return None

            best_alignment = blast_record.alignments[0]
            best_hsp = best_alignment.hsps[0]
            if best_hsp.expect > threshold:
                logger.info("No BLAST hits below the threshold E-value.")
                return None
            hit_id = best_alignment.hit_id

            # like sp|P01308.1|INS_HUMAN
            accession = hit_id.split("|")[1].split(".")[0] if "|" in hit_id else hit_id
        return self.get_by_accession(accession)

    def _local_blast(self, seq: str, threshold: float) -> Optional[str]:
        """
        Perform local BLAST search using local BLAST database.
        :param seq: The protein sequence.
        :param threshold: E-value threshold for BLAST searcher.
        :return: The accession number of the best hit or None if not found.
        """
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+", suffix=".fa", delete=False
            ) as tmp:
                tmp.write(f">query\n{seq}\n")
                tmp_name = tmp.name

            cmd = [
                "blastp",
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
            logger.debug("Running local blastp: %s", " ".join(cmd))
            out = subprocess.check_output(cmd, text=True).strip()
            os.remove(tmp_name)
            if out:
                return out.split("\n", maxsplit=1)[0]
            return None
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Local blastp failed: %s", exc)
            return None

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(RequestException),
        reraise=True,
    )
    async def search(
        self, query: str, threshold: float = 0.7, **kwargs
    ) -> Optional[Dict]:
        """
        Search UniProt with either an accession number, keyword, or FASTA sequence.
        :param query: The searcher query (accession number, keyword, or FASTA sequence).
        :param threshold: E-value threshold for BLAST searcher.
        :return: A dictionary containing the best hit information or None if not found.
        """

        # auto detect query type
        if not query or not isinstance(query, str):
            logger.error("Empty or non-string input.")
            return None
        query = query.strip()

        logger.debug("UniProt searcher query: %s", query)

        loop = asyncio.get_running_loop()

        # check if fasta sequence
        if query.startswith(">") or re.fullmatch(
            r"[ACDEFGHIKLMNPQRSTVWY\s]+", query, re.I
        ):
            coro = loop.run_in_executor(
                _get_pool(), self.get_by_fasta, query, threshold
            )

        # check if accession number
        elif re.fullmatch(r"[A-NR-Z0-9]{6,10}", query, re.I):
            coro = loop.run_in_executor(_get_pool(), self.get_by_accession, query)

        else:
            # otherwise treat as keyword
            coro = loop.run_in_executor(_get_pool(), self.get_best_hit, query)

        result = await coro
        if result:
            result["_search_query"] = query
        return result


# TODO: use local UniProt database for large-scale searchs
