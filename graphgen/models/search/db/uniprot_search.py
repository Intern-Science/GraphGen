from io import StringIO
from typing import Dict, Optional

from Bio import ExPASy, SeqIO, SwissProt, UniProt
from Bio.Blast import NCBIWWW, NCBIXML

from graphgen.utils import logger


class UniProtSearch:
    """
    UniProt Search client to search with UniProt.
    1) Get the protein by accession number.
    2) Search with keywords or protein names (fuzzy search).
    """

    def get_by_accession(self, accession: str) -> Optional[dict]:
        try:
            handle = ExPASy.get_sprot_raw(accession)
            record = SwissProt.read(handle)
            handle.close()
            return self._swissprot_to_dict(record)
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
        :param keyword: The search keyword.
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

        except Exception as e:  # pylint: disable=broad-except
            logger.error("Keyword %s not found: %s", keyword, e)
            return None

    def get_by_fasta(self, fasta_sequence: str, threshold: float) -> Optional[Dict]:
        """
        Search UniProt with a FASTA sequence and return the best hit.
        :param fasta_sequence: The FASTA sequence.
        :param threshold: E-value threshold for BLAST search.
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

        # UniProtKB/Swiss-Prot BLAST API
        try:
            result_handle = NCBIWWW.qblast(
                program="blastp",
                database="swissprot",
                sequence=seq,
                hitlist_size=1,
                expect=threshold,
            )
            blast_record = NCBIXML.read(result_handle)
        except Exception as e:  # pylint: disable=broad-except
            logger.error("BLAST search failed: %s", e)
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
