#!/bin/bash

set -e

# Downloads the latest release of UniProt, putting it in a release-specific directory.
# Creates associated BLAST databases.
# We need makeblastdb on our PATH
# For Ubuntu/Debian: sudo apt install ncbi-blast+
# For CentOS/RHEL/Fedora: sudo dnf install ncbi-blast+
# Or download from: https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/

# Better to use a stable DOWNLOAD_TMP name to support resuming downloads
DOWNLOAD_TMP=_downloading
mkdir -p ${DOWNLOAD_TMP}
cd ${DOWNLOAD_TMP}

wget -c "ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/RELEASE.metalink"

# Extract the release name (like 2017_10 or 2017_1)
# Use sed for cross-platform compatibility (works on both macOS and Linux)
RELEASE=$(sed -n 's/.*<version>\([0-9]\{4\}_[0-9]\{1,2\}\)<\/version>.*/\1/p' RELEASE.metalink | head -1)

wget -c "ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz"
wget -c "ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_trembl.fasta.gz"
wget -c "ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/reldate.txt"
wget -c "ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/README"
wget -c "ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/LICENSE"

cd ..

mkdir ${RELEASE}
mv ${DOWNLOAD_TMP}/* ${RELEASE}
rmdir ${DOWNLOAD_TMP}

cd ${RELEASE}

gunzip uniprot_sprot.fasta.gz
gunzip uniprot_trembl.fasta.gz

cat uniprot_sprot.fasta uniprot_trembl.fasta >uniprot_${RELEASE}.fasta

makeblastdb -in uniprot_${RELEASE}.fasta -out uniprot_${RELEASE} -dbtype prot -parse_seqids -title uniprot_${RELEASE}
makeblastdb -in uniprot_sprot.fasta -out uniprot_sprot -dbtype prot -parse_seqids -title uniprot_sprot
makeblastdb -in uniprot_trembl.fasta -out uniprot_trembl -dbtype prot -parse_seqids -title uniprot_trembl

cd ..

echo "BLAST databases created successfully!"
echo "Database locations:"
echo "  - Combined: $(pwd)/${RELEASE}/uniprot_${RELEASE}"
echo "  - Swiss-Prot: $(pwd)/${RELEASE}/uniprot_sprot"
echo "  - TrEMBL: $(pwd)/${RELEASE}/uniprot_trembl"
echo ""
echo "To use these databases, set in your config:"
echo "  local_blast_db: $(pwd)/${RELEASE}/uniprot_sprot  # or uniprot_${RELEASE} or uniprot_trembl"

