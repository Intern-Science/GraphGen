#!/bin/bash

set -e

# Downloads RNAcentral sequences and creates BLAST databases.
# This script downloads the RNAcentral active database, which is the same
# data source used for online RNAcentral searches, ensuring consistency
# between local and online search results.
#
# RNAcentral is a comprehensive database of non-coding RNA sequences that
# integrates data from multiple expert databases including RefSeq, Rfam, etc.
#
# Usage: ./build_rna_blast_db.sh [all|list|database_name]
#   all (default): Download complete active database (~8.4G compressed)
#   list: List all available database subsets
#   database_name: Download specific database subset (e.g., refseq, rfam, mirbase)
#
# Available database subsets (examples):
#   - refseq.fasta (~98M): RefSeq RNA sequences
#   - rfam.fasta (~1.5G): Rfam RNA families
#   - mirbase.fasta (~10M): microRNA sequences
#   - ensembl.fasta (~2.9G): Ensembl annotations
#   - See "list" option for complete list
#
# The complete "active" database contains all sequences from all expert databases.
# Using a specific database subset provides a smaller, focused database.
#
# We need makeblastdb on our PATH
# For Ubuntu/Debian: sudo apt install ncbi-blast+
# For CentOS/RHEL/Fedora: sudo dnf install ncbi-blast+
# Or download from: https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/

# RNAcentral HTTP base URL (using HTTPS for better reliability)
RNACENTRAL_BASE="https://ftp.ebi.ac.uk/pub/databases/RNAcentral"
RNACENTRAL_RELEASE_URL="${RNACENTRAL_BASE}/current_release"
RNACENTRAL_SEQUENCES_URL="${RNACENTRAL_RELEASE_URL}/sequences"
RNACENTRAL_BY_DB_URL="${RNACENTRAL_SEQUENCES_URL}/by-database"

# Parse command line argument
DB_SELECTION=${1:-all}

# List available databases if requested
if [ "${DB_SELECTION}" = "list" ]; then
    echo "Available RNAcentral database subsets:"
    echo ""
    echo "Fetching list from RNAcentral FTP..."
    curl -s "${RNACENTRAL_BY_DB_URL}/" | \
        grep -oE '<a href="[^"]*\.fasta">' | \
        sed 's/<a href="//;s/">//' | \
        sort | \
        while read db; do
            size=$(curl -s "${RNACENTRAL_BY_DB_URL}/" | grep -A 1 "${db}" | grep -oE '[0-9.]+[GMK]' | head -1 || echo "unknown")
            echo "  - ${db%.fasta}: ${size}"
        done
    echo ""
    echo "Usage: $0 [database_name]"
    echo "  Example: $0 refseq    # Download only RefSeq sequences (~98M)"
    echo "  Example: $0 rfam      # Download only Rfam sequences (~1.5G)"
    echo "  Example: $0 all       # Download complete active database (~8.4G)"
    exit 0
fi

# Better to use a stable DOWNLOAD_TMP name to support resuming downloads
DOWNLOAD_TMP=_downloading_rnacentral
mkdir -p ${DOWNLOAD_TMP}
cd ${DOWNLOAD_TMP}

# Get RNAcentral release version from release notes
echo "Getting RNAcentral release information..."
RELEASE_NOTES_URL="${RNACENTRAL_RELEASE_URL}/release_notes.txt"
RELEASE_NOTES="release_notes.txt"
wget -q "${RELEASE_NOTES_URL}" 2>/dev/null || {
    echo "Warning: Could not download release notes, using current date as release identifier"
    RELEASE=$(date +%Y%m%d)
}

if [ -f "${RELEASE_NOTES}" ]; then
    # Try to extract version from release notes (first line usually contains version info)
    RELEASE=$(head -1 "${RELEASE_NOTES}" | grep -oE '[0-9]+\.[0-9]+' | head -1 | tr -d '.' || date +%Y%m%d)
    if [ -z "${RELEASE}" ] || [ "${RELEASE}" = "$(date +%Y%m%d)" ]; then
        RELEASE=$(date +%Y%m%d)
        echo "Using date as release identifier: ${RELEASE}"
    else
        echo "RNAcentral release: ${RELEASE}"
    fi
else
    RELEASE=$(date +%Y%m%d)
    echo "Using date as release identifier: ${RELEASE}"
fi

# Download RNAcentral FASTA file
if [ "${DB_SELECTION}" = "all" ]; then
    # Download complete active database
    FASTA_FILE="rnacentral_active.fasta.gz"
    DB_NAME="rnacentral"
    echo "Downloading RNAcentral active sequences (~8.4G)..."
    echo "  Contains sequences currently present in at least one expert database"
    echo "  Uses standard URS IDs (e.g., URS000149A9AF)"
    echo "  ⭐ MATCHES the online RNAcentral API database - ensures consistency"
    FASTA_URL="${RNACENTRAL_SEQUENCES_URL}/${FASTA_FILE}"
    IS_COMPRESSED=true
else
    # Download specific database subset
    DB_NAME="${DB_SELECTION}"
    FASTA_FILE="${DB_SELECTION}.fasta"
    echo "Downloading RNAcentral database subset: ${DB_SELECTION}"
    echo "  This is a subset of the active database from a specific expert database"
    echo "  File: ${FASTA_FILE}"
    FASTA_URL="${RNACENTRAL_BY_DB_URL}/${FASTA_FILE}"
    IS_COMPRESSED=false
    
    # Check if database exists
    if ! curl -s -o /dev/null -w "%{http_code}" "${FASTA_URL}" | grep -q "200"; then
        echo "Error: Database '${DB_SELECTION}' not found"
        echo "Run '$0 list' to see available databases"
        exit 1
    fi
fi

echo "Downloading from: ${FASTA_URL}"
echo "This may take a while depending on your internet connection..."
if [ "${DB_SELECTION}" = "all" ]; then
    echo "File size is approximately 8-9GB, please be patient..."
else
    echo "Downloading database subset..."
fi
wget -c --progress=bar:force "${FASTA_URL}" 2>&1 || {
    echo "Error: Failed to download RNAcentral FASTA file"
    echo "Please check your internet connection and try again"
    echo "You can also try downloading manually from: ${FASTA_URL}"
    exit 1
}

if [ ! -f "${FASTA_FILE}" ]; then
    echo "Error: Downloaded file not found"
    exit 1
fi

cd ..

# Create release directory
if [ "${DB_SELECTION}" = "all" ]; then
    OUTPUT_DIR="rnacentral_${RELEASE}"
else
    OUTPUT_DIR="rnacentral_${DB_NAME}_${RELEASE}"
fi
mkdir -p ${OUTPUT_DIR}
mv ${DOWNLOAD_TMP}/* ${OUTPUT_DIR}/ 2>/dev/null || true
rmdir ${DOWNLOAD_TMP} 2>/dev/null || true

cd ${OUTPUT_DIR}

# Extract FASTA file if compressed
echo "Preparing RNAcentral sequences..."
if [ -f "${FASTA_FILE}" ]; then
    if [ "${IS_COMPRESSED}" = "true" ]; then
        echo "Decompressing ${FASTA_FILE}..."
        OUTPUT_FASTA="${DB_NAME}_${RELEASE}.fasta"
        gunzip -c "${FASTA_FILE}" > "${OUTPUT_FASTA}" || {
            echo "Error: Failed to decompress FASTA file"
            exit 1
        }
        # Optionally remove the compressed file to save space
        # rm "${FASTA_FILE}"
    else
        # File is not compressed, just copy/rename
        OUTPUT_FASTA="${DB_NAME}_${RELEASE}.fasta"
        cp "${FASTA_FILE}" "${OUTPUT_FASTA}" || {
            echo "Error: Failed to copy FASTA file"
            exit 1
        }
    fi
else
    echo "Error: FASTA file not found"
    exit 1
fi

# Check if we have sequences
if [ ! -s "${OUTPUT_FASTA}" ]; then
    echo "Error: FASTA file is empty"
    exit 1
fi

# Get file size for user information
FILE_SIZE=$(du -h "${OUTPUT_FASTA}" | cut -f1)
echo "FASTA file size: ${FILE_SIZE}"

echo "Creating BLAST database..."
# Create BLAST database for RNA sequences (use -dbtype nucl for nucleotide)
# Note: RNAcentral uses RNAcentral IDs (URS...) as sequence identifiers,
# which matches the format expected by the RNACentralSearch class
DB_OUTPUT_NAME="${DB_NAME}_${RELEASE}"
makeblastdb -in "${OUTPUT_FASTA}" \
    -out "${DB_OUTPUT_NAME}" \
    -dbtype nucl \
    -parse_seqids \
    -title "RNAcentral_${DB_NAME}_${RELEASE}"

echo ""
echo "BLAST database created successfully!"
echo "Database location: $(pwd)/${DB_OUTPUT_NAME}"
echo ""
echo "To use this database, set in your config (search_rna_config.yaml):"
echo "  rnacentral_params:"
echo "    use_local_blast: true"
echo "    local_blast_db: $(pwd)/${DB_OUTPUT_NAME}"
echo ""
echo "Note: The database files are:"
ls -lh ${DB_OUTPUT_NAME}.* | head -5
echo ""
if [ "${DB_SELECTION}" = "all" ]; then
    echo "This database uses RNAcentral IDs (URS...), which matches the online"
    echo "RNAcentral search API, ensuring consistent results between local and online searches."
else
    echo "This is a subset database from ${DB_SELECTION} expert database."
    echo "For full coverage matching online API, use 'all' option."
fi

cd ..

