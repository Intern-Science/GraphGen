#!/bin/bash

set -e

# Downloads NCBI RefSeq RNA sequences and creates BLAST databases.
# This script specifically downloads RNA sequences (mRNA, rRNA, tRNA, etc.)
# from RefSeq, which is suitable for RNA sequence searches.
#
# Usage: ./build_rna_blast_db.sh [representative|complete|all]
#   representative: Download RNA sequences from major categories (recommended, smaller)
#                    Includes: vertebrate_mammalian, vertebrate_other, bacteria, archaea, fungi, invertebrate, plant, viral
#   complete: Download all RNA sequences from complete/ directory (very large)
#   all: Download all RNA sequences from all categories (very large)
#
# We need makeblastdb on our PATH
# For Ubuntu/Debian: sudo apt install ncbi-blast+
# For CentOS/RHEL/Fedora: sudo dnf install ncbi-blast+
# Or download from: https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/

DOWNLOAD_TYPE=${1:-representative}

# Better to use a stable DOWNLOAD_TMP name to support resuming downloads
DOWNLOAD_TMP=_downloading_rna
mkdir -p ${DOWNLOAD_TMP}
cd ${DOWNLOAD_TMP}

# Download RefSeq release information
echo "Downloading RefSeq release information..."
wget -c "https://ftp.ncbi.nlm.nih.gov/refseq/release/RELEASE_NUMBER" || {
    echo "Warning: Could not download RELEASE_NUMBER, using current date as release identifier"
    RELEASE=$(date +%Y%m%d)
}

if [ -f "RELEASE_NUMBER" ]; then
    RELEASE=$(cat RELEASE_NUMBER | tr -d '\n')
    echo "RefSeq release: ${RELEASE}"
else
    RELEASE=$(date +%Y%m%d)
    echo "Using date as release identifier: ${RELEASE}"
fi

# Download based on type
case ${DOWNLOAD_TYPE} in
    representative)
        echo "Downloading RefSeq representative RNA sequences (recommended, smaller size)..."
        echo "Downloading RNA sequences from major categories..."
        for category in vertebrate_mammalian vertebrate_other bacteria archaea fungi invertebrate plant viral; do
            echo "Downloading ${category} RNA sequences..."
            curl -s "https://ftp.ncbi.nlm.nih.gov/refseq/release/${category}/" | \
                grep -oE 'href="[^"]*\.rna\.fna\.gz"' | \
                sed 's/href="\(.*\)"/\1/' | \
                while read filename; do
                    echo "  Downloading ${filename}..."
                    wget -c -q --show-progress \
                        "https://ftp.ncbi.nlm.nih.gov/refseq/release/${category}/${filename}" || {
                        echo "Warning: Failed to download ${filename}"
                    }
                done
        done
        ;;
    complete)
        echo "Downloading RefSeq complete RNA sequences (WARNING: very large, may take hours)..."
        curl -s "https://ftp.ncbi.nlm.nih.gov/refseq/release/complete/" | \
            grep -oE 'href="[^"]*\.rna\.fna\.gz"' | \
            sed 's/href="\(.*\)"/\1/' | \
            while read filename; do
                echo "  Downloading ${filename}..."
                wget -c -q --show-progress \
                    "https://ftp.ncbi.nlm.nih.gov/refseq/release/complete/${filename}" || {
                    echo "Warning: Failed to download ${filename}"
                }
            done
        ;;
    all)
        echo "Downloading all RefSeq RNA sequences from all categories (WARNING: extremely large, may take many hours)..."
        for category in vertebrate_mammalian vertebrate_other bacteria archaea fungi invertebrate plant viral protozoa mitochondrion plastid plasmid other; do
            echo "Downloading ${category} RNA sequences..."
            curl -s "https://ftp.ncbi.nlm.nih.gov/refseq/release/${category}/" | \
                grep -oE 'href="[^"]*\.rna\.fna\.gz"' | \
                sed 's/href="\(.*\)"/\1/' | \
                while read filename; do
                    echo "  Downloading ${filename}..."
                    wget -c -q --show-progress \
                        "https://ftp.ncbi.nlm.nih.gov/refseq/release/${category}/${filename}" || {
                        echo "Warning: Failed to download ${filename}"
                    }
                done
        done
        ;;
    *)
        echo "Error: Unknown download type '${DOWNLOAD_TYPE}'"
        echo "Usage: $0 [representative|complete|all]"
        exit 1
        ;;
esac

cd ..

# Create release directory
mkdir -p refseq_rna_${RELEASE}
mv ${DOWNLOAD_TMP}/* refseq_rna_${RELEASE}/ 2>/dev/null || true
rmdir ${DOWNLOAD_TMP} 2>/dev/null || true

cd refseq_rna_${RELEASE}

# Extract and combine sequences
echo "Extracting and combining RNA sequences..."

# Extract all downloaded RNA sequences
if [ $(find . -name "*.rna.fna.gz" -type f | wc -l) -gt 0 ]; then
    echo "Extracting RNA sequences..."
    find . -name "*.rna.fna.gz" -type f -exec gunzip {} \;
fi

# Combine all FASTA files into one
echo "Combining all FASTA files..."
FASTA_FILES=$(find . -name "*.fna" -type f)
if [ -z "$FASTA_FILES" ]; then
    FASTA_FILES=$(find . -name "*.fa" -type f)
fi

if [ -z "$FASTA_FILES" ]; then
    echo "Error: No FASTA files found to combine"
    exit 1
fi

echo "$FASTA_FILES" | while read -r file; do
    if [ -f "$file" ]; then
        cat "$file" >> refseq_rna_${RELEASE}.fasta
    fi
done

# Check if we have sequences
if [ ! -s "refseq_rna_${RELEASE}.fasta" ]; then
    echo "Error: Combined FASTA file is empty"
    exit 1
fi

echo "Creating BLAST database..."
# Create BLAST database for RNA sequences (use -dbtype nucl for nucleotide)
makeblastdb -in refseq_rna_${RELEASE}.fasta \
    -out refseq_rna_${RELEASE} \
    -dbtype nucl \
    -parse_seqids \
    -title "RefSeq_RNA_${RELEASE}"

echo "BLAST database created successfully!"
echo "Database location: $(pwd)/refseq_rna_${RELEASE}"
echo ""
echo "To use this database, set in your config:"
echo "  local_blast_db: $(pwd)/refseq_rna_${RELEASE}"
echo ""
echo "Note: The database files are:"
ls -lh refseq_rna_${RELEASE}.*

cd ..

