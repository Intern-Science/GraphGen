#!/bin/bash

set -e

# Downloads NCBI RefSeq nucleotide sequences and creates BLAST databases.
# 
# RefSeq 目录结构说明（按生物分类组织）：
#   - vertebrate_mammalian (哺乳动物)
#   - vertebrate_other (其他脊椎动物)
#   - bacteria (细菌)
#   - archaea (古菌)
#   - fungi (真菌)
#   - invertebrate (无脊椎动物)
#   - plant (植物)
#   - viral (病毒)
#   - protozoa (原生动物)
#   - mitochondrion (线粒体)
#   - plastid (质体)
#   - plasmid (质粒)
#   - other (其他)
#   - complete/ (完整基因组，包含所有分类)
#
# 每个分类目录下包含：
#   - {category}.{number}.genomic.fna.gz (基因组序列)
#   - {category}.{number}.rna.fna.gz (RNA序列)
#
# Usage: ./build_dna_blast_db.sh [representative|complete|all]
#   representative: Download genomic sequences from major categories (recommended, smaller)
#                    Includes: vertebrate_mammalian, vertebrate_other, bacteria, archaea, fungi
#   complete: Download all complete genomic sequences from complete/ directory (very large)
#   all: Download all genomic sequences from all categories (very large)
#
# We need makeblastdb on our PATH
# For Ubuntu/Debian: sudo apt install ncbi-blast+
# For CentOS/RHEL/Fedora: sudo dnf install ncbi-blast+
# Or download from: https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/

DOWNLOAD_TYPE=${1:-representative}

# Better to use a stable DOWNLOAD_TMP name to support resuming downloads
DOWNLOAD_TMP=_downloading_dna
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
        echo "Downloading RefSeq representative sequences (recommended, smaller size)..."
        # Download major categories for representative coverage
        # Note: You can modify this list based on your specific requirements
        for category in vertebrate_mammalian vertebrate_other bacteria archaea fungi; do
            echo "Downloading ${category} sequences..."
            curl -s "https://ftp.ncbi.nlm.nih.gov/refseq/release/${category}/" | \
                grep -oE 'href="[^"]*\.genomic\.fna\.gz"' | \
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
        echo "Downloading RefSeq complete genomic sequences (WARNING: very large, may take hours)..."
        curl -s "https://ftp.ncbi.nlm.nih.gov/refseq/release/complete/" | \
            grep -oE 'href="[^"]*\.genomic\.fna\.gz"' | \
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
        echo "Downloading all RefSeq genomic sequences from all categories (WARNING: extremely large, may take many hours)..."
        # Download genomic sequences from all categories
        for category in vertebrate_mammalian vertebrate_other bacteria archaea fungi invertebrate plant viral protozoa mitochondrion plastid plasmid other; do
            echo "Downloading ${category} genomic sequences..."
            curl -s "https://ftp.ncbi.nlm.nih.gov/refseq/release/${category}/" | \
                grep -oE 'href="[^"]*\.genomic\.fna\.gz"' | \
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
        echo "Note: For RNA sequences, use build_rna_blast_db.sh instead"
        exit 1
        ;;
esac

cd ..

# Create release directory
mkdir -p refseq_${RELEASE}
mv ${DOWNLOAD_TMP}/* refseq_${RELEASE}/ 2>/dev/null || true
rmdir ${DOWNLOAD_TMP} 2>/dev/null || true

cd refseq_${RELEASE}

# Extract and combine sequences
echo "Extracting and combining sequences..."

# Extract all downloaded genomic sequences
if [ $(find . -name "*.genomic.fna.gz" -type f | wc -l) -gt 0 ]; then
    echo "Extracting genomic sequences..."
    find . -name "*.genomic.fna.gz" -type f -exec gunzip {} \;
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
        cat "$file" >> refseq_${RELEASE}.fasta
    fi
done

# Check if we have sequences
if [ ! -s "refseq_${RELEASE}.fasta" ]; then
    echo "Error: Combined FASTA file is empty"
    exit 1
fi

echo "Creating BLAST database..."
# Create BLAST database for DNA sequences (use -dbtype nucl for nucleotide)
makeblastdb -in refseq_${RELEASE}.fasta \
    -out refseq_${RELEASE} \
    -dbtype nucl \
    -parse_seqids \
    -title "RefSeq_${RELEASE}"

echo "BLAST database created successfully!"
echo "Database location: $(pwd)/refseq_${RELEASE}"
echo ""
echo "To use this database, set in your config:"
echo "  local_blast_db: $(pwd)/refseq_${RELEASE}"
echo ""
echo "Note: The database files are:"
ls -lh refseq_${RELEASE}.*

cd ..

