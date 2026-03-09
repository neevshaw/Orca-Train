set -e
cd orca
cd resources
wget -qO- https://zenodo.org/record/6234936/files/resources_core.tar.gz | \
tar -xz resources_core/Homo_sapiens.GRCh38.dna.primary_assembly.fa
rm resources_core.tar.gz

cd misc
python make_genome_memmap.py
