#! /bin/bash

for f in ligand_*.pdbqt; do
    b=`basename $f .pdbqt`
    echo Processing $b
    mkdir -p $b
    vina --config config.txt --ligand $f
done
