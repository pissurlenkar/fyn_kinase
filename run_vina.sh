#!/bin/bash

directory="$(dirname "$0")"

for file in "$directory"/*.pdbqt; do
    if [ "$file" != "$directory/protein.pdbqt" ]; then
        vina --ligand "$file" --config config.txt
    fi
done
