@echo off

for %%a in ("%~dp0\Ligand\*.pdbqt") do (
"vina.exe" --ligand "%%a" --config config.txt
)