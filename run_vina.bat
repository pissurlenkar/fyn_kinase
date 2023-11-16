@echo off

for %%a in ("%~dp0\*.pdbqt") do (
"vina.exe" --ligand "%%a" --config config.txt
)
