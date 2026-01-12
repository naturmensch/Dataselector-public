# Changelog

## [2026-01-11] - Final master grid fixes
- Fixed 28 sheets using final master_coords list.
- See `data/changes_applied.csv` and `data/duplikate_manual_followup.csv` for remaining manual cases.

## [2026-01-11] - Ultimative Master-Korrektur (David Rumsey Grid)
- Applied verified master_coords to 33 sheets across Masuren, Westpreußen, Bremen-Verden, Mittelkette, Bayern.
- Recomputed RIGHT/BOTTOM/N; geometry validation: 0 errors.
- Filename fix for Sheet 649 → KDR_649.PNG.
- Saved final dataset: data/all_png_tiles_final_ultimative.csv.
- Remaining position duplicates: 28 rows (14 pairs) enumerated for review (e.g., 661↔670, 662↔671, 422↔043, 397↔495, 285↔309, 253↔520, 134↔228, 103↔197, 095↔158, 029↔102, 009↔032, 019↔053, 003↔104, 001↔075).
- Next: add authoritative coords for the above "Hausbesetzer" to master_coords for full deconfliction.

## [2026-01-11] - DBF-Reset und Master-Fix
- Converted original DBF to CSV via dbfread, dropping true duplicates by BLATTNUMME.
- Normalized columns and applied ultimate MASTER_COORDS.
- Outputs: data/all_png_tiles_from_dbf.csv, data/all_png_tiles_final_from_dbf.csv
- Validation: Geometry errors = 0; Remaining position duplicates = 26 rows (13 Paare), v.a. Bayern (635/648, 636/649) und Ostpreußen (z.B. 052/053, 030/031, 018/019, 073/103, 101/102, 075/104, 106/137 etc.).
