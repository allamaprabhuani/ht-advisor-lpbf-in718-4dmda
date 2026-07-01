# LPBF IN718 literature extraction summary

## Corpus

- PDFs in combined corpus: 18
- Extracted PDF tables: 70
- Source text dumps: `ml_project/extracted_data/text/`
- Extracted table CSVs: `ml_project/extracted_data/tables/`

## Candidate data files

- `excel_seed_mechanical_dataset.csv`: 23 rows from the original workbook with simple heat-treatment class flags.
- `candidate_heat_treatments.csv`: 134 PDF excerpts containing heat-treatment terms plus temperatures/times.
- `candidate_mechanical_properties.csv`: 745 PDF excerpts containing mechanical-property terms or values.
- `candidate_fatigue_data.csv`: 562 PDF excerpts containing fatigue/S-N terms.
- `paper_inventory.csv`: per-PDF page counts, text character counts, table counts, and term flags.

## Machine-learning readiness

- The Excel seed table is immediately usable for a small baseline analysis, but it is too sparse for the final expert system.
- The PDF candidate files are audit-ready extraction layers, not final cleaned training data. The next step is manual/LLM-assisted curation into one row per heat-treatment condition and one row per digitised S-N point.
- The credible model remains a physics-informed hierarchical Bayesian/GP surrogate, not a deep model trained from scratch.

## Direct-download blockers

- OA01: Effects of Print Parameters and Heat Treatment on Fatigue of Laser Powder Bed Fused Inconel 718 - https://oaktrust.library.tamu.edu/bitstreams/e06ac7ee-208b-4613-88c3-cea2181bc623/download
- OA02: Optimization of the Post-Process Heat Treatment of Inconel 718 Superalloy Fabricated by LPBF - https://www.mdpi.com/2075-4701/11/1/144/pdf
- OA03: Influence of Homogenization and Solution Treatments Time on Microstructure and Hardness of Inconel 718 Fabricated by LPBF - https://www.mdpi.com/1996-1944/13/11/2574/pdf
- OA05: Effects of Process Parameters and Heat Treatment on Microstructure and Mechanical Characteristics of LPBF Inconel 718 - https://www.mdpi.com/2079-6412/13/1/189/pdf
- OA07: Tailoring Heat Treatments for Metals Processed by Laser Powder Bed Fusion - https://bib-pubdb1.desy.de/record/476532/files/2022%20Van%20Cauwenbergh%20-%20Tailoring%20heat%20treatments%20for%20metals%20processed%20by%20laser%20powder%20bed%20fusion.pdf
- OA09: Improved Properties of Additively Prepared Inconel 718 Alloy Post-Processed with a New Heat Treatment - https://papers.ssrn.com/sol3/Delivery.cfm/0fbcf52c-0205-4de0-b004-01557f709baf-MECA.pdf?abstractid=4798171
- OA11: Microstructure Evolution in Inconel 718 Produced by Powder Bed Fusion - https://opus4.kobv.de/opus4-bam/frontdoor/deliver/index/docId/54275/file/10_3390_jmmp6010020.pdf

## AM-only filtering update

- Current combined corpus after adding Wiley: 19 PDFs.
- Core AM-only filtered candidate files created: `am_only_candidate_heat_treatments.csv`, `am_only_candidate_mechanical_properties.csv`, and `am_only_candidate_fatigue_data.csv`.
- Excluded from core training: trade/industry-only background sources unless manually promoted.
- Mixed LPBF-versus-forged/rolled rows are retained but marked with `manual_filter_note` when comparator terms appear.
- Machine scan created: `machine_term_scan.csv`; several sources use EOS/EOSINT M270/M280/M290 machines, which is useful for aligning to your EOS LPBF validation.
- Manual download priority list: `manual_download_recommendations.csv`.
