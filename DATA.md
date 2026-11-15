# Data schema

## Corpus and correction

`corpus-original.jsonl` and `corpus-corrected.jsonl` contain one document family per line. A family identifies its structural design, public instance, attribute type, split, four ordered candidates, public target and reader. Its `worlds` contain the truth index and ordered chunks. Each chunk has an identifier, text and complete reader list. `utility` contains two public queries and their relevant chunk identifiers. `utility_context` identifies the event and deadline used to generate competing public events.

`correction.json` records the 60 changed sentences and checksums of the two distributed corpora. `generate_corpus.py` regenerates both corpora and compares every scientific field. `length-controls.json` retains the precomputed token counts and equal-length indicators for both pinned models.

## Observation files

All files in `observations/` use gzip compression. The uncompressed checksums refer to the original saved observations. JSONL has one attack record per line; JSON contains the grouped records and original summaries.

| File | Records | Role |
|---|---:|---|
| `security.jsonl.gz` | 5,760 | 2 models × 4 encodings × 3 endpoints × 60 held-out families × 4 worlds |
| `candidate-only.jsonl.gz` | 5,760 | Same conditions with candidate-only queries |
| `query-selection.jsonl.gz` | 480 | 2 query forms × 60 development families × 4 worlds; small model, L/S |
| `sensitivity.json.gz` | 31,680 | 15,840 development records per model, including invariant controls |
| `controlled-utility.json.gz` | 3,840 | 2 models × 4 encodings × 60 families × 4 worlds × 2 queries |
| `scifact.json.gz` | 1,200 | 2 models × 2 distinct encoding inputs × 300 test queries |
| `indexing-cost.json.gz` | 8 conditions | 2 models × 4 encodings, with five timings per condition |

An attack record includes `model`, `method`, `tier`, family/design/instance identifiers, attribute type, world, truth, prediction, correctness, candidate decision values, tie indicator, absent-target count, and a four-entry `transcript`. Each transcript entry contains the query and all returned rows. S rows expose identifier, text, rank and score. R rows expose identifier, text and rank. O exposes only the authorized target's identifier and direct score. The oracle is not a top-10 search interface.

Held-out records also identify the selected query form and equal-token-length membership. Sensitivity records identify restricted-span position, filler-token count, changed target wording, constant hidden context and distractor count. Their selected development families are recorded separately for each model.

Controlled-utility records contain returned rows and query identifiers. Relevance is reconstructed from the corrected corpus using the published public-identifier mapping. SciFact records contain ranked document identifiers; test relevance judgments are provided in `scifact-qrels.tsv`. P and A share L's inputs on all-public SciFact and are not represented as separately executed conditions.

Indexing records retain the five timing samples, invocation and stored-vector counts, and the complete five-chunk/four-principal fixture. Timing summaries reproduce the original measurements; they do not predict timings on another machine.

## Provenance

`provenance.json` records original uncompressed observation hashes, compressed hashes, the scientific source signature and implementation-file hashes. The corpora retain original and corrected text, identifiers, permissions, query inputs, labels and split assignments. The correction timing and absence of independent human semantic review are stated in the README.

The data include the weak and null sensitivity conditions. No observations are selected by attack success. The Figure 2 family is the lexicographically first held-out month family, selected after evaluation as an illustration.
