# Restricted-attribute inference in late-chunked retrieval

Reproducibility materials for **An Empirical Study of Restricted-Attribute Inference in Late-Chunked Retrieval**.

Lalfakzuala and Lalhruaizela Chhangte  
Department of Computer Science and Engineering, National Institute of Technology Mizoram  
Contact: dt25cs002@nitmz.ac.in; lalhruaizela.cse@nitmz.ac.in

The package provides the deterministic corpus generator, original and corrected synthetic corpora, sentence-level correction ledger, saved search responses, analysis code and optional model-inference code. Raw observations retain the returned rows and original floating-point values.

## Reconstruct the results

Use Python 3.12 on Linux. Clone the repository, or extract a release archive and enter its directory:

```bash
git clone https://github.com/netaimllab/late_chunk.git
cd late_chunk
```

Then install the analysis dependencies and reconstruct the results:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python verify.py
.venv/bin/python generate_corpus.py --output generated-corpus
.venv/bin/python reproduce.py --output reproduced-results
```

Each output directory must be new. Analysis replay uses NumPy, pandas and Matplotlib; it requires no model download or GPU. `verify.py` uses only the Python standard library. If needed, `uv venv --python python3.12 .venv` and `uv pip install --python .venv/bin/python -r requirements.txt` provide an alternative environment setup.

The generator verifies all 240 families and the 15-family/60-world correction. The analysis reconstructs predictions, tie handling, target absence, retrieval metrics, timing summaries and descriptive intervals. Numerical comparisons use absolute tolerance `1e-12`. Expected results and validation scope are recorded in `expected/` and `VALIDATION.json`.

| Manuscript result | Generated output |
|---|---|
| Table 2: held-out security estimates | `table-2.csv` |
| Table 3: controlled retrieval | `table-3.csv` |
| Table 4: all-public SciFact retrieval | `table-4.csv` |
| Table 5: indexing calls, vectors and timing | `table-5.csv` |
| Table 6: development sensitivity | `table-6.csv` |
| Figure 2: four-world search example | `figure-2.csv`, `figure-2.png` |
| Equal-length, per-attribute and paired estimates | `security-summary.json`, `attribute-accuracy.csv` |
| Candidate-only comparison | `candidate-only.csv` |
| Controlled-utility uncertainty | `utility-intervals.json` |
| All development conditions | `sensitivity-all.csv` |

Output values retain full precision; manuscript tables round them. Figure 2 uses the same selected family and observations. Table 1 compares related work. Figure 1 illustrates the encoding groups implemented by `document()` in `src/chunkprivacy/encoders.py`.

## Optional model inference

Install the additional dependencies:

```bash
.venv/bin/python -m pip install -r requirements-inference.txt
```

A small end-to-end check uses one held-out month family:

```bash
.venv/bin/python run_inference.py --stage security --model small \
  --family month-12-v3-0 --output inference-small-example
```

For complete stages, omit `--family`:

```bash
.venv/bin/python run_inference.py --stage security --model small --output inference-small-security
.venv/bin/python run_inference.py --stage utility --model small --output inference-small-utility
.venv/bin/python run_inference.py --stage sensitivity --model small --output inference-small-sensitivity
.venv/bin/python run_inference.py --stage scifact --model small --output inference-small-scifact
.venv/bin/python run_inference.py --stage cost --model small --output inference-small-cost
```

Use `--model base` and new output directories for the base model. `--cache PATH` selects the model/vector cache. Inference uses CPU float32, pinned model revisions and the corresponding tokenizer. The first run downloads the models and executes their pinned custom implementation. The SciFact stage downloads the public BEIR distribution and verifies its declared checksum.

The security stage evaluates both query forms. `data/query-rule.json` records the selected form, verified against 480 development observations. Rerunning the released corpus is not a new blinded held-out evaluation.

Measured fresh-process peak resident memory was 5.27 GB for the small model and 7.83 GB for the base model, with contexts up to 1,024 tokens. Allow additional memory and storage for dependencies, checkpoints and caches. These measurements are not minimum requirements. Numerical scores and timings can vary across environments; analysis replay uses the supplied observations.

## Experimental scope

The corpus contains four counterfactual worlds for each of 240 families. Training, development and testing use disjoint structural designs: 8/4/4 designs and 120/60/60 families. Cities and topics have disjoint candidate vocabularies; months use a fixed vocabulary. Public text, candidate order and distractors are coupled across each family's worlds.

The correction changes the registration-deadline sentence in design 15. Its ledger records all 60 original and corrected sentences. Targets, attributes, candidates, permissions, relevance judgments and splits are unchanged. The correction followed development outcomes and preceded held-out prediction. Independent human semantic review and external preregistration were not performed.

E encodes independent chunks; L uses full context; P uses contiguous equal-ACL partitions; A uses the authorized view. S returns scored top-10 results, R returns ranks without scores, and O is a separate direct-similarity diagnostic. L0 holds hidden context constant in development. Search results are filtered to authorized chunks before ranking.

Descriptive 95% intervals use 2,000 paired resamples of structural designs and public instances. Four held-out designs support the security and controlled-utility intervals. The equal-token-length subset has 20 month families. Development sensitivity and all-public SciFact do not establish security generalization to other domains.

## Files and provenance

`DATA.md` defines observation schemas and provenance. `src/` contains the algorithms, `data/` contains corpora and compressed observations, and `config.json` records settings and seeds. `MANIFEST.json` verifies distributed files. `data/provenance.json` records original observation checksums and implementation hashes. `CITATION.cff` provides author and citation metadata.

## Licensing

No license is currently assigned to the authors' code, synthetic data or documentation. Included SciFact judgments retain their attribution and license in `SCIFACT-LICENSE.md`. Model checkpoints and the SciFact document corpus are obtained separately under their upstream terms.
