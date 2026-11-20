# Sparse regression under correlated features

This repository compares ISTA, cyclic coordinate descent and ADMM on the same LASSO objective, using a shared primal-dual gap stopping criterion. It contains the implementation, frozen experiment configurations, original iteration logs, split manifests, generated tables and the paper source.

The study includes controlled synthetic problems, Diabetes and UCI Superconductivity, plus sensitivity experiments on certificate frequency and the ADMM penalty. The 162 original, 210 sensitivity and 27 added-real-data timed solves are separate experiment sets; timing repetitions are not independent datasets.

## Install

The recorded environment uses Python 3.13. Numerical libraries and hardware can change timings.

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
```

## Export the published results without training

```sh
.venv/bin/python export_formal.py
.venv/bin/python export_review_analysis.py
.venv/bin/python export_ablations.py
.venv/bin/python export_realdata.py
```

The `latest_*.txt` files select the frozen result directories. The exporters reject incomplete or nonconverged results.

## Run new experiments

```sh
.venv/bin/python run_formal.py
.venv/bin/python run_ablations.py
.venv/bin/python data_superconductivity.py --download
.venv/bin/python run_realdata.py
```

Runners create new result directories and update the corresponding latest pointer; they preserve the published logs. To export the original results after a fresh run, restore the published relative paths in `latest_formal.txt`, `latest_ablations.txt` and `latest_realdata.txt` before exporting. The formal exporter also accepts `--run results/<published-directory>`. Changing a protocol does not automatically update the paper's numerical claims.

Raw downloads, processed arrays, virtual environments and local draft revisions are excluded. Synthetic data regenerate from seeds; Diabetes is bundled with scikit-learn; the Superconductivity downloader verifies pinned checksums. The UCI data are CC BY 4.0, credited to Kam Hamidieh (2018), [DOI: 10.24432/C53P47](https://doi.org/10.24432/C53P47). Identical Superconductivity predictor vectors are grouped before splitting.

## Build the paper

With a TeX installation:

```sh
cd report
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The paper source points to the published output directories. The submission-ready PDF is [here](output/pdf/HOU%20Zhitian%2B26017658r.pdf).

## Implementation and validation

- `solvers.py`: ISTA, full-sweep cyclic CD, ADMM and the common primal-dual certificate.
- `study_common.py`: training-only preprocessing, matched synthetic data and basic correctness checks. Native thread requests are applied before numerical imports.
- `run_*.py`, `export_*.py`, `verify_*.py`: experiment runners, saved-result exports and audits for the primary, ablation and real-data studies.
- `review_checks.py`: independent numerical checks against scikit-learn and a split-variable quadratic program, including unequal feature scales, rank deficiency, zero columns and iteration caps.
- `check_repository.py`: source/raw-record/submission-PDF checksums, isolated regeneration of numerical tables and paper compilation. Generated plots and audit JSON files are excluded from strict hash checks; numerical tables must reproduce exactly. It does not rerun experiments.

```sh
.venv/bin/python review_checks.py
.venv/bin/python check_repository.py
# If TeX is unavailable:
.venv/bin/python check_repository.py --skip-tex
```

`review_checks.py` runs the independent numerical cases on a fresh clone. Its additional coefficient replay is explicitly skipped when processed arrays are absent. After regenerating a full primary run, use `review_checks.py --run results/<new-run>` to replay it. The original local study was checked across all saved problem/algorithm combinations; that historical full replay is not claimed for a clone without those arrays.

```sh
.venv/bin/python verify_formal.py
.venv/bin/python verify_ablations.py
# Requires the downloaded UCI data, but does not train:
.venv/bin/python verify_realdata.py
```

The auditors identify whether excluded coefficient arrays were available. The published audit JSON files record the original local audit, which included those arrays. Fresh audits without them have a narrower scope and are written to the ignored `output/validation/` directory, preserving the published audits. Exporting saved results or running these audits does not invalidate the repository check.

## Protocol and interpretation

The objective is `||Xw-y||²/(2n) + lambda*||w||1`. Methods start at zero in float64, use `lambda = 0.05*lambda_max`, and stop at relative primal-dual gap `1e-6`, with a cap of 20,000 outer iterations. The primary ADMM penalty is 1. Every primary iteration is certified; the ablation changes the checking interval and ADMM penalty without changing the target objective or final tolerance.

Synthetic data use 1,000 training and 1,000 test rows, 200 features, 20 nonzero coefficients, noise SD 0.5 and AR(1) correlations 0/0.5/0.9, with seeds 1–5. Random innovations, support, signs and noise are matched across correlations. Diabetes uses seeds 1–3 with 309/66/67 training/reserved-validation/test rows. Superconductivity retains every row but groups identical predictors before splitting; there are 15,170 distinct predictor groups. Preprocessing uses training observations only. Reserved validation data do not select parameters, and no test-based tuning is performed.

Timing includes method-specific preparation, updates, certificates and in-memory history bookkeeping. It excludes shared preprocessing, reference fits, test evaluation, disk output and plotting. Each problem uses the median of three timing repetitions, then means and sample SD across data seeds or splits. Requested native threads are one; Apple Accelerate's actual thread count is not fully enumerated by `threadpoolctl`. No GPU is used.

Original timing records and their `results/*/source/` snapshots are preserved. Those snapshots intentionally retain the original module names; active code uses `study_common.py`. Cleanup changes file organization and validation only, and does not replace published timing measurements. NumPy/SciPy supply numerical kernels; scikit-learn supplies reference fits, not a timed fourth competitor. CD uses Python coordinate loops, so the speed ranking should not be generalized to compiled production solvers or arbitrary matrix dimensions.
