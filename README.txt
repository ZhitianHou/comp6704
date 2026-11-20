COMP6704 Individual Assignment — LASSO study
HOU Zhitian, 26017658r
Reviewed: 2026-11-20

STATUS
Formal study complete: 18 data problems, 54 problem/method combinations,
162 timed solves (3 timing repeats per combination). All reached the fixed
relative duality-gap tolerance 1e-6. No new timing study was run during review.
The reviewed numerical code reproduces all 54 coefficient vectors bitwise.

Read research_plan.html for the model and original plan.
report/main.tex is the revised English paper. It discusses modern solver work,
derives the dual certificate, and separates numerical from statistical error.
Code repository: https://github.com/ZhitianHou/comp6704
The paper links to this repository after the abstract. Canvas submission is
separate and is not performed by these scripts.

INSTALL (Python 3.13; separate environment)
  python3.13 -m venv .venv
  .venv/bin/python -m pip install -r requirements-lock.txt
Pinned versions match the local run; numerical kernels/OS can affect timing.

REPRODUCE
  .venv/bin/python run_formal.py
Runs write a fresh results/formal_* directory without overwriting old logs.
Parameters come from formal_protocol.json. The default frozen protocol is used
for the submitted results. Study counts must agree with configured dimensions.
The implementation supports float64, every-iteration certificates, and a
requested native thread count of one. Unsupported settings fail explicitly.
All optimization loops start at zero; no warm starts are used.

EXPORT SAVED RESULTS (no solving)
  .venv/bin/python export_formal.py
  .venv/bin/python export_review_analysis.py
For a custom run directory:
  .venv/bin/python export_formal.py --run /absolute/path/to/saved_run
  .venv/bin/python verify_formal.py --run /absolute/path/to/saved_run
The exporter rejects incomplete/nonconverged studies instead of summarizing
iteration-capped runtime as time-to-tolerance. Failure records remain in raw logs.
The formal paper, plots and audit are scoped to the frozen study. Changing the
protocol requires reviewing the fixed figure layout and narrative rather than
assuming that old prose automatically describes a new experiment.

CHECKS
  .venv/bin/python verify_formal.py
  .venv/bin/python review_checks.py
verify_formal audits source snapshots, 162 records, split reconstruction and,
when present locally, exact processed arrays and saved-solution certificates.
The repository excludes .npz arrays; an audit without them is explicitly partial.
review_checks compares unequal-scale designs with sklearn and an independent
nonnegative split-variable quadratic program; it covers rank deficiency, zero
columns, iteration caps, invalid inputs, and export refusal on nonconvergence.
Its 54-case numerical replay needs the locally retained processed .npz arrays.
Those arrays can be regenerated with run_formal.py after cloning the repository.
Replay results validate coefficients; fresh wall times are not compared with old
timings or merged into formal seed statistics.

BUILD PAPER (installed TeX distribution)
  cd report
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
No attachment or external data cache is needed to compile the paper.
The paper uses frozen outputs via resultdir/ablationdir/realdir macros.
A new study requires updating both those macros and the numerical narrative.

LOCAL DELIVERY VALIDATION (from the study root)
  .venv/bin/python package_delivery.py
  .venv/bin/python check_delivery.py
These optional local handoff tools verify checksums, regenerate saved tables
without solving, and compile the paper in a separate temporary directory.
The public repository and its README are the code entry cited in the report.

DATA AND TIMING
Synthetic: n_train=n_test=1000, p=200, 20 signed nonzero coefficients,
noise SD=.5, AR(1) correlation 0/.5/.9, seeds 1..5. Matched latent draws,
support/signs/noise across correlation levels within each seed.
Diabetes: sklearn-supplied raw data, 309/66/67 train/reserved-validation/test,
split seeds 1..3, training-only preprocessing. No parameter selection uses tests.
lambda=.05*lambda_max, ADMM rho=1, max 20,000 outer iterations.
Primary timing: method-specific preparation plus updates and every-iteration
certificate/history work; shared preprocessing, reference fits and disk export
are excluded. Each problem uses median of 3 timing repetitions, then mean and
sample SD across 5 synthetic seeds or 3 overlapping real-data splits. Timing
repeats are not extra independent data samples. Native libraries are requested
to use one thread; threadpoolctl cannot fully enumerate Apple Accelerate.

PROVENANCE
solvers.py implements known ISTA/CD/ADMM updates, with NumPy/SciPy kernels.
sklearn is an accuracy reference, not a fourth timed method. No novel optimizer
is claimed. The formal result directory preserves its exact original source.
Reviewed root files add validation/config/portability fixes; original raw timings
refer to the frozen source, not the subsequently reviewed runtime overhead.
CODE_REVIEW.txt and output/review_checks.json describe review scope and evidence.
A pre-study launcher serialization failure produced no timed formal data; its
local directory is excluded from the delivery and formal statistics.

OPTIONAL PILOT
  .venv/bin/python run_pilot.py
  .venv/bin/python render_pilot_report.py
Pilot runs and results are distinct from formal data; these commands create the
pilot pointer and files if starting after cloning the repository.

POST-HOC SUPPLEMENTARY EXPERIMENTS (2026-11-20)
  .venv/bin/python run_ablations.py
  .venv/bin/python export_ablations.py
Separate ablation_protocol.json is fixed before the supplementary run. It varies
certificate interval q=1/10 on c=0/.9 and ADMM rho=.1/1/10 on c=.9. Five original
training data seeds and three timing repeats give 210 unique timed solves, with
the rho=1/q=1 baseline shared across analyses. All comparison baselines are from
the same supplementary session; old primary timings are not mixed with these.
Changing q alters only checking/termination opportunities; the final gap criterion
is identical. Check at iteration zero and at the cap is always performed.
The experiments are explicitly post-hoc sensitivity analyses, not new data seeds,
parameter selection, or a replacement of primary results. All original logs stay
unchanged. Saved supplementary coefficients are local; the repository
includes raw iteration/timing logs and source, with data regenerated from seeds.

ADDITIONAL REAL DATASET (2026-11-20)
  .venv/bin/python data_superconductivity.py --download
  .venv/bin/python run_realdata.py
  .venv/bin/python export_realdata.py
  .venv/bin/python verify_realdata.py
The first command downloads the official UCI id=464 ZIP (network required only
when absent), verifying pinned ZIP and train.csv SHA-256 values. The raw CSV has
21,263 rows and 81 engineered predictors; critical_temp is the target. Only
train.csv is used. Raw data are CC BY 4.0: Hamidieh (2018), UCI dataset DOI
10.24432/C53P47. The cache is excluded from the repository.

15,170 unique predictor vectors define label-independent groups. All original
rows are retained. Split seeds 1/2/3 shuffle groups, allocating 70/15/15 percent
to training/reserved validation/test, with no identical descriptor group crossing
sets. Training counts: 15064/14897/14819; test counts: 3129/3129/3150. Training-only
centering/scaling is saved with exact row indices. This prevents descriptor
replication across sets but is not a chemical-family generalization split.
The fixed original lambda fraction, rho, tolerance and checking policy are kept.
Three algorithms x three splits x three timing repetitions give 27 solves, all
certified. Median timings are summarized across splits with sample SD. MSE and
R-squared are prediction checks; we claim no materials-science prediction novelty.

The report now covers 162 original, 210 sensitivity and 27 additional-real-data
solves, kept in separate frozen result directories. latest_realdata.txt points
to the added run. The report source uses its realdir macro; source and raw logs
are packaged along with checksum/provenance and split manifests. The repository
excludes raw downloads and .npz coefficient files. A full local audit
checks them; a fresh download reproduces data and group splitting.
