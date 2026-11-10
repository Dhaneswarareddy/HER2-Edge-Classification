# Instructions for AI agents working in this repository

Read this before opening any other file. This is a final-year research project on HER2 IHC scoring (classes
0/1+/2+/3+): training on the BCI dataset, testing on a second laboratory's data (HER2-IHC-40x), and deployment on
Android phones and a Raspberry Pi 4B.

## Read in this order

1. `PROJECT_STATUS_AND_ROADMAP.md`: every experiment in pipeline order, its result, and where the evidence is.
   - §2: the semester split. **This semester = Steps 1–14** (data, model development, in-domain evaluation).
     **Next semester = Steps 15 onward** (other laboratory, deployment, phone power, E6/E7, future work).
   - §5: the must-disclose list. Every report or presentation must include these points.
   - §7: a slide plan for this semester, with the figure file for each slide.
2. `FYP3_FINAL_REPORT.md` (corrected 10 Nov 2026): full method and results for all completed steps.
3. `experiments_v2/<nn>_<name>/README.md`: details of the extended experiments.
4. `experiments_v2/REPORT_CORRECTIONS_v2.md`: what earlier versions got wrong. Use it to avoid repeating old errors.

## Rules for numbers

- **Quote numbers only from the files above or from primary result files** (`results/runs/*/summary.json`,
  `final_reports/tables/*.csv`, `experiments_v2/*/outputs/*`). Never estimate, round up or "fill in" a number.
- **Do not use these files as sources** (where present). They are historical logs with superseded numbers:
  `ANTIGRAVITY_*`, `CLAUDE_*.txt`, `FYP3_FINAL_AUDIT.md`, `REPORT_CORRECTIONS.md` (old), `results/REPORT*.md`,
  `results/metrics_all.csv`, `results/selection.json`, `deployment/benchmark_results/*.csv` without a `2026-11-10` date,
  and `final_report_figures/fig7_raspberry_pi4_hardware_benchmark.png` (superseded 31 Oct run).
- **Phone power and E6/E7 have no results yet** (`experiments_v2/10_power/`, `experiments_v2/09_resolution_e6_e7/`).
  Describe them only as next-semester work in progress.
- **Validation is the decision split.** BCI test numbers are reported once and were never used for selection.
  Validation figures are the fairer headline (primary model: val acc 0.950, QWK 0.935; test acc 0.975, QWK 0.959).
- **The external test split (HER2-IHC-40x, 1,847 tiles) was examined before.** Never call it untouched or held out;
  external results are exploratory.

## Wording to avoid

"Clinically validated", "clinically ready", "generalises", "real-time whole-slide", "zero throttling" on phones,
"energy-efficient" (no power result exists yet), "synthetic generalisation is a myth". This is research software and
not a medical device.

## Practical notes

- Figures for the report: `final_report_figures/` (index and data sources in its `README.md`) and
  `experiments_v2/*/outputs/*.png`.
- `fig_robustness.png` plots macro-F1; the robustness numbers quoted in the text are balanced accuracy.
- Do not modify result files, checkpoints, splits or manifests.
