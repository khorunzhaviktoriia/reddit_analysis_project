# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A coursework data-analysis project: how the stance toward Donald Trump expressed on Reddit changes between his first term (Jan-Jun 2017 and 2018) and second term (Jan-Jun 2025 and 2026), compared with Pew and YouGov approval surveys. The deliverables are Jupyter notebooks read as reports (the assignment brief is in `assessments/`, e.g. the EDA notebook must be self-contained, mostly visual, and exported to PDF/HTML). There is no package, no test suite and no linter.

## Environment and commands

Windows machine. Always use the project virtual environment, never system Python:

```
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts\<script>.py
.venv\Scripts\jupyter.exe lab                                         # notebooks
.venv\Scripts\jupyter.exe nbconvert --to html notebooks\EDA\eda.ipynb --output eda_report
```

The brief asks for the EDA as a PDF. `notebooks/EDA/eda_report.html` is an untracked export that includes the code cells.

- `requirements.txt` is a `pip freeze` saved as UTF-16 LE (PowerShell redirect); `grep` sees it as binary. If regenerating it, write UTF-8. It does not list `pyreadstat` (needed by `combine_pew_surveys.py`) or `openpyxl` (needed by `yougov_preprocessing.ipynb`).
- GPU scripts: `infer_stance.py` refuses to run without CUDA; `--smoke` classifies 2,000 comments and writes nothing. `label_stance_part{1,2}.py` support `--dry-run` (no GPU) and `--limit N`, and are resumable after Ctrl+C.

## Data layout and pipeline

`data/` and `assessments/` are gitignored, and the work is split across two computers, so any `data/` subdirectory (raw inputs, `data/stance/`, `data/models/`, ...) may be missing on a given machine. Check before assuming that a path exists. Scripts locate the root via `Path(__file__).parent.parent`; notebooks walk up from `cwd` until they find a `data/` directory, so they run from any subfolder.

Reddit pipeline (stages in order):
1. `scripts/extract_direct_trump_mentions.py`: `data/filtered/RS_/RC_<month>.csv` -> `data/processed/trump_discourse/*_trump.csv`. Keeps comments whose own body matches `\btrump\b`, and submissions that match in title/selftext or contain such a comment (`submission_mentions_trump` / `has_trump_comment`). The hardcoded `MONTHS` list covers the first term only; the second term was processed with it changed.
2. `scripts/combine_reddit_data.py`: monthly CSVs -> `data/processed/combined/{comments,submissions}_trump.parquet`. The per-term outputs were kept as `*_trump_2017_2018.parquet` / `*_trump_2025_2026.parquet`.
3. `scripts/combine_reddit_periods.py`: concatenates the two terms (coercing mixed-type object columns for Parquet) -> `data/processed/combined/*_trump.parquet`.
4. `notebooks/preprocessing/reddit_cleaning.ipynb` -> `data/cleaned/{submissions,comments}_clean.parquet` (adds `body_clean`, `period`, `term`, `year`, flags such as `trump_only_in_quote`, `author_deleted`). Validated by `notebooks/supporting_analysis/reddit_cleaning_validation.ipynb`.
5. Stance labelling (see below) appends `student_label`, `student_p_against|neutral|pro` to `comments_clean.parquet`. Re-running `reddit_cleaning.ipynb` overwrites the table without these columns, and restoring them means hours of GPU inference.

There are two unrelated stance fields. `stance` (supporter / undecided / nonsupporter / other) is the declared user flair, set by the cleaning notebook and only non-null in r/AskTrumpSupporters. `student_label` (against / neutral / pro) is the model's prediction of the stance a comment's text takes toward Trump.

Survey pipeline:
- Pew: `scripts/combine_pew_surveys.py --zips <dir> --out <dir>` reads `.sav` files from the original Pew zips using the per-wave `CONFIG` crosswalk (question names, date formats and weights differ per wave; derived in `notebooks/supporting_analysis/pew_raw_exploration.ipynb`) -> `pew_surveys.csv`, then `notebooks/preprocessing/pew_preprocessing.ipynb` -> `data/cleaned/pew_surveys_clean.csv`. `pew_2026_waves.csv` holds 2026 topline data.
- YouGov: `notebooks/preprocessing/yougov_preprocessing.ipynb`: `data/raw/yougov/*.xlsx` -> `data/cleaned/yougov_approval_clean.csv`.

## Stance model (teacher -> student distillation)

The labelling method is chosen and justified in `notebooks/EDA/model_selection.ipynb`, which writes to `data/sentiment/` (section 1, off-the-shelf models vs manual labels) and `data/stance/` (sections 2-5):
1. Teacher: local Qwen3.5-4B GGUF served by llama.cpp `llama-server`. `scripts/label_stance_part1.py` / `part2.py` label a 100k-comment sample split across two computers; the files are identical except for `PART`, so change both together. The prompt is copied verbatim from section 2.2 of the notebook and guarded by a SHA-256 assert: changing the prompt in one place requires changing it in the other and updating `PROMPT_SHA256`. llama.cpp is downloaded into `.venv/llama.cpp/`, the model into `data/models/`.
2. Student: DistilRoBERTa fine-tuned on the teacher labels in section 5 (saved to `data/stance/student/distilroberta_mild`).
3. `scripts/infer_stance.py` runs the student over all comments, one row group at a time, saving resumable parts in `data/stance/student/inference_parts/`. It then atomically replaces `comments_clean.parquet` and keeps a backup. Comments where Trump appears only in a quote get no label. The script exits if `student_label` already exists, so to re-infer, start from the backup or from a freshly cleaned table.

Classes are `against` / `neutral` / `pro` (the section 1 models use `favor` instead of `pro`).

## Analysis notebook

`notebooks/EDA/eda.ipynb` is the main report (sections: basic exploration, meaning of stance labels, stance change, whose voice, vocabulary/topics, Pew, Reddit vs Pew vs YouGov, synthesis). Its setup cells define `PERIODS = T1_Y1, T1_Y2, T2_Y1, T2_Y2`, the six-subreddit order (`politics`, `AskTrumpSupporters`, `Conservative`, `PoliticalDiscussion`, `democrats`, `Republican`), colour dictionaries and matplotlib rcParams. There is no shared module: `model_selection.ipynb` keeps its own copy of these constants, and its copy has already drifted (its stance colours are different hex values). Inside a notebook, reuse its existing constants rather than defining new ones. `comments_clean.parquet` is ~4.5M rows; the notebook reads it without the text columns and streams `body_clean` per row group to stay within memory.

The author edits `eda.ipynb` by hand: change only the cells asked for, never regenerate or reformat the whole notebook. Notebook prose and any written output are in English, with no emojis.