from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "data" / "processed" / "trump_discourse"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "combined"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

comment_files = sorted(INPUT_DIR.glob("RC_????-??_trump.csv"))
submission_files = sorted(INPUT_DIR.glob("RS_????-??_trump.csv"))

if not comment_files:
    raise FileNotFoundError(f"No comment files found in {INPUT_DIR}")

if not submission_files:
    raise FileNotFoundError(f"No submission files found in {INPUT_DIR}")


print(f"Comment files: {len(comment_files)}")
print(f"Submission files: {len(submission_files)}\n")

comment_frames = []

for path in comment_files:
    print(f"Reading {path.name}")
    df = pd.read_csv(path, low_memory=False)
    comment_frames.append(df)

comments = pd.concat(comment_frames,ignore_index=True)

print(f"\nCombined comments: {len(comments):,}")
print(f"Columns: {len(comments.columns)}\n")

submission_frames = []

for path in submission_files:
    print(f"Reading {path.name}")
    df = pd.read_csv(path, low_memory=False)
    submission_frames.append(df)

submissions = pd.concat(submission_frames,ignore_index=True)

print(f"\nCombined submissions: {len(submissions):,}")
print(f"Columns: {len(submissions.columns)}")

comments_output = OUTPUT_DIR / "comments_trump.parquet"
submissions_output = OUTPUT_DIR / "submissions_trump.parquet"

comments.to_parquet(comments_output,index=False)
submissions.to_parquet(submissions_output,index=False)

comments_size_mb = comments_output.stat().st_size / (1024 ** 2)
submissions_size_mb = submissions_output.stat().st_size / (1024 ** 2)

print("\nDone!")

print(
    f"\nComments:\n"
    f"  rows: {len(comments):,}\n"
    f"  size: {comments_size_mb:,.2f} MB\n"
    f"  saved: {comments_output}"
)

print(
    f"\nSubmissions:\n"
    f"  rows: {len(submissions):,}\n"
    f"  size: {submissions_size_mb:,.2f} MB\n"
    f"  saved: {submissions_output}"
)