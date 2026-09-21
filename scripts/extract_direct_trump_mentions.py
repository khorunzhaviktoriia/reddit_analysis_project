from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "data" / "filtered"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "trump_discourse"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MONTHS = [
    "2017-01",
    "2017-02",
    "2017-03",
    "2017-04",
    "2017-05",
    "2017-06",
    "2018-01",
    "2018-02",
    "2018-03",
    "2018-04",
    "2018-05",
    "2018-06",
]

CHUNK_SIZE = 200_000

TRUMP_PATTERN = r"\btrump\b"

def contains_trump(series):
    return series.fillna("").astype("string").str.contains(TRUMP_PATTERN,case=False,regex=True,na=False)

def normalize_submission_id(series):
    # У comments: link_id = t3_5abcde
    # У submissions: id = 5abcde

    return (series.astype("string").str.replace(r"^t3_","",regex=True))

def require_columns(df, required_columns, filename):
    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"{filename}: missing required columns: {missing}")


# load all submissions

print("=" * 80)
print("STEP 1: LOADING SUBMISSIONS")
print("=" * 80)

submission_frames = []

for month in MONTHS:
    path = INPUT_DIR / f"RS_{month}.csv"

    if not path.exists():
        raise FileNotFoundError(f"Submission file not found:\n{path}")

    print(f"Reading {path.name}...")

    df = pd.read_csv(path,low_memory=False)

    require_columns(
        df,
        [
            "id",
            "title",
            "selftext",
            "subreddit",
        ],
        path.name
    )

    df["source_month"] = month
    submission_frames.append(df)

submissions = pd.concat(submission_frames,ignore_index=True)
print(f"\nTotal submissions loaded: {len(submissions):,}")


# deduplicate submissions

print("\n" + "=" * 80)
print("STEP 2: DEDUPLICATING SUBMISSIONS")
print("=" * 80)

before = len(submissions)

submissions["id"] = submissions["id"].astype("string")

submissions = submissions.drop_duplicates(subset="id",keep="first").copy()

submission_duplicates_dropped = before - len(submissions)

print(f"Unique submissions: {len(submissions):,}")
print(f"Duplicate submissions dropped: {submission_duplicates_dropped:,}")


# identify direct trump submissions

print("\n" + "=" * 80)
print("STEP 3: IDENTIFYING DIRECT TRUMP SUBMISSIONS")
print("=" * 80)

title_mentions = contains_trump(submissions["title"])

selftext_mentions = contains_trump(submissions["selftext"])

submissions["submission_mentions_trump"] = (title_mentions | selftext_mentions)

direct_trump_submission_count = int(submissions["submission_mentions_trump"].sum())

print(f"Direct Trump submissions: {direct_trump_submission_count:,}")
print(f"Share of all submissions: {direct_trump_submission_count / len(submissions):.2%}")


# process comments

print("\n" + "=" * 80)
print("STEP 4: FILTERING COMMENTS")
print("=" * 80)
print("comment is retained ONLY if its own body explicitly mentions 'Trump'")

submission_ids_with_trump_comments = set()

summary_rows = []

for month in MONTHS:
    comments_path = INPUT_DIR / f"RC_{month}.csv"

    output_comments_path = OUTPUT_DIR / f"RC_{month}_trump.csv"

    if not comments_path.exists():
        raise FileNotFoundError(f"Comments file not found:\n{comments_path}")

    print("\n" + "-" * 80)
    print(f"Processing comments: {month}")
    print("-" * 80)

    if output_comments_path.exists():
        output_comments_path.unlink()

    total_comments = 0
    retained_comments = 0
    duplicate_comments_dropped = 0

    first_output_chunk = True

    seen_comment_ids = set()

    original_columns = None

    reader = pd.read_csv(comments_path,chunksize=CHUNK_SIZE,low_memory=False)

    for chunk_number, comments in enumerate(reader,start=1):
        if original_columns is None:
            original_columns = list(comments.columns)

        require_columns(
            comments,
            [
                "body",
                "link_id",
            ],
            comments_path.name
        )

        total_comments += len(comments)

        mask = contains_trump(comments["body"])

        comments_trump = comments.loc[mask].copy()

        if not comments_trump.empty and "id" in comments_trump.columns:
            before = len(comments_trump)

            comments_trump = comments_trump.drop_duplicates(subset="id",keep="first")
            duplicate_comments_dropped += (before - len(comments_trump))
            comment_ids = (comments_trump["id"].astype("string"))

            already_seen = comment_ids.isin(seen_comment_ids)
            duplicate_comments_dropped += int(already_seen.sum())

            comments_trump = comments_trump.loc[~already_seen].copy()

            new_comment_ids = comments_trump["id"].astype("string").dropna().tolist()

            seen_comment_ids.update(new_comment_ids)

        if not comments_trump.empty:
            comments_trump["submission_id"] = normalize_submission_id(comments_trump["link_id"])
            comments_trump["source_month"] = month
            comments_trump["mentions_trump"] = True

            parent_ids = comments_trump["submission_id"].dropna().tolist()

            submission_ids_with_trump_comments.update(parent_ids)


        if not comments_trump.empty:
            comments_trump.to_csv(
                output_comments_path,
                mode=(
                    "w"
                    if first_output_chunk
                    else "a"
                ),
                header=first_output_chunk,
                index=False,
                encoding="utf-8"
            )

            first_output_chunk = False

        retained_comments += len(comments_trump)

        print(f"Chunk {chunk_number:>3} | processed: {total_comments:,} | retained: {retained_comments:,}")


    if first_output_chunk:
        empty_columns = (
            original_columns.copy()
            if original_columns is not None
            else []
        )

        extra_columns = [
            "submission_id",
            "source_month",
            "mentions_trump",
        ]

        for column in extra_columns:
            if column not in empty_columns:
                empty_columns.append(column)

        pd.DataFrame(columns=empty_columns).to_csv(output_comments_path,index=False,encoding="utf-8")

    retention_rate = (
        retained_comments / total_comments
        if total_comments
        else 0
    )

    summary_rows.append({
        "month": month,
        "all_comments": total_comments,
        "trump_comments": retained_comments,
        "comment_retention_rate": retention_rate,
        "duplicate_comments_dropped": duplicate_comments_dropped,
    })

    print("\nSummary for", month)
    print(f"All comments: {total_comments:,}")
    print(f"Comments directly mentioning Trump: {retained_comments:,}")
    print(f"Retention rate: {retention_rate:.2%}")
    print(f"Duplicate comment IDs dropped: {duplicate_comments_dropped:,}")
    print(f"Saved -> {output_comments_path.name}")


# identify submissions that have direct-trump comments

print("\n" + "=" * 80)
print("STEP 5: FINDING PARENT SUBMISSIONS")
print("=" * 80)

print(f"Unique parent submission IDs found in direct-Trump comments: {len(submission_ids_with_trump_comments):,}")

submissions["has_trump_comment"] = submissions["id"].isin(submission_ids_with_trump_comments)


# build final relevant submission dataset

print("\n" + "=" * 80)
print("STEP 6: BUILDING RELEVANT SUBMISSION DATASET")
print("=" * 80)

relevant_submission_mask = (submissions["submission_mentions_trump"] | submissions["has_trump_comment"])

relevant_submissions = submissions.loc[relevant_submission_mask].copy()

print(f"All submissions: {len(submissions):,}")
print(f"Relevant submissions: {len(relevant_submissions):,}")


# submission type statistics

direct_only_mask = (
    relevant_submissions["submission_mentions_trump"]
    &
    ~relevant_submissions["has_trump_comment"]
)

comment_only_mask = (
    ~relevant_submissions["submission_mentions_trump"]
    &
    relevant_submissions["has_trump_comment"]
)

both_mask = (
    relevant_submissions["submission_mentions_trump"]
    &
    relevant_submissions["has_trump_comment"]
)


direct_only_count = int(direct_only_mask.sum())
comment_only_count = int(comment_only_mask.sum())
both_count = int(both_mask.sum())

print("\nRelevant submission types:")

print(f"Direct Trump mention only: {direct_only_count:,}")
print(f"Trump appears only in comments: {comment_only_count:,}")
print(f"Both direct submission mention and Trump comment: {both_count:,}")


# save relevant submissions

print("\n" + "=" * 80)
print("STEP 8: SAVING RELEVANT SUBMISSIONS")
print("=" * 80)

for month in MONTHS:
    month_submissions = (
        relevant_submissions[
            relevant_submissions["source_month"]
            == month
        ]
        .copy()
    )

    output_path = (OUTPUT_DIR / f"RS_{month}_trump.csv")

    month_submissions.to_csv(output_path,index=False,encoding="utf-8")

    print(f"{month}: saved {len(month_submissions):,} submissions -> {output_path.name}")


# add submission statistics to month summary

summary = pd.DataFrame(summary_rows)
submission_summary_rows = []

for month in MONTHS:
    month_all = submissions[submissions["source_month"] == month]

    month_relevant = relevant_submissions[relevant_submissions["source_month"] == month]

    month_direct = int(month_all["submission_mentions_trump"].sum())

    month_has_trump_comment = int(month_all["has_trump_comment"].sum())

    month_comment_only = int(
        (
            ~month_all["submission_mentions_trump"]
            &
            month_all["has_trump_comment"]
        ).sum()
    )

    submission_summary_rows.append({
        "month": month,
        "all_submissions": len(month_all),
        "relevant_submissions": len(month_relevant),
        "direct_trump_submissions": month_direct,
        "submissions_with_trump_comment": month_has_trump_comment,
        "comment_only_submissions": month_comment_only,
    })


submission_summary = pd.DataFrame(submission_summary_rows)

summary = summary.merge(
    submission_summary,
    on="month",
    how="left"
)

# save summary

summary_path = OUTPUT_DIR / "filtering_summary.csv"

summary.to_csv(summary_path,index=False,encoding="utf-8")

print("\n" + "=" * 80)
print("FILTERING COMPLETE")
print("=" * 80)
print("\nSummary:")
print(summary.to_string(index=False))
print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"\nSummary saved to: {summary_path}")