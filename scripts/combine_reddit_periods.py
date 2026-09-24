import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed" / "combined"

comments_17_18 = pd.read_parquet(DATA_DIR / "comments_trump_2017_2018.parquet")
comments_25_26 = pd.read_parquet(DATA_DIR / "comments_trump_2025_2026.parquet")
submissions_17_18 = pd.read_parquet(DATA_DIR / "submissions_trump_2017_2018.parquet")
submissions_25_26 = pd.read_parquet(DATA_DIR / "submissions_trump_2025_2026.parquet")

comments = pd.concat(
    [comments_17_18, comments_25_26],
    ignore_index=True,
    sort=False
)

submissions = pd.concat(
    [submissions_17_18, submissions_25_26],
    ignore_index=True,
    sort=False
)

def prepare_for_parquet(df, dataset_name):
    df = df.copy()

    print(f"\nChecking mixed-type columns in {dataset_name}...")

    for column in df.select_dtypes(include="object").columns:
        non_null = df[column].dropna()

        if non_null.empty:
            continue

        value_types = non_null.map(type).unique()

        if len(value_types) <= 1:
            continue

        type_names = [t.__name__ for t in value_types]

        print(f"{column}: mixed types {type_names}")

        normalized_values = set(
            non_null.astype(str)
            .str.strip()
            .str.lower()
            .unique()
        )

        if normalized_values <= {"true", "false"}:
            def to_boolean(value):
                if value is None or value is pd.NA:
                    return pd.NA

                if isinstance(value, float) and pd.isna(value):
                    return pd.NA

                value = str(value).strip().lower()

                if value == "true":
                    return True

                if value == "false":
                    return False

                return pd.NA

            df[column] = df[column].map(to_boolean).astype("boolean")

            print("  -> converted to boolean")

        else:
            df[column] = df[column].astype("string")
            print("  -> converted to string")
    return df


comments = prepare_for_parquet(comments,"comments")
submissions = prepare_for_parquet(submissions,"submissions")

comments_output = DATA_DIR / "comments_trump.parquet"
submissions_output = DATA_DIR / "submissions_trump.parquet"

comments.to_parquet(comments_output,index=False)

submissions.to_parquet(submissions_output,index=False)

print("DONE")

print("Comments:", comments.shape)
print("Submissions:", submissions.shape)

print("Comment months:",comments["source_month"].nunique())
print("Submission months:",submissions["source_month"].nunique())

print("\nSaved:")
print(comments_output)
print(submissions_output)