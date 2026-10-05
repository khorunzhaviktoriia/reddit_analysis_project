import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
COMMENTS_PATH = ROOT / "data" / "cleaned" / "comments_clean.parquet"
STUDENT_DIR = ROOT / "data" / "stance" / "student"
MODEL_DIR = STUDENT_DIR / "distilroberta_mild"
PARTS_DIR = STUDENT_DIR / "inference_parts"
CONFIG_PATH = STUDENT_DIR / "inference_config.json"

CLASSES = ["against", "neutral", "pro"]
NEW_COLUMNS = ["student_label", "student_p_against", "student_p_neutral", "student_p_pro"]


def load_model(device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()
    assert {int(k): v for k, v in model.config.id2label.items()} == dict(enumerate(CLASSES)), model.config.id2label
    return tokenizer, model


@torch.inference_mode()
def probabilities(tokenizer, model, texts, max_length, batch_size, device):
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = np.zeros((len(texts), 3), dtype=np.float32)

    def run(idx):
        encoded = tokenizer([texts[j] for j in idx], truncation=True, max_length=max_length, padding=True,
                            return_tensors="pt").to(device)
        try:
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(**encoded).logits
        except torch.cuda.OutOfMemoryError:
            if len(idx) == 1:
                raise
            torch.cuda.empty_cache()
            run(idx[:len(idx) // 2])
            run(idx[len(idx) // 2:])
            return
        out[idx] = torch.softmax(logits.float(), dim=-1).cpu().numpy()

    for start in range(0, len(texts), batch_size):
        run(order[start:start + batch_size])
    return out


def classify_row_group(pf, i, tokenizer, model, device, max_length, batch_size, limit=None):
    table = pf.read_row_group(i, columns=["id", "body_clean", "trump_only_in_quote"])
    n = table.num_rows
    only_in_quote = table["trump_only_in_quote"].to_numpy(zero_copy_only=False).astype(bool)
    rows = np.flatnonzero(~only_in_quote)
    if limit:
        rows = rows[:limit]
    body = table["body_clean"].to_pylist()

    probs = np.full((n, 3), np.nan, dtype=np.float32)
    probs[rows] = probabilities(tokenizer, model, [body[j] or "" for j in rows], max_length, batch_size, device)
    labels = np.array(CLASSES, dtype=object)[np.nan_to_num(probs).argmax(axis=1)]
    labels[np.isnan(probs[:, 0])] = None

    return pa.table({
        "id": table["id"],
        "student_label": pa.array(labels, type=pa.string()),
        **{col: pa.array(probs[:, k], type=pa.float32(), from_pandas=True) for k, col in enumerate(NEW_COLUMNS[1:])},
    })


def part_path(i):
    return PARTS_DIR / f"row_group_{i:04d}.parquet"


def join_parts(pf):
    new_path = COMMENTS_PATH.with_name(COMMENTS_PATH.name + ".new")
    backup_path = COMMENTS_PATH.with_name(COMMENTS_PATH.name + ".bak")
    compression = pf.metadata.row_group(0).column(0).compression.lower()
    writer, n_labelled = None, 0
    try:
        for i in range(pf.metadata.num_row_groups):
            old, part = pf.read_row_group(i), pq.read_table(part_path(i))
            assert old.num_rows == part.num_rows and old["id"].equals(part["id"]), f"row group {i}: the parts do not match the table"
            joined = old
            for col in NEW_COLUMNS:
                joined = joined.append_column(col, part[col])
            if writer is None:
                writer = pq.ParquetWriter(new_path, joined.schema, compression=compression)
            writer.write_table(joined, row_group_size=joined.num_rows)      # the row groups stay as they were
            n_labelled += joined.num_rows - part["student_label"].null_count
    finally:
        if writer is not None:
            writer.close()

    check = pq.ParquetFile(new_path)
    assert check.metadata.num_rows == pf.metadata.num_rows and check.metadata.num_row_groups == pf.metadata.num_row_groups
    assert check.schema_arrow.names == pf.schema_arrow.names + NEW_COLUMNS
    print(f"\njoined: {check.metadata.num_rows:,} rows, {n_labelled:,} with a stance, {check.metadata.num_rows - n_labelled:,} without (Trump only in a quote)")
    try:
        os.replace(COMMENTS_PATH, backup_path)
        os.replace(new_path, COMMENTS_PATH)
    except PermissionError:
        raise SystemExit(f"Could not rename the files: close the notebooks that have {COMMENTS_PATH.name} open and start the script again "
                         f"(the finished parts are kept; the joined table is {new_path.name}).")
    print(f"{COMMENTS_PATH.name} has the new columns now; the old table is {backup_path.name} (delete it after a check)")
    return n_labelled


def main():
    parser = argparse.ArgumentParser(description="Add the student's stance to comments_clean.parquet.")
    parser.add_argument("--max-length", type=int, default=512, help="tokens of a comment that the student reads (default 512)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--smoke", action="store_true", help="classify 2,000 comments of the first row group, print the result, write nothing")
    args = parser.parse_args()
    assert 8 <= args.max_length <= 512

    pf = pq.ParquetFile(COMMENTS_PATH)
    if NEW_COLUMNS[0] in pf.schema_arrow.names:
        raise SystemExit(f"{COMMENTS_PATH.name} already has the column {NEW_COLUMNS[0]}.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("No GPU found: on a CPU the table would take days.")
    tokenizer, model = load_model(device)
    n_groups, n_rows = pf.metadata.num_row_groups, pf.metadata.num_rows
    print(f"{n_rows:,} comments in {n_groups} row groups; {torch.cuda.get_device_name(0)}; max_length {args.max_length}")

    if args.smoke:
        start = time.perf_counter()
        part = classify_row_group(pf, 0, tokenizer, model, device, args.max_length, args.batch_size, limit=2000)
        seconds = time.perf_counter() - start
        labelled = part["student_label"].drop_null().to_pylist()
        shares = {c: f"{100 * labelled.count(c) / len(labelled):.1f}%" for c in CLASSES}
        print(f"{len(labelled):,} comments in {seconds:.1f} s = {len(labelled) / seconds:.0f} comments/s "
              f"(the whole table: about {n_rows / (len(labelled) / seconds) / 3600:.1f} h); shares {shares}")
        return

    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    settings = {"max_length": args.max_length, "model_dir": str(MODEL_DIR), "row_groups": n_groups, "rows": n_rows}
    if CONFIG_PATH.exists() and any(PARTS_DIR.iterdir()):
        earlier = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if {k: earlier.get(k) for k in settings} != settings:
            raise SystemExit(f"{PARTS_DIR} holds parts made with other settings {earlier}; delete the folder to start again.")
    CONFIG_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")

    start, done_rows = time.perf_counter(), 0
    try:
        for i in range(n_groups):
            path = part_path(i)
            if path.exists():
                assert pq.ParquetFile(path).metadata.num_rows == pf.metadata.row_group(i).num_rows, f"{path.name} has a wrong number of rows"
                continue
            part = classify_row_group(pf, i, tokenizer, model, device, args.max_length, args.batch_size)
            pq.write_table(part, path.with_name(path.name + ".tmp"))
            os.replace(path.with_name(path.name + ".tmp"), path)
            done_rows += part.num_rows
            speed = done_rows / (time.perf_counter() - start)
            left = n_rows - sum(pq.ParquetFile(part_path(k)).metadata.num_rows for k in range(n_groups) if part_path(k).exists())
            print(f"row group {i + 1:>2}/{n_groups} done  {speed:,.0f} rows/s  about {left / speed / 3600:.1f} h left (rows without Trump in their own text are skipped)", flush=True)
    except KeyboardInterrupt:
        raise SystemExit("\nstopped by the user; start the script again to continue")

    n_labelled = join_parts(pf)
    CONFIG_PATH.write_text(json.dumps({**settings, "labelled": n_labelled, "run_at": time.strftime("%Y-%m-%d %H:%M"),
                                       "torch": torch.__version__, "transformers": transformers.__version__}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()