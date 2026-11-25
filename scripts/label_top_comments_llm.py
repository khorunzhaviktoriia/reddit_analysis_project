"""Label the 500 hand-labelled top-level comments of `top_comments_eda.ipynb` with the local Qwen3.5-4B, twice:

    v1        the comment alone: the prompt of the teacher (`label_stance_part1.py`, section 2.2 of model_selection.ipynb)
    v2-post   the post and then the comment: the prompt of section 3.2 of model_selection.ipynb, copied without changes

The model, the runtime and the request settings are those of `label_stance_part1.py`, which is imported, not copied.

Step 1, on the computer that has `data/cleaned/submissions_clean.parquet` (no GPU needed):

    .venv\\Scripts\\python.exe scripts\\label_top_comments_llm.py --prepare

writes `data/top_comments/llm_input.csv` (the post and the comment of every sampled comment).

Step 2, on the computer with the GPU, after copying `llm_input.csv` into `data/top_comments/`:

    .venv\\Scripts\\python.exe scripts\\label_top_comments_llm.py

writes `data/top_comments/llm_predictions.csv` and `llm_run_config.json`; copy both back. llama.cpp and the model are
downloaded when missing, as in `label_stance_part1.py`. The run can be stopped (Ctrl+C) and started again: the
answers are saved every SAVE_EVERY comments and the (comment, prompt) pairs that have an answer are skipped.

    --dry-run    print the input and the prompt lengths and stop (no GPU needed)
"""

import argparse
import csv
import ctypes
import json
import os
import time

import pandas as pd
import pyarrow.parquet as pq

import label_stance_part1 as teacher

ROOT = teacher.ROOT
OUT_DIR = ROOT / "data" / "top_comments"
SAMPLE_PATH = OUT_DIR / "label_sample.csv"
SUBMISSIONS_PATH = ROOT / "data" / "cleaned" / "submissions_clean.parquet"
INPUT_PATH = OUT_DIR / "llm_input.csv"
PREDICTIONS_PATH = OUT_DIR / "llm_predictions.csv"
CONFIG_PATH = OUT_DIR / "llm_run_config.json"
teacher.SERVER_LOG_PATH = OUT_DIR / "llama_server.log"     # the server class of the teacher writes its log here

SAVE_EVERY = 50

# ---------------------------------------------------------------------------------------------------------------------
# prompt v2-post: copied without changes from section 3.2 (cell 64) of model_selection.ipynb
# ---------------------------------------------------------------------------------------------------------------------
POST_PROMPT_VERSION = "v2-post"
POST_PROMPT_SHA256 = "c430661f3b522cba32fcdab1f580276e5c011eaaab56d6e3d44fbb9e74ea141a"     # from data/sentiment/llm_post_run_config.json
POST_MAX_CHARS = 1000                 # a longer post is cut at the end
NO_POST = "(the post is not available)"
CONTROL_POST = "Trump holds a rally in Ohio"          # the post given to the control sentences, as in section 3.3

POST_SYSTEM_PROMPT = teacher.SYSTEM_PROMPT.replace(
    "You classify the stance of the author of a Reddit text toward Donald Trump.",
    "You classify the stance of the author of a Reddit comment toward Donald Trump.\n\n"
    "You receive the post under which the comment was written, and then the comment. The post is only context: it shows what the comment is about. "
    "Classify the position of the author of the comment, not of the post.")
assert POST_SYSTEM_PROMPT != teacher.SYSTEM_PROMPT

# the ten comments of section 2.2 with a post added (the headline became a comment under that headline), and three short comments that cannot be read without the post
POST_FEW_SHOT = [
    ("CNN panel calls Trump's press conference 'unhinged'",
     "The media is so dishonest it's disgusting. They twist every single thing Trump says and then act shocked when nobody trusts them anymore.", "pro"),
    ("Appeals court refuses to reinstate Trump's travel ban",
     "Best news all week! The court just threw out his travel ban again. Pop the champagne, folks.", "against"),
    ("Trump signs executive order on steel and aluminum tariffs",
     "The order takes effect in 15 days and exempts Canada and Mexico for now.", "neutral"),
    ("Trump calls himself 'a very stable genius'",
     "Oh sure, Trump is definitely a stable genius. Nothing says genius like bankrupting a casino. /s", "against"),
    ("Trump announces new tariffs on Chinese goods",
     "Can someone explain how the tariffs Trump announced actually work? Who pays them, the importer or the exporter?", "neutral"),
    ("Economists warn that Trump's policies could slow growth",
     "Yeah, because the economy was doing SO great before Trump took office. /s", "pro"),
    ("Democrats search for a message after their election losses",
     "With or without Trump on the ballot, the Democrats have no message at all. They lost the working class years ago and still don't understand why.", "neutral"),
    ("Fact check: Trump's claims about the size of the inauguration crowd",
     "He lied about the crowd size on day one and he has not stopped lying since. This administration is a disgrace.", "against"),
    ("One year in: how do Trump voters feel now?",
     "I don't like his tweets, but the tax cut and the judges are exactly what I voted for. He is delivering.", "pro"),
    ("Weekend open thread",
     "My rent went up again and I spent the whole weekend fixing my car. At least the Trump debate on TV was good background noise.", "neutral"),
    ("Trump fires FBI director James Comey",
     "About time. Should have happened on day one.", "pro"),
    ("Trump's approval rating falls to a new low",
     "Couldn't happen to a nicer guy.", "against"),
    ("Senate confirms Trump's Supreme Court nominee",
     "Who was the last nominee that got confirmed this fast?", "neutral"),
]


def post_turn(post, comment, max_chars=teacher.TEXT_MAX_CHARS):
    return {"role": "user", "content": f"Post:\n{post[:POST_MAX_CHARS] or NO_POST}\n\nComment:\n{comment[:max_chars]}"}


# everything before the last message is the same in every request and is reused from the cache of the server
POST_STATIC_MESSAGES = [{"role": "system", "content": POST_SYSTEM_PROMPT}]
for example_post, example, answer in POST_FEW_SHOT:
    POST_STATIC_MESSAGES += [post_turn(example_post, example), {"role": "assistant", "content": answer}]

assert teacher.sha256(json.dumps(POST_STATIC_MESSAGES)) == POST_PROMPT_SHA256, "the prompt differs from prompt v2-post of model_selection.ipynb"


def build_post_messages(post, comment, max_chars=teacher.TEXT_MAX_CHARS):
    return POST_STATIC_MESSAGES + [post_turn(post, comment, max_chars)]


# ---------------------------------------------------------------------------------------------------------------------
# input
# ---------------------------------------------------------------------------------------------------------------------
def prepare():
    """The post of every sampled comment: `text_clean` of its submission (the title and, for a text post, its body),
    the same field that section 3.1 of model_selection.ipynb gave to the model."""
    sample = pd.read_csv(SAMPLE_PATH, dtype=str, keep_default_na=False)
    posts = pq.read_table(SUBMISSIONS_PATH, columns=["id", "text_clean"],
                          filters=[("id", "in", sample["submission_id"].unique().tolist())]).to_pandas()
    posts = posts.rename(columns={"id": "submission_id", "text_clean": "post"})
    out = sample[["id", "submission_id", "comment"]].merge(posts, on="submission_id", how="left", validate="many_to_one")
    out["post"] = out["post"].fillna("").str.strip()
    out[["id", "submission_id", "post", "comment"]].to_csv(INPUT_PATH, index=False, lineterminator="\n")
    print(f"{len(out)} comments written to {INPUT_PATH}; {(out['post'] == '').sum()} have no post")


def load_input():
    if not INPUT_PATH.exists():
        raise SystemExit(f"{INPUT_PATH} was not found. Run this script with --prepare on the computer that has "
                         f"{SUBMISSIONS_PATH.name}, and copy the file here.")
    data = pd.read_csv(INPUT_PATH, dtype=str, keep_default_na=False)
    assert data["id"].is_unique and len(data) == 500, f"{INPUT_PATH.name} should hold the 500 comments of the sample"
    return data


def describe(data):
    no_post = data["post"] == ""
    v1 = data["comment"].map(lambda c: len(teacher.user_turn(c)["content"]))
    v2 = [len(post_turn(p, c)["content"]) for p, c in zip(data["post"], data["comment"])]
    print(f"{len(data)} comments; {no_post.sum()} have no post and get the note {NO_POST!r}; "
          f"{(data['post'].str.len() > POST_MAX_CHARS).sum()} posts are cut at {POST_MAX_CHARS} characters")
    print(f"last message in characters, median and longest: v1 {v1.median():.0f} and {v1.max()}, "
          f"{POST_PROMPT_VERSION} {pd.Series(v2).median():.0f} and {max(v2)}")


# ---------------------------------------------------------------------------------------------------------------------
# labelling
# ---------------------------------------------------------------------------------------------------------------------
class PostServer(teacher.LlamaServer):
    """The server of the teacher, which can also classify a comment together with its post."""

    def classify(self, text, post=None):
        if post is None:
            return super().classify(text)
        # the teacher's classify builds its messages with teacher.build_messages and halves the text when the prompt does
        # not fit the context; pointing that name to the post prompt for one call keeps the post and halves the comment
        original = teacher.build_messages
        teacher.build_messages = lambda comment, max_chars=teacher.TEXT_MAX_CHARS: build_post_messages(post, comment, max_chars)
        try:
            row = super().classify(text)
        finally:
            teacher.build_messages = original
        return {**row, "post_chars_used": min(len(post), POST_MAX_CHARS)}

    def check_post_prompt(self):
        control = {text: self.classify(text, CONTROL_POST)["label"] for text in teacher.CONTROL}
        assert control == teacher.CONTROL, f"unexpected labels on the control sentences with prompt {POST_PROMPT_VERSION}: {control}"
        return control


COLUMNS = ["id", "prompt_version", "label", "raw_output", "p_pro", "p_against", "p_neutral", "score",
           "prompt_tokens", "cached_tokens", "post_chars_used", "text_chars_used", "seconds"]


def saved_pairs():
    """(id, prompt version) of every answer already saved. A last line cut by a crash is removed from the file."""
    if not PREDICTIONS_PATH.exists():
        return set()
    data = PREDICTIONS_PATH.read_bytes()
    if not data.endswith(b"\n"):
        with open(PREDICTIONS_PATH, "wb") as f:
            f.write(data[:data.rfind(b"\n") + 1])
    saved = pd.read_csv(PREDICTIONS_PATH, dtype={"id": str}, usecols=["id", "prompt_version"])
    return set(zip(saved["id"], saved["prompt_version"]))


def save(rows):
    new = not PREDICTIONS_PATH.exists() or PREDICTIONS_PATH.stat().st_size == 0
    with open(PREDICTIONS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    rows.clear()


def classify_with_restarts(server, text, post):
    for attempt in range(teacher.MAX_RESTARTS + 1):
        try:
            return server.classify(text, post)
        except teacher.RequestFailed as e:
            if attempt == teacher.MAX_RESTARTS:
                raise
            print(f"\nrequest failed ({e}); restarting the server, attempt {attempt + 1} of {teacher.MAX_RESTARTS}")
            server.restart()


def label(data):
    done = saved_pairs()
    assert {i for i, _ in done} <= set(data["id"]), f"{PREDICTIONS_PATH.name} holds comments that are not in {INPUT_PATH.name}"
    # one prompt after the other, so the server reuses the cached prefix of that prompt
    versions = [(teacher.PROMPT_VERSION, False), (POST_PROMPT_VERSION, True)]
    todo = {v: data.loc[[(i, v) not in done for i in data["id"]]] for v, _ in versions}
    print(f"\nanswers already saved: {len(done)}; to label in this run: "
          + ", ".join(f"{v} {len(t)}" for v, t in todo.items()))
    if not any(len(t) for t in todo.values()):
        return

    teacher.ensure_llama_server()
    model_path = teacher.ensure_model()
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)  # Windows does not go to sleep during the run
    rows = []
    with PostServer(model_path) as server:
        post_control = server.check_post_prompt()
        CONFIG_PATH.write_text(json.dumps({
            "model": teacher.MODEL_NAME, "repo": teacher.MODEL_REPO, "file": teacher.MODEL_FILE, "sha256": teacher.sha256(model_path),
            "prompts": {teacher.PROMPT_VERSION: teacher.PROMPT_SHA256, POST_PROMPT_VERSION: POST_PROMPT_SHA256},
            "seed": teacher.SEED, "llama_cpp": teacher.LLAMA_BUILD, "server_flags": " ".join(teacher.SERVER_FLAGS),
            "generation": teacher.GENERATION, "context": teacher.CTX, "text_max_chars": teacher.TEXT_MAX_CHARS,
            "post_max_chars": POST_MAX_CHARS, "gpu": teacher.gpu_description(), "run_at": time.strftime("%Y-%m-%d %H:%M"),
            **server.checks, "post_control": post_control,
        }, indent=2), encoding="utf-8")
        print(f"server is ready: {server.checks['layers_on_gpu']} layers on the GPU, build {server.checks['build']}, "
              "thinking is off, control sentences correct with both prompts\n")
        try:
            for version, with_post in versions:
                part, start = todo[version], time.perf_counter()
                counts = dict.fromkeys(teacher.LABELS + ["error"], 0)
                for n, (comment_id, post, comment) in enumerate(zip(part["id"], part["post"], part["comment"]), start=1):
                    row = classify_with_restarts(server, comment, post if with_post else None)
                    counts[row["label"]] += 1
                    row["raw_output"] = row["raw_output"].replace("\r", "\\r").replace("\n", "\\n")
                    rows.append({"id": comment_id, "prompt_version": version, "post_chars_used": 0,
                                 **{k: v for k, v in row.items() if k in COLUMNS}})
                    if len(rows) >= SAVE_EVERY or n == len(part):
                        save(rows)
                        print(f"{version:>8} {n:>4}/{len(part)}  {n / (time.perf_counter() - start):.2f} comments/s  "
                              + "  ".join(f"{k} {v}" for k, v in counts.items()), flush=True)
        except KeyboardInterrupt:
            print("\nstopped by the user")
        finally:
            if rows:
                save(rows)
            print(f"answers are saved in {PREDICTIONS_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Label the hand-labelled top-level comments with Qwen3.5-4B, without and with the post.")
    parser.add_argument("--prepare", action="store_true", help=f"write {INPUT_PATH.name} from the sample and the submissions, and stop")
    parser.add_argument("--dry-run", action="store_true", help="print the input and the prompt lengths, and stop")
    args = parser.parse_args()

    if args.prepare:
        prepare()
        return
    data = load_input()
    describe(data)
    if not args.dry_run:
        label(data)


if __name__ == "__main__":
    main()
