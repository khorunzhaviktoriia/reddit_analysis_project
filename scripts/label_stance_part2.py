"""Label the stance toward Donald Trump of a sample of 100,000 Reddit comments with a local Qwen3.5-4B.

The sample is split into two parts, one for each computer. This file labels the part given in PART;
`label_stance_part1.py` and `label_stance_part2.py` differ only in that line.

Run from the project root, with the virtual environment of the project:

    .venv\\Scripts\\python.exe scripts\\label_stance_part2.py

The run can be stopped (Ctrl+C) and started again at any time: the answers are saved every
SAVE_EVERY comments and the comments that already have an answer are skipped.

    --dry-run    draw the sample, print its composition and stop (no GPU needed)
    --limit N    stop when N comments of this part have an answer

The model, the runtime, the prompt and the request settings are those tested in section 2 of
`notebooks/EDA/model_selection.ipynb` (model `qwen4b`, prompt `v1`, the comment alone without its thread).
"""

import argparse
import csv
import ctypes
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests

PART = 2                          # the part of the sample labelled by this file: 1 or 2

ROOT = Path(__file__).resolve().parent.parent
COMMENTS_PATH = ROOT / "data" / "cleaned" / "comments_clean.parquet"
MODEL_DIR = ROOT / "data" / "models"
OUT_DIR = ROOT / "data" / "stance"

SAMPLE_PATH = OUT_DIR / f"sample_part{PART}.parquet"
LABELS_PATH = OUT_DIR / f"labels_part{PART}.csv"
ERRORS_PATH = OUT_DIR / f"errors_part{PART}.log"
CONFIG_PATH = OUT_DIR / f"run_config_part{PART}.json"
SERVER_LOG_PATH = OUT_DIR / f"llama_server_part{PART}.log"

SEED = 42

# ---------------------------------------------------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------------------------------------------------
N_SAMPLE = 100_000                # comments in the whole sample, both parts together
N_PARTS = 2
# SHA-256 of the sorted ids of the whole sample. Both computers must draw the same sample from their copy of the data;
# a different value stops the run before two computers label different or overlapping comments.
EXPECTED_SAMPLE_SHA256 = "b853d6183a4bf4ff0f4520e8eb260b5e9ccb6e2cfa94dff0e96751301c505c84"

# ---------------------------------------------------------------------------------------------------------------------
# model and runtime
# ---------------------------------------------------------------------------------------------------------------------
LLAMA_BUILD = "b11146"            # llama.cpp release (CUDA 12.4 build for Windows); Qwen3.5 needs b8121 or newer
LLAMA_SERVER = Path(sys.prefix) / "llama.cpp" / LLAMA_BUILD / "llama-server.exe"      # kept inside the virtual environment
MODEL_NAME = "qwen4b"
MODEL_REPO, MODEL_FILE = "unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf"

CTX = 2048                        # context of the server, in tokens
# One slot: on the GTX 1650 two and four slots gave the same speed (1.49, 1.51 and 1.48 texts per second)
# and changed 2.5% to 4% of the labels.
SLOTS = 1
SHARED_LIMIT_MIB = 150            # more "shared GPU memory" than this means the model spilled from VRAM into system RAM
TEXT_MAX_CHARS = 4000             # a longer text is cut at the end
SAVE_EVERY = 100                  # answers are written to the disk every SAVE_EVERY comments
SPILL_CHECK_EVERY = 1000          # the shared GPU memory is checked again every SPILL_CHECK_EVERY comments
MAX_RESTARTS = 3                  # server restarts for one comment before the run stops

# all layers on the GPU, no thinking, no visual encoder (mmproj)
SERVER_FLAGS = ["-ngl", "99", "-c", str(CTX * SLOTS), "-np", str(SLOTS), "--no-kv-unified", "--seed", str(SEED),
                "--reasoning", "off", "--no-mmproj", "--no-webui", "-lv", "4"]

GENERATION = {
    "temperature": 0, "max_tokens": 5, "seed": SEED, "cache_prompt": True,
    "chat_template_kwargs": {"enable_thinking": False},
    "grammar": 'root ::= "pro" | "against" | "neutral"',       # GBNF: the output can only be one of the three labels
    "logprobs": True, "top_logprobs": 20,                       # probabilities of the first token, for a score from -1 to +1
}
LABELS = ["pro", "against", "neutral"]
CONTROL = {"Trump is the worst president we have ever had.": "against", "Go Trump, the best president ever!": "pro"}

# ---------------------------------------------------------------------------------------------------------------------
# prompt: copied without changes from section 2.2 of model_selection.ipynb
# ---------------------------------------------------------------------------------------------------------------------
PROMPT_VERSION = "v1"
PROMPT_SHA256 = "374edfd531e9ba4a7591156792fdd8f6115e8f4ad8db8c684493618557394c6d"       # from data/sentiment/llm_run_config.json

SYSTEM_PROMPT = """You classify the stance of the author of a Reddit text toward Donald Trump.

Answer with exactly one word: pro, against or neutral.

- against: the author criticises, mocks or opposes Trump, his decisions or his administration.
- pro: the author supports, defends or praises Trump or his decisions.
- neutral: the author shows no clear position toward Trump. This covers a plain news headline, a question, a fact, and a text about something else that only mentions Trump.

Rules:
1. Judge the author's position toward Trump, not the tone of the text. An angry text is pro when the anger defends Trump against his critics. A cheerful text is against when the author is happy about a failure of Trump.
2. Sarcasm and irony (often marked with /s) are judged by what the author means, not by the words.
3. A headline that only reports what happened is neutral, even when the news is bad for Trump. A headline with an evaluation in it is not neutral.
4. An attack on Trump's opponents or on the media is pro only when the defence of Trump is clear. Otherwise it is neutral.
5. A text that criticises one decision and supports Trump in general (or the opposite) gets the label of its main point.
6. A quote of Trump with no comment from the author is neutral."""

FEW_SHOT = [
    ("The media is so dishonest it's disgusting. They twist every single thing Trump says and then act shocked when nobody trusts them anymore.", "pro"),          # negative tone, supportive stance
    ("Best news all week! The court just threw out his travel ban again. Pop the champagne, folks.", "against"),                                                  # positive tone, opposing stance
    ("Trump signs executive order on steel and aluminum tariffs", "neutral"),                                                                                     # plain headline
    ("Oh sure, Trump is definitely a stable genius. Nothing says genius like bankrupting a casino. /s", "against"),                                               # sarcasm, praise in words only
    ("Can someone explain how the tariffs Trump announced actually work? Who pays them, the importer or the exporter?", "neutral"),                               # question
    ("Yeah, because the economy was doing SO great before Trump took office. /s", "pro"),                                                                         # sarcasm, criticism in words only
    ("With or without Trump on the ballot, the Democrats have no message at all. They lost the working class years ago and still don't understand why.", "neutral"),  # attack on opponents, no defence of Trump
    ("He lied about the crowd size on day one and he has not stopped lying since. This administration is a disgrace.", "against"),                                # plain opposition
    ("I don't like his tweets, but the tax cut and the judges are exactly what I voted for. He is delivering.", "pro"),                                           # mixed, the main point is support
    ("My rent went up again and I spent the whole weekend fixing my car. At least the Trump debate on TV was good background noise.", "neutral"),                 # not about Trump
]


def user_turn(text, max_chars=TEXT_MAX_CHARS):
    return {"role": "user", "content": "Text:\n" + text[:max_chars]}


# everything except the last message is the same in every request, so the server computes this prefix once and reuses it
STATIC_MESSAGES = [{"role": "system", "content": SYSTEM_PROMPT}]
for example, answer in FEW_SHOT:
    STATIC_MESSAGES += [user_turn(example), {"role": "assistant", "content": answer}]


def build_messages(text, max_chars=TEXT_MAX_CHARS):
    return STATIC_MESSAGES + [user_turn(text, max_chars)]


def sha256(data):
    """SHA-256 of a file (given its path) or of a string."""
    digest = hashlib.sha256()
    if isinstance(data, Path):
        with open(data, "rb") as f:
            while chunk := f.read(1 << 22):
                digest.update(chunk)
    else:
        digest.update(data.encode("utf-8"))
    return digest.hexdigest()


assert sha256(json.dumps(STATIC_MESSAGES)) == PROMPT_SHA256, "the prompt differs from the prompt tested in model_selection.ipynb"


# ---------------------------------------------------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------------------------------------------------
def allocate(sizes, total):
    """Split `total` over the strata as evenly as their sizes allow: every stratum gets the same number of rows,
    a stratum smaller than that is taken whole, and what it could not give goes to the larger strata."""
    assert sizes.sum() >= total, f"only {sizes.sum()} eligible comments for a sample of {total}"
    sizes = sizes.sort_index()
    low, high = 0, int(sizes.max())
    while low < high:                                           # the largest cap that does not exceed `total`
        cap = (low + high + 1) // 2
        if sizes.clip(upper=cap).sum() <= total:
            low = cap
        else:
            high = cap - 1
    n = sizes.clip(upper=low)
    larger = sizes.index[sizes > low]
    n.loc[larger[:total - n.sum()]] += 1                        # the remainder, one row each
    assert n.sum() == total
    return n, low


def draw_sample():
    """The whole sample with its texts. Eligible is a comment that mentions Trump in its own text and not only in a quote.
    The strata are subreddit x month. Inside a stratum the comments are ranked by a hash of their id, so the same
    comments are chosen on every computer and with every version of pandas."""
    file = pq.ParquetFile(COMMENTS_PATH)
    columns = ["id", "subreddit", "period", "source_month", "created_at"]
    groups = []
    for i in range(file.metadata.num_row_groups):               # one row group at a time: the texts do not fit in memory together
        table = file.read_row_group(i, columns=columns + ["trump_only_in_quote", "body_clean"])
        eligible = pc.and_(pc.invert(table["trump_only_in_quote"]), pc.match_substring(table["body_clean"], "trump", ignore_case=True))
        groups.append(table.filter(pc.fill_null(eligible, False)).select(columns).to_pandas())
    com = pd.concat(groups, ignore_index=True)
    assert com["id"].is_unique

    strata = ["subreddit", "source_month"]
    com["rank"] = [hashlib.sha256(f"{SEED}:{i}".encode()).hexdigest()[:16] for i in com["id"]]
    com = com.sort_values(strata + ["rank"], ignore_index=True)
    sizes = com.groupby(strata).size()
    n, cap = allocate(sizes, N_SAMPLE)
    index = pd.MultiIndex.from_frame(com[strata])
    com["stratum_size"], com["stratum_sampled"] = sizes.reindex(index).to_numpy(), n.reindex(index).to_numpy()
    chosen = com.loc[com.groupby(strata).cumcount() < com["stratum_sampled"]].copy()

    chosen["part"] = np.arange(len(chosen)) % N_PARTS + 1       # rows are ordered by stratum, so the parts get the same mix
    chosen = chosen.sort_values("rank", ignore_index=True)      # a shuffled order: an unfinished part is still a balanced sample
    chosen["order"] = chosen.groupby("part").cumcount() + 1

    text = pq.read_table(COMMENTS_PATH, columns=["id", "body_clean"], filters=[("id", "in", chosen["id"].tolist())]).to_pandas()
    chosen = chosen.merge(text.rename(columns={"body_clean": "text"}), on="id", validate="one_to_one")
    print(f"sample: {len(chosen):,} of {len(com):,} eligible comments, at most {cap + 1} per subreddit and month; "
          f"{(sizes <= cap).sum()} of {len(sizes)} strata are taken whole")
    return chosen[["id", "subreddit", "period", "source_month", "created_at", "part", "order", "stratum_size", "stratum_sampled", "text"]]


def sample_sha256(ids):
    return sha256("\n".join(sorted(ids)))


def load_sample():
    """The comments of this part, in the order of labelling. The sample is drawn once and saved, one file for each part."""
    if not SAMPLE_PATH.exists():
        whole = draw_sample()
        digest = sample_sha256(whole["id"])
        print(f"sample sha256: {digest}")
        if EXPECTED_SAMPLE_SHA256 and digest != EXPECTED_SAMPLE_SHA256:
            raise SystemExit(f"This computer drew a different sample than expected ({EXPECTED_SAMPLE_SHA256}): its copy of "
                             f"{COMMENTS_PATH.name} differs. Copy {SAMPLE_PATH.name} from the other computer into {OUT_DIR}.")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for part, rows in whole.groupby("part"):
            rows.to_parquet(OUT_DIR / f"sample_part{part}.parquet", index=False)
    sample = pd.read_parquet(SAMPLE_PATH).sort_values("order", ignore_index=True)
    assert sample["id"].is_unique and (sample["part"] == PART).all()
    return sample


def describe(sample):
    months = pd.crosstab(sample["source_month"], sample["subreddit"], margins=True)
    print(f"\npart {PART}: {len(sample):,} comments, rows = month, columns = subreddit")
    print(months.to_string())
    chars = sample["text"].str.len()
    print(f"\ntext length in characters: median {chars.median():.0f}, 90% below {chars.quantile(0.9):.0f}, "
          f"cut at {TEXT_MAX_CHARS}: {(chars > TEXT_MAX_CHARS).mean():.1%} of the comments")


# ---------------------------------------------------------------------------------------------------------------------
# runtime
# ---------------------------------------------------------------------------------------------------------------------
class DoesNotFit(RuntimeError):
    """The model is not completely on the GPU."""


class RequestFailed(RuntimeError):
    """The server did not answer a request. This is not an answer of the model, so the comment gets no label."""


class ContextOverflow(RuntimeError):
    """The prompt is longer than the context of the server."""


def process_shared_gpu_memory_mib(pid):
    """Shared GPU memory of one process, from the Windows performance counters. It is system RAM that the driver lends
    to the GPU, and it grows when the VRAM is full. None when the counters cannot be read."""
    command = (f"(Get-Counter '\\GPU Process Memory(pid_{pid}_*)\\Shared Usage').CounterSamples "
               "| ForEach-Object { [long]$_.CookedValue }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True, timeout=60).stdout
        values = [int(line) for line in out.split()]
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    return sum(values) / 2**20 if values else None


def gpu_description():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def label_probabilities(choice):
    """Probabilities of the three labels as the first output token, scaled to sum to 1."""
    try:
        first = {t["token"]: np.exp(t["logprob"]) for t in choice["logprobs"]["content"][0]["top_logprobs"]}
    except (KeyError, IndexError, TypeError):
        first = {}
    total = sum(first.get(label, 0.0) for label in LABELS)
    return {f"p_{label}": first.get(label, 0.0) / total if total else np.nan for label in LABELS}


class LlamaServer:
    """One llama-server process. The model must be on the GPU completely, and the visual encoder (mmproj) is not loaded."""

    def __init__(self, model_path):
        self.model_path = model_path

    def __enter__(self):
        with socket.socket() as s:                              # a free port
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        self._log = open(SERVER_LOG_PATH, "w", encoding="utf-8")
        self.process = subprocess.Popen([str(LLAMA_SERVER), "-m", str(self.model_path), *SERVER_FLAGS, "--host", "127.0.0.1", "--port", str(port)],
                                        stdout=self._log, stderr=subprocess.STDOUT)
        try:
            for _ in range(600):
                if self.process.poll() is not None:
                    raise DoesNotFit(f"llama-server stopped while loading {self.model_path.name}; see {SERVER_LOG_PATH}")
                try:
                    if requests.get(self.url + "/health", timeout=2).status_code == 200:
                        self.checks = self.check()
                        return self
                except requests.RequestException:
                    pass
                time.sleep(0.5)
            raise TimeoutError(f"llama-server did not become ready; see {SERVER_LOG_PATH}")
        except BaseException:
            self.__exit__()
            raise

    def __exit__(self, *exc):
        self.process.terminate()
        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.process.kill()
        self._log.close()

    def restart(self):
        self.__exit__()
        time.sleep(5)
        self.__enter__()

    def chat(self, messages, **overrides):
        body = {k: v for k, v in {**GENERATION, **overrides, "messages": messages}.items() if v is not None}
        try:
            response = requests.post(self.url + "/v1/chat/completions", json=body, timeout=600)
        except requests.RequestException as e:
            raise RequestFailed(f"{type(e).__name__}: {e}") from e
        if response.status_code == 400 and "context" in response.text:
            raise ContextOverflow(response.text)
        if response.status_code != 200:
            raise RequestFailed(f"HTTP {response.status_code}: {response.text[:300]}")
        return response.json()

    def classify(self, text):
        """One text -> label, raw output, label probabilities, token counts and time. An answer that is not a label becomes 'error'."""
        start, max_chars = time.perf_counter(), TEXT_MAX_CHARS
        while True:
            try:
                answer = self.chat(build_messages(text, max_chars))
                break
            except ContextOverflow:                             # a dense text: more tokens than the context holds
                max_chars //= 2
                if max_chars < 250:
                    raise RequestFailed("the prompt does not fit the context even with a text of 250 characters")
        try:
            choice, timings = answer["choices"][0], answer["timings"]
            raw = choice["message"].get("content") or ""
            thinking = choice["message"].get("reasoning_content") or ""
        except (KeyError, IndexError, TypeError) as e:
            raise RequestFailed(f"unexpected answer of the server: {str(answer)[:300]}") from e
        if thinking or "<think>" in raw:
            raise RuntimeError(f"the model is thinking, and thinking must be off: {raw!r} {thinking[:200]!r}")
        row = {"label": raw if raw in LABELS else "error", "raw_output": raw, **label_probabilities(choice),
               "prompt_tokens": timings["prompt_n"] + timings["cache_n"], "cached_tokens": timings["cache_n"],
               "text_chars_used": min(len(text), max_chars), "seconds": round(time.perf_counter() - start, 3)}
        row["score"] = row["p_pro"] - row["p_against"]            # from -1 (surely against) to +1 (surely pro)
        return row

    def check_memory(self):
        shared = process_shared_gpu_memory_mib(self.process.pid)
        if shared is None:
            print("warning: the shared GPU memory of the server could not be read; only the number of layers on the GPU is checked")
        elif shared > SHARED_LIMIT_MIB:
            raise DoesNotFit(f"{self.model_path.name}: {shared:.0f} MiB of shared GPU memory (limit {SHARED_LIMIT_MIB}): "
                             "the model spilled into system RAM. Close other programs that use the GPU and start again.")
        return shared

    def check(self):
        """Stop when the model is not fully on the GPU, when it thinks, or when it fails on the two control sentences."""
        self._log.flush()
        log = SERVER_LOG_PATH.read_text(encoding="utf-8", errors="replace")
        on_gpu, layers = map(int, re.search(r"offloaded (\d+)/(\d+) layers to GPU", log).groups())
        if on_gpu < layers:
            raise DoesNotFit(f"{self.model_path.name}: only {on_gpu}/{layers} layers are on the GPU")
        shared = self.check_memory()

        # the grammar would hide a thinking block, so thinking is tested with a free answer
        text = next(iter(CONTROL))
        rendered = requests.post(self.url + "/apply-template", json={
            "messages": build_messages(text), "chat_template_kwargs": GENERATION["chat_template_kwargs"]}).json()["prompt"]
        free = self.chat(build_messages(text), grammar=None, logprobs=None, top_logprobs=None, max_tokens=40)["choices"][0]["message"]
        assert rendered.rstrip().endswith("</think>"), f"the prompt does not close the thinking block: {rendered[-80:]!r}"
        assert not free.get("reasoning_content") and "<think>" not in free["content"], f"the model is thinking: {free}"

        control = {text: self.classify(text)["label"] for text in CONTROL}
        assert control == CONTROL, f"unexpected labels on the control sentences: {control}"
        return {"layers_on_gpu": f"{on_gpu}/{layers}", "shared_gpu_memory_mib": shared, "free_answer": free["content"],
                "prompt_ends_with": rendered[-40:], "build": requests.get(self.url + "/props").json()["build_info"]}


# ---------------------------------------------------------------------------------------------------------------------
# labelling
# ---------------------------------------------------------------------------------------------------------------------
COLUMNS = ["id", "label", "raw_output", *[f"p_{label}" for label in LABELS], "score", "prompt_tokens", "cached_tokens",
           "text_chars_used", "seconds", "model", "prompt_version"]


def saved_labels():
    """Label of every comment that already has an answer. A last line that was cut by a crash is removed from the file."""
    if not LABELS_PATH.exists():
        return {}
    data = LABELS_PATH.read_bytes()
    if not data.endswith(b"\n"):
        with open(LABELS_PATH, "wb") as f:
            f.write(data[:data.rfind(b"\n") + 1])
    saved = pd.read_csv(LABELS_PATH, dtype={"id": str}, usecols=["id", "label", "prompt_version"])
    assert set(saved["prompt_version"]) <= {PROMPT_VERSION}, f"{LABELS_PATH} holds answers of another prompt version"
    return dict(zip(saved["id"], saved["label"]))


def save(rows):
    new = not LABELS_PATH.exists() or LABELS_PATH.stat().st_size == 0
    with open(LABELS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    rows.clear()


def log_error(comment_id, raw):
    with open(ERRORS_PATH, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{comment_id}\t{raw!r}\n")


def classify_with_restarts(server, text):
    for attempt in range(MAX_RESTARTS + 1):
        try:
            return server.classify(text)
        except RequestFailed as e:
            if attempt == MAX_RESTARTS:
                raise
            print(f"\nrequest failed ({e}); restarting the server, attempt {attempt + 1} of {MAX_RESTARTS}")
            server.restart()


def label_sample(sample, limit):
    done = saved_labels()
    counts = pd.Series(list(done.values())).value_counts().reindex(LABELS + ["error"], fill_value=0).to_dict()
    todo = sample.loc[~sample["id"].isin(done.keys())]
    target = len(sample) if limit is None else min(limit, len(sample))
    todo = todo.head(max(target - len(done), 0))
    print(f"\n{len(done):,} of {len(sample):,} comments already have an answer; {len(todo):,} to label in this run")
    if todo.empty:
        return

    if not LLAMA_SERVER.exists():
        raise SystemExit(f"llama-server was not found at {LLAMA_SERVER}. Unpack the llama.cpp release {LLAMA_BUILD} "
                         "(Windows, CUDA 12.4, with the cudart files) into that folder.")
    model_path = MODEL_DIR / MODEL_FILE
    if not model_path.exists():
        from huggingface_hub import hf_hub_download
        hf_hub_download(MODEL_REPO, MODEL_FILE, local_dir=MODEL_DIR)

    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)  # Windows does not go to sleep during the run
    rows, n_done, start = [], 0, time.perf_counter()
    with LlamaServer(model_path) as server:
        CONFIG_PATH.write_text(json.dumps({
            "part": PART, "model": MODEL_NAME, "repo": MODEL_REPO, "file": MODEL_FILE, "sha256": sha256(model_path),
            "prompt_version": PROMPT_VERSION, "prompt_sha256": PROMPT_SHA256, "seed": SEED, "llama_cpp": LLAMA_BUILD,
            "server_flags": " ".join(SERVER_FLAGS), "generation": GENERATION, "context": CTX, "text_max_chars": TEXT_MAX_CHARS,
            "gpu": gpu_description(), "run_at": time.strftime("%Y-%m-%d %H:%M"), **server.checks,
        }, indent=2), encoding="utf-8")
        print(f"server is ready: {server.checks['layers_on_gpu']} layers on the GPU, build {server.checks['build']}, thinking is off\n")
        try:
            for comment_id, text in zip(todo["id"], todo["text"]):
                row = classify_with_restarts(server, text)
                if row["label"] == "error":
                    log_error(comment_id, row["raw_output"])
                counts[row["label"]] += 1
                row["raw_output"] = row["raw_output"].replace("\r", "\\r").replace("\n", "\\n")       # one answer = one line of the file
                rows.append({"id": comment_id, **row, "model": MODEL_NAME, "prompt_version": PROMPT_VERSION})
                n_done += 1
                if len(rows) >= SAVE_EVERY:
                    save(rows)
                    speed = n_done / (time.perf_counter() - start)
                    print(f"{len(done) + n_done:>7,}/{target:,}  {speed:.2f} texts/s  {(len(todo) - n_done) / speed / 3600:5.1f} h left  "
                          + "  ".join(f"{label} {n:,}" for label, n in counts.items()), flush=True)
                if n_done % SPILL_CHECK_EVERY == 0:
                    server.check_memory()
        except KeyboardInterrupt:
            print("\nstopped by the user")
        finally:
            if rows:
                save(rows)
            print(f"{len(done) + n_done:,} answers are saved in {LABELS_PATH}")


def main():
    parser = argparse.ArgumentParser(description=f"Label part {PART} of the stance sample with Qwen3.5-4B.")
    parser.add_argument("--dry-run", action="store_true", help="draw the sample, print its composition and stop")
    parser.add_argument("--limit", type=int, help="stop when this many comments of the part have an answer")
    args = parser.parse_args()

    sample = load_sample()
    describe(sample)
    if not args.dry_run:
        label_sample(sample, args.limit)


if __name__ == "__main__":
    main()
