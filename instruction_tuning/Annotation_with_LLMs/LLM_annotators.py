"""
Long-format structured multi-LLM voting (Gemini call logic EXACTLY mirrors [Create_Model_Comparison.py](http://_vscodecontentref_/1) style).

Adds rows (annotator_type ∈ {GPT_5, Gemini_2_5_Pro, Claude_Sonnet_4, Aggregate_LLM})
Schema per row:
  annotator_type, source_type, text_hash, text, model_A, model_B,
  choice, instruction_A, response_A, instruction_B, response_B, timestamp

Key: (source_type, text_hash, model_A, model_B, annotator_type)

Aggregate_LLM only when all 3 individual LLM votes exist.
Prompt body unchanged.
"""
from __future__ import annotations
import vertexai
from vertexai.preview.generative_models import GenerativeModel
import argparse, json, os, sys, time, hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random
import pandas as pd
from tqdm import tqdm

from huggingface_hub import HfApi, hf_hub_download, create_repo
try:
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    class HfHubHTTPError(Exception): pass

from openai import OpenAI
import anthropic

from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import threading
from collections import defaultdict

# Add lock for thread-safe operations
lock = threading.Lock()
progress_lock = threading.Lock()
push_lock = threading.Lock()

# ---------------- CONFIG ----------------
PAIRS_CSV = Path("outputs/pairs.csv")
ANNOT_CSV_LOCAL = Path("annotations_Wiki_Native.csv")
HF_REPO = "jmcinern/Irish_Prompt_Response_Human_Feedback"
HF_FILENAME = "annotations_Wiki_Native.csv"

OPENAI_VOTE_MODEL = "gpt-5"
GEMINI_VOTE_MODEL =  "gemini-2.5-pro" # "gemini-2.5-pro"
ANTHROPIC_VOTE_MODEL = "claude-sonnet-4-20250514"

LLM_ANNOTATORS = ["GPT_5", "Gemini_2_5_Pro", "Claude_Sonnet_4"]
AGG_ANNOTATOR = "Aggregate_LLM"

RETRY_MAX = 2
RETRY_SLEEP = 2.0

# Number of worker threads per LLM provider
WORKERS_PER_LLM = 3
MAX_TOTAL_WORKERS = 12

# ------------- Utils -------------
def sha1_short(t: str, length: int = 16) -> str:
    return hashlib.sha1(t.encode("utf-8")).hexdigest()[:length]

def utc_timestamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def load_secrets(path="secrets.json") -> Dict[str, str]:
    if not os.path.isfile(path): return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        data = data[0] if data else {}
    return data

def download_existing() -> pd.DataFrame:
    api = HfApi()
    try:
        p = hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME, repo_type="dataset")
        df = pd.read_csv(p)
        print(f"Loaded existing annotations from HF: {HF_REPO}/{HF_FILENAME} (rows={len(df)})")
        return df
    except HfHubHTTPError as e:
        st = getattr(getattr(e, "response", None), "status_code", None)
        if st == 404 or "404" in str(e):
            print("No existing HF annotations file (starting new).")
            return pd.DataFrame()
        print(f"[WARN] HF download error: {e}")
        return pd.DataFrame()
    except Exception as e:
        if "404" in str(e):
            print("No existing HF annotations file (starting new).")
            return pd.DataFrame()
        print(f"[WARN] Unexpected HF download error: {e}")
        return pd.DataFrame()

def load_pairs() -> pd.DataFrame:
    if not PAIRS_CSV.exists():
        print(f"pairs.csv not found at {PAIRS_CSV}")
        sys.exit(1)
    df = pd.read_csv(PAIRS_CSV)
    needed = {"model", "source_type", "instruction", "response", "text"}
    miss = needed.difference(df.columns)
    if miss:
        print(f"pairs.csv missing columns: {miss}")
        sys.exit(1)
    return df

def build_comparisons(pairs_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    for (stype, text), group in pairs_df.groupby(["source_type", "text"]):
        text_hash = sha1_short(text)
        group = group.drop_duplicates(subset=["model"], keep="last")
        models = sorted(group["model"].tolist())
        if len(models) < 2: continue
        by_model = group.set_index("model")[["instruction", "response"]].to_dict("index")
        for i in range(len(models)-1):
            for j in range(i+1, len(models)):
                mA, mB = models[i], models[j]
                rA, rB = by_model[mA], by_model[mB]
                rows.append({
                    "source_type": stype,
                    "text_hash": text_hash,
                    "text": text,
                    "model_A": mA,
                    "model_B": mB,
                    "instruction_A": rA["instruction"],
                    "response_A": rA["response"],
                    "instruction_B": rB["instruction"],
                    "response_B": rB["response"],
                })
    df = pd.DataFrame(rows)
    print(f"Built {len(df)} comparisons from {len(pairs_df)} model outputs.")
    return df

def comp_key(base: Dict[str, str]) -> str:
    return f"{base['source_type']}||{base['text_hash']}||{base['model_A']}||{base['model_B']}"

def vote_key(base: Dict[str, str], annotator_type: str) -> str:
    return f"{comp_key(base)}||{annotator_type}"

# exponential retry and jitter to reduce pressure on API
def call_with_retry(label: str, fn):
    last_exception = None
    base_delay = RETRY_SLEEP # e.g., 2.0 seconds
    for attempt in range(RETRY_MAX):
        try:
            return fn()
        except Exception as e:
            last_exception = e
            # Only print the warning for retriable errors, or be more specific later
            print(f"[WARN] {label} attempt {attempt + 1} failed: {e}")

            if attempt == RETRY_MAX - 1:
                # Break on the last attempt without sleeping
                break

            # Exponential backoff with jitter
            delay = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
            print(f"[INFO] Retrying in {delay:.2f} seconds...")
            time.sleep(delay)

    print(f"[ERROR] {label} failed after {RETRY_MAX} attempts. Last error: {last_exception}")
    return None


# ---------- Prompt (UNCHANGED) ----------
def build_vote_prompt(source_text: str,
                      model_A: str, instr_A: str, resp_A: str,
                      model_B: str, instr_B: str, resp_B: str) -> str:
    return f"""You are evaluating Irish QA pairs.

Reference Text:
{source_text}

Instruction A:
{instr_A}
Response A:
{resp_A}

Instruction B:
{instr_B}
Response B:
{resp_B}

Question: Which Question–Answer pair exhibits a stronger command of Irish grammar and semantic coherence? 
Take into account use of the reference text. If unsure, pick the one with stronger Irish grammar.

Output MUST be exactly a single character: 'A' OR 'B'
No punctuation, no explanation, no extra whitespace.
Answer:
"""

# ---------- Structured vote (no heuristic fallback) ----------

def openai_vote(client: OpenAI, model: str, prompt: str) -> Optional[str]:
    resp = client.responses.create(
        model=model,
        reasoning={"effort": "low"},
        instructions="Only output the character A or B as response.",
        input=prompt,
    )
    print(f"OpenAI response: {resp.output_text}")
    v = resp.output_text.strip().upper()
    return v if v in ("A", "B") else None

# Anthropic tool schema
ANTHROPIC_TOOL = {
    "name": "record_vote",
    "description": "Return which answer is better.",
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {"vote": {"type": "string", "enum": ["A", "B"]}},
        "required": ["vote"]
    }
}

def anthropic_vote(client: anthropic.Anthropic, model: str, prompt: str) -> Optional[str]:
    r = client.messages.create(
        model=model,
        max_tokens=64,
        temperature=0.0,
        tools=[ANTHROPIC_TOOL],
        tool_choice={"type": "tool", "name": "record_vote"},
        messages=[{"role": "user", "content": prompt}]
    )
    for block in r.content:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", "") == "record_vote":
            vote = (getattr(block, "input", {}) or {}).get("vote")
            return vote if vote in ("A", "B") else None
    print("[WARN] Anthropic structured vote missing")
    return None

def gemini_vote(model_obj: GenerativeModel, prompt: str) -> Optional[str]:
    """
    Calls Gemini for standard text generation using a GenerativeModel object.
    This now exactly matches the pattern from the successful test script.
    """
    try:
        gemini_project_id = "gen-lang-client-0817118952" 
        gcloud_location = "us-central1"

        vertexai.init(project=gemini_project_id, location=gcloud_location)

        model = GenerativeModel('gemini-2.5-pro')
        response = model.generate_content(prompt)

        return response.text or None

    except:
        print(f"[WARN] Gemini vote parse failed. Raw output: {response!r}")
        return None


def majority_three(votes: List[str]) -> Optional[str]:
    if len(votes) != 3 or any(v not in ("A","B") for v in votes): return None
    return "A" if votes.count("A") > votes.count("B") else "B"

def push_to_hf(df: pd.DataFrame, hf_token: str, message: str = "Incremental update") -> bool:
    """
    Push the current dataframe to Hugging Face.
    Returns True if successful, False otherwise.
    """
    try:
        # Save locally first
        tmp = ANNOT_CSV_LOCAL.with_suffix(".tmp.csv")
        df.to_csv(tmp, index=False)
        tmp.replace(ANNOT_CSV_LOCAL)
        
        # Push to HF
        api = HfApi()
        create_repo(HF_REPO, repo_type="dataset", exist_ok=True, token=hf_token)
        api.upload_file(
            path_or_fileobj=str(ANNOT_CSV_LOCAL),
            path_in_repo=HF_FILENAME,
            repo_id=HF_REPO,
            repo_type="dataset",
            token=hf_token,
            commit_message=message
        )
        return True
    except Exception as e:
        print(f"[WARN] Push to HF failed: {e}")
        return False

def process_single_llm_vote(
    annotator: str,
    base: dict,
    prompt: str,
    existing_keys: set,
    openai_client: Optional[OpenAI],
    anthro_client: Optional[anthropic.Anthropic],
    gemini_model: Optional[GenerativeModel],
    pbar: tqdm
) -> Optional[Tuple[str, dict]]:
    """
    Process a single LLM vote and update progress bar.
    Returns (annotator_type, result_dict) or None
    """
    k = vote_key(base, annotator)
    
    # Skip if already exists
    if k in existing_keys:
        with progress_lock:
            pbar.update(1)
        return None
    
    vote = None
    try:
        if annotator == "GPT_5":
            vote = call_with_retry(
                "GPT_5", 
                lambda: openai_vote(openai_client, OPENAI_VOTE_MODEL, prompt)
            )
        elif annotator == "Gemini_2_5_Pro":
            vote = call_with_retry(
                "Gemini_2_5_Pro",
                lambda: gemini_vote(gemini_model, prompt)
            )
        elif annotator == "Claude_Sonnet_4":
            vote = call_with_retry(
                "Claude_Sonnet_4",
                lambda: anthropic_vote(anthro_client, ANTHROPIC_VOTE_MODEL, prompt)
            )
    finally:
        # Always update progress bar
        with progress_lock:
            pbar.update(1)
    
    if vote in ("A", "B"):
        return (annotator, {
            "annotator_type": annotator,
            **base,
            "choice": vote,
            "timestamp": utc_timestamp()
        })
    
    return None

# ------------- Main -------------
def main():
    parser = argparse.ArgumentParser(description="Structured LLM voting (Gemini logic mirrored).")
    parser.add_argument("--limit", type=int, default=None, help="Limit pending comparisons")
    parser.add_argument("--dry-run", action="store_true", help="Plan only (no API / write / push)")
    parser.add_argument("--overwrite-llm", action="store_true", help="Re-annotate existing LLM votes")
    parser.add_argument("--push-interval", type=int, default=50, 
                       help="Push to HF every N annotations (default: 50, 0 = only push at end)")
    args = parser.parse_args()

    secrets = load_secrets()
    open_ai_key = secrets.get("open_ai")
    anthropic_key = secrets.get("anthropic")
    google_key = secrets.get("google")
    hf_token = secrets.get("hf") or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

    if not (open_ai_key and anthropic_key and google_key and hf_token):
        print("Missing required keys/token.")
        sys.exit(1)

    openai_client = OpenAI(api_key=open_ai_key)
    anthro_client = anthropic.Anthropic(api_key=anthropic_key)
    
    gemini_model_obj = GenerativeModel('gemini-2.5-pro')


    existing_df = download_existing()
    required_cols = [
        "annotator_type","source_type","text_hash","text",
        "model_A","model_B","choice",
        "instruction_A","response_A","instruction_B","response_B",
        "timestamp"
    ]
    for c in required_cols:
        if c not in existing_df.columns:
            existing_df[c] = ""

    if existing_df["text_hash"].eq("").any():
        mask = existing_df["text_hash"].eq("") & existing_df["text"].ne("")
        existing_df.loc[mask,"text_hash"] = existing_df.loc[mask,"text"].astype(str).apply(sha1_short)

    existing_long = existing_df[existing_df["annotator_type"].ne("")]
    existing_keys = set(
        f"{r.source_type}||{r.text_hash}||{r.model_A}||{r.model_B}||{r.annotator_type}"
        for r in existing_long.itertuples()
    )
    existing_keys_copy = existing_keys.copy()  # Thread-safe copy

    pairs_df = load_pairs()
    comp_df = build_comparisons(pairs_df)

    pending: List[Dict[str,str]] = []
    overwrite = args.overwrite_llm
    for row in comp_df.itertuples():
        base = {
            "source_type": row.source_type,
            "text_hash": row.text_hash,
            "text": row.text,
            "model_A": row.model_A,
            "model_B": row.model_B,
            "instruction_A": row.instruction_A,
            "response_A": row.response_A,
            "instruction_B": row.instruction_B,
            "response_B": row.response_B,
        }
        if overwrite:
            pending.append(base)
            continue
        have_all = True
        for annot in LLM_ANNOTATORS:
            if vote_key(base, annot) not in existing_keys:
                have_all = False
                break
        if not have_all:
            pending.append(base)

    total_pending = len(pending)
    if args.limit is not None:
        pending = pending[:args.limit]

    print(f"Total comparisons: {len(comp_df)}")
    print(f"Pending needing LLM votes: {total_pending}")
    print(f"Selected this run: {len(pending)} (limit={'none' if args.limit is None else args.limit})")
    if args.push_interval > 0:
        print(f"Will push to HF every {args.push_interval} annotations")

    if args.dry_run:
        print("DRY RUN sample (≤5):")
        for b in pending[:5]:
            print(f"{b['source_type']}|{b['text_hash']}|{b['model_A']}|{b['model_B']}")
        return

    if overwrite and pending:
        comp_keys = {comp_key(b) for b in pending}
        mask_remove = existing_df["annotator_type"].isin(LLM_ANNOTATORS + [AGG_ANNOTATOR]) & \
            existing_df.apply(lambda r: f"{r['source_type']}||{r['text_hash']}||{r['model_A']}||{r['model_B']}" in comp_keys, axis=1)
        removed = int(mask_remove.sum())
        if removed:
            existing_df = existing_df[~mask_remove]
            existing_keys = {
                k for k in existing_keys
                if not any(k.startswith(f"{ck}||") for ck in comp_keys)
            }
            existing_keys_copy = existing_keys.copy()
        print(f"Overwrite removed {removed} existing LLM/aggregate rows.")

    per_llm_stats = {a: {"A":0,"B":0} for a in LLM_ANNOTATORS}
    structured_success = {a:0 for a in LLM_ANNOTATORS}
    structured_fail = {a:0 for a in LLM_ANNOTATORS}
    aggregates_added = 0
    aggregates_skipped = 0
    new_rows: List[Dict[str,str]] = []
    
    # Tracking for incremental pushes
    annotations_since_push = 0
    total_pushes = 0

    # Calculate total tasks for each LLM
    tasks_per_llm = defaultdict(int)
    for base in pending:
        for annot in LLM_ANNOTATORS:
            k = vote_key(base, annot)
            if k not in existing_keys_copy:
                tasks_per_llm[annot] += 1

    # Create progress bars for each LLM
    pbar_gpt = tqdm(total=tasks_per_llm["GPT_5"], desc="GPT-5", unit="votes", position=0)
    pbar_gemini = tqdm(total=tasks_per_llm["Gemini_2_5_Pro"], desc="Gemini 2.5 Pro", unit="votes", position=1)
    pbar_claude = tqdm(total=tasks_per_llm["Claude_Sonnet_4"], desc="Claude Sonnet 4", unit="votes", position=2)
    
    pbar_map = {
        "GPT_5": pbar_gpt,
        "Gemini_2_5_Pro": pbar_gemini,
        "Claude_Sonnet_4": pbar_claude
    }

    # Process all comparisons with parallel LLM calls
    with ThreadPoolExecutor(max_workers=MAX_TOTAL_WORKERS) as executor:
        for base in pending:
            prompt = build_vote_prompt(
                source_text=base["text"],
                model_A=base["model_A"],
                instr_A=base["instruction_A"],
                resp_A=base["response_A"],
                model_B=base["model_B"],
                instr_B=base["instruction_B"],
                resp_B=base["response_B"]
            )

            # Check for existing votes (if not overwriting)
            votes = {}
            if not overwrite:
                for annot in LLM_ANNOTATORS:
                    k = vote_key(base, annot)
                    if k in existing_keys:
                        row_match = existing_df[
                            (existing_df["annotator_type"] == annot) &
                            (existing_df["source_type"] == base["source_type"]) &
                            (existing_df["text_hash"] == base["text_hash"]) &
                            (existing_df["model_A"] == base["model_A"]) &
                            (existing_df["model_B"] == base["model_B"])
                        ]
                        if not row_match.empty:
                            cv = row_match.iloc[0]["choice"]
                            if cv in ("A","B"):
                                votes[annot] = cv

            # Submit all LLM tasks in parallel
            futures = []
            for annot in LLM_ANNOTATORS:
                if annot not in votes:
                    future = executor.submit(
                        process_single_llm_vote,
                        annot,
                        base,
                        prompt,
                        existing_keys_copy,
                        openai_client if annot == "GPT_5" else None,
                        anthro_client if annot == "Claude_Sonnet_4" else None,
                        gemini_model_obj if annot == "Gemini_2_5_Pro" else None,
                        pbar_map[annot]
                    )
                    futures.append(future)

            # Collect results
            for future in as_completed(futures):
                result = future.result()
                if result:
                    annot_type, row_data = result
                    votes[annot_type] = row_data["choice"]
                    per_llm_stats[annot_type][row_data["choice"]] += 1
                    structured_success[annot_type] += 1
                    
                    with lock:
                        new_rows.append(row_data)
                        existing_keys.add(vote_key(base, annot_type))
                        existing_keys_copy.add(vote_key(base, annot_type))
                        annotations_since_push += 1
                else:
                    # Track failures
                    for annot in LLM_ANNOTATORS:
                        if annot not in votes:
                            structured_fail[annot] += 1

            # Check for aggregate
            if all(a in votes for a in LLM_ANNOTATORS):
                agg = majority_three([votes[a] for a in LLM_ANNOTATORS])
                if agg:
                    k_agg = vote_key(base, AGG_ANNOTATOR)
                    if k_agg not in existing_keys:
                        with lock:
                            new_rows.append({
                                "annotator_type": AGG_ANNOTATOR,
                                **base,
                                "choice": agg,
                                "timestamp": utc_timestamp()
                            })
                            existing_keys.add(k_agg)
                            existing_keys_copy.add(k_agg)
                            annotations_since_push += 1
                        aggregates_added += 1
            else:
                aggregates_skipped += 1

            # Check if we should push to HF
            if args.push_interval > 0 and annotations_since_push >= args.push_interval:
                with push_lock:
                    if annotations_since_push >= args.push_interval:  # Double-check after lock
                        # Update dataframe with new rows
                        current_df = pd.concat([existing_df, pd.DataFrame(new_rows)], ignore_index=True)
                        
                        # Reorder columns
                        front = required_cols
                        trailing = [c for c in current_df.columns if c not in front]
                        current_df = current_df[front + trailing]
                        
                        # Push to HF
                        if push_to_hf(current_df, hf_token, 
                                    f"Incremental update: {len(new_rows)} annotations added"):
                            total_pushes += 1
                            print(f"\n[PUSH {total_pushes}] Pushed {len(new_rows)} annotations to HF "
                                  f"(total: {len(current_df)} rows)")
                            # Update existing_df to include new rows
                            existing_df = current_df
                            annotations_since_push = 0

    # Close progress bars
    pbar_gpt.close()
    pbar_gemini.close()
    pbar_claude.close()

    # Final update and push
    if new_rows:
        existing_df = pd.concat([existing_df, pd.DataFrame(new_rows)], ignore_index=True)

    front = required_cols
    trailing = [c for c in existing_df.columns if c not in front]
    existing_df = existing_df[front + trailing]

    tmp = ANNOT_CSV_LOCAL.with_suffix(".tmp.csv")
    existing_df.to_csv(tmp, index=False)
    tmp.replace(ANNOT_CSV_LOCAL)
    print(f"\nSaved updated annotations to {ANNOT_CSV_LOCAL}")

    # Final push if there are remaining annotations or if no incremental pushes were done
    if annotations_since_push > 0 or (args.push_interval == 0 and new_rows):
        if push_to_hf(existing_df, hf_token, 
                     f"Final update: {len(new_rows)} total annotations added"):
            total_pushes += 1
            print(f"Final push: {HF_FILENAME} to HF repo {HF_REPO}")
        else:
            print(f"[WARN] Final push failed")

    print("\n=== Run Summary ===")
    for annot, d in per_llm_stats.items():
        print(f"{annot}: A={d['A']} B={d['B']}")
    print(f"Aggregate rows added: {aggregates_added}")
    print(f"Comparisons lacking 3 votes (no aggregate): {aggregates_skipped}")
    print("Structured success/fail:")
    for a in LLM_ANNOTATORS:
        print(f"  {a}: success={structured_success[a]} fail={structured_fail[a]}")
    print(f"New rows added (incl aggregate): {len(new_rows)}")
    if args.push_interval > 0:
        print(f"Total incremental pushes to HF: {total_pushes}")
    print("Done.")

if __name__ == "__main__":
    main()