# ab_app_k4_final.py
# Gradio A/B annotation app
# - Single combined criterion (grammar + semantics)
# - Two sources (Wikipedia, Oireachtas) only
# - Deterministic sampling: EXACTLY K=4 comparisons per model pair per source
# - Deterministic A/B assignment with 2/2 split per pair
# - Single output file: annotations.csv
# - No wrap-around after last item: annotators see "Done" and stop

import gradio as gr
import pandas as pd
import time
from itertools import combinations
from pathlib import Path
import json
import hashlib

PAIRS_CSV = "./outputs/pairs.csv"  # columns: run_id, model, source_type, instruction, response, text


def load_secrets(path="./secrets.json"):
    with open(path, "r", encoding="utf-8") as f:
        # your file is a list with one dict
        return json.load(f)[0]


secrets = load_secrets()
open_ai_key = secrets.get("open_ai")

# Exactly 4 comparisons per model pair per source
K = 4
OUT_FILE = "./annotations.csv"
SCHEMA = [
    "annotator_id",
    "source_type",
    "text",
    "model_A",
    "model_B",
    "choice",
    "instruction_A",
    "response_A",
    "instruction_B",
    "response_B",
    "timestamp",
]
if not Path(OUT_FILE).exists():
    pd.DataFrame(columns=SCHEMA).to_csv(OUT_FILE, index=False)

pairs_all = pd.read_csv(PAIRS_CSV)


def _shared_texts(df, m1, m2):
    t1 = set(df[df["model"] == m1]["text"])
    t2 = set(df[df["model"] == m2]["text"])
    return list(t1 & t2)


def _stable_hash(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16)


def build_comparisons_k(source_type: str, k: int):
    df = pairs_all[pairs_all["source_type"] == source_type].copy()
    if df.empty:
        return []

    models = sorted(df["model"].unique().tolist())
    comps = []

    # For each unordered pair, deterministically pick k texts and fix A/B sides
    for m1, m2 in combinations(models, 2):
        shared = _shared_texts(df, m1, m2)
        if not shared:
            continue
        # Sort shared texts by stable hash of (source|m1|m2|text)
        keyed = []
        for t in shared:
            h = _stable_hash(f"{source_type}|{m1}|{m2}|{t}")
            keyed.append((h, t))
        keyed.sort(key=lambda x: x[0])
        ordered_texts = [t for _, t in keyed]

        # Take first k (cycle deterministically if fewer than k)
        chosen = []
        idx = 0
        while len(chosen) < k:
            chosen.append(ordered_texts[idx % len(ordered_texts)])
            idx += 1

        # Build comparisons with deterministic A/B: even index -> A=m1, odd index -> A=m2
        for j, t in enumerate(chosen):
            r1 = df[(df["model"] == m1) & (df["text"] == t)].iloc[0]
            r2 = df[(df["model"] == m2) & (df["text"] == t)].iloc[0]
            if j % 2 == 0:
                A, B = (m1, r1), (m2, r2)
            else:
                A, B = (m2, r2), (m1, r1)
            comps.append(
                {
                    "source_type": source_type,
                    "text": t,
                    "model_A": A[0],
                    "instruction_A": A[1]["instruction"],
                    "response_A": A[1]["response"],
                    "model_B": B[0],
                    "instruction_B": B[1]["instruction"],
                    "response_B": B[1]["response"],
                }
            )

    # Deterministic overall ordering: by (source_type, model_A, model_B, text)
    comps.sort(key=lambda d: (d["source_type"], d["model_A"], d["model_B"], d["text"]))
    return comps


def save_row(annotator_id, item, choice):
    row = {
        "annotator_id": annotator_id,
        "source_type": item["source_type"],
        "text": item["text"],
        "model_A": item["model_A"],
        "model_B": item["model_B"],
        "choice": choice,  # "A" or "B"
        "instruction_A": item["instruction_A"],
        "response_A": item["response_A"],
        "instruction_B": item["instruction_B"],
        "response_B": item["response_B"],
        "timestamp": time.time(),
    }
    pd.DataFrame([row]).to_csv(OUT_FILE, mode="a", header=False, index=False)


QUESTION_MD = (
    "**Question:** Which Question–Answer pair exhibits a stronger command of Irish grammar and "
    "semantic coherence? take the use of the reference text into account. If unsure, pick the one "
    "with a stronger display of Irish grammar. Choose A or B."
)


with gr.Blocks() as demo:
    gr.Markdown(
        "### Irish QA Pair Comparison\nProvide your name once, then choose a source (Wikipedia or Oireachtas). No ties."
    )

    # Persistent state
    annotator = gr.State("")
    source_state = gr.State(None)  # "Wiki" | "Oireachtas"
    comps_state = gr.State([])  # list of dicts
    idx_state = gr.State(0)

    # Name gate
    with gr.Row():
        name_in = gr.Textbox(
            label="Your Name (required once)", placeholder="e.g., me_01", scale=3
        )
        save_name_btn = gr.Button("Save Name", scale=1)
    name_status = gr.Markdown()

    def save_name(name):
        name = (name or "").strip()
        if not name:
            return "**Please enter your name to proceed.**", ""
        return f"Name saved: **{name}**", name

    save_name_btn.click(save_name, inputs=[name_in], outputs=[name_status, annotator])

    # Start buttons: source only
    gr.Markdown("#### Start: Choose Source")
    with gr.Row():
        wiki_btn = gr.Button("Wikipedia")
        oir_btn = gr.Button("Oireachtas")

    crit = gr.Markdown()
    counter = gr.Markdown()
    ref_text = gr.Textbox(label="Reference Text", interactive=False, lines=8)

    with gr.Row():
        with gr.Column():
            instA = gr.Textbox(label="Instruction A", interactive=False)
            respA = gr.Textbox(label="Response A", interactive=False, lines=8)
        with gr.Column():
            instB = gr.Textbox(label="Instruction B", interactive=False)
            respB = gr.Textbox(label="Response B", interactive=False, lines=8)

    with gr.Row():
        btnA = gr.Button("A is Better")
        btnB = gr.Button("B is Better")
    status = gr.Markdown()

    def _require_name(name):
        return "" if (name or "").strip() else "**Enter your name first.**"

    def start(source):
        comp_list = build_comparisons_k(source, K)
        if not comp_list:
            return (
                "**No items found for selection.**",
                "",
                "",
                "",
                "",
                "",
                "",
                source,
                [],
                0,
            )
        i = 0
        item = comp_list[i]
        return (
            QUESTION_MD,
            item["text"],
            item["instruction_A"],
            item["response_A"],
            item["instruction_B"],
            item["response_B"],
            f"{i+1} / {len(comp_list)}",
            source,
            comp_list,
            i,
        )

    # Wire start (name gate → start)
    wiki_btn.click(lambda n: _require_name(n), inputs=[annotator], outputs=[status]).then(
        lambda: start("Wiki"),
        outputs=[
            crit,
            ref_text,
            instA,
            respA,
            instB,
            respB,
            counter,
            source_state,
            comps_state,
            idx_state,
        ],
        queue=False,
    )
    oir_btn.click(lambda n: _require_name(n), inputs=[annotator], outputs=[status]).then(
        lambda: start("Oireachtas"),
        outputs=[
            crit,
            ref_text,
            instA,
            respA,
            instB,
            respB,
            counter,
            source_state,
            comps_state,
            idx_state,
        ],
        queue=False,
    )

    def choose(choice, name, source, comp_list, i):
        name = (name or "").strip()
        if not name:
            return "**Enter your name first.**", gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), i
        if not comp_list:
            return "**No comparisons loaded.**", gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), i

        item = comp_list[i]
        save_row(name, item, choice)

        i += 1
        if i >= len(comp_list):  # no wrap-around, stop here
            return (
                "**Done — thank you!**",
                "",
                "",
                "",
                "",
                "",
                f"{len(comp_list)} / {len(comp_list)}",
                i,
            )

        nxt = comp_list[i]
        return (
            f"Saved: {choice}",
            nxt["text"],
            nxt["instruction_A"],
            nxt["response_A"],
            nxt["instruction_B"],
            nxt["response_B"],
            f"{i+1} / {len(comp_list)}",
            i,
        )

    btnA.click(
        lambda name, s, cs, i: choose("A", name, s, cs, i),
        inputs=[annotator, source_state, comps_state, idx_state],
        outputs=[status, ref_text, instA, respA, instB, respB, counter, idx_state],
    )
    btnB.click(
        lambda name, s, cs, i: choose("B", name, s, cs, i),
        inputs=[annotator, source_state, comps_state, idx_state],
        outputs=[status, ref_text, instA, respA, instB, respB, counter, idx_state],
    )


demo.launch(share=True)
