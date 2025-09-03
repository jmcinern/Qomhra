# ab_app_k4_two_page.py
# Two-page Gradio app for open-sourced annotation (Master’s thesis)
# Page 1: consent + annotator type (Learner/Native) + source (Wiki/Oireachtas)
# Page 2: task only (QUESTION_MD + A/B), deterministic K=4 per model pair per source
# Saves: annotator_type, source_type, item info, choice, timestamp

import gradio as gr
import pandas as pd
import time
from itertools import combinations
from pathlib import Path
import json
import hashlib

PAIRS_CSV = "./outputs/pairs.csv"  # columns: run_id, model, source_type, instruction, response, text

# --- Config ---
K = 4
OUT_FILE = "./annotations.csv"
SCHEMA = [
    "annotator_type",   # Learner | Native
    "source_type",      # Wiki | Oireachtas
    "text",
    "model_A",
    "model_B",
    "choice",           # A | B
    "instruction_A",
    "response_A",
    "instruction_B",
    "response_B",
    "timestamp",
]
if not Path(OUT_FILE).exists():
    pd.DataFrame(columns=SCHEMA).to_csv(OUT_FILE, index=False)

pairs_all = pd.read_csv(PAIRS_CSV)

# --- Helpers for deterministic schedule ---
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

    # For each unordered pair, pick k texts deterministically; A/B flips alternate (2/2 over k=4)
    for m1, m2 in combinations(models, 2):
        shared = _shared_texts(df, m1, m2)
        if not shared:
            continue
        keyed = [(_stable_hash(f"{source_type}|{m1}|{m2}|{t}"), t) for t in shared]
        keyed.sort(key=lambda x: x[0])
        ordered_texts = [t for _, t in keyed]

        chosen = []
        idx = 0
        while len(chosen) < k:
            chosen.append(ordered_texts[idx % len(ordered_texts)])
            idx += 1

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

    comps.sort(key=lambda d: (d["source_type"], d["model_A"], d["model_B"], d["text"]))
    return comps

def save_row(annotator_type, item, choice):
    row = {
        "annotator_type": annotator_type,
        "source_type": item["source_type"],
        "text": item["text"],
        "model_A": item["model_A"],
        "model_B": item["model_B"],
        "choice": choice,
        "instruction_A": item["instruction_A"],
        "response_A": item["response_A"],
        "instruction_B": item["instruction_B"],
        "response_B": item["response_B"],
        "timestamp": time.time(),
    }
    pd.DataFrame([row]).to_csv(OUT_FILE, mode="a", header=False, index=False)

QUESTION_MD = (
    "**Question:** Which Question–Answer pair exhibits a stronger command of Irish grammar and "
    "semantic coherence? Take the use of the reference text into account. If unsure, pick the one "
    "with a stronger display of Irish grammar. Choose A or B."
)

CONSENT_MD = f"""
### Irish QA Pair Comparison (Master’s Thesis)

You are invited to take part in a study on Large Language Model Irish-language QA quality.  
By continuing, you consent to the following:

- Your annotations will be **anonymised** (we only record whether you are a **Learner** or **Native speaker**).
- The dataset (reference text + model outputs + your choices) will be released **open-source** for both research and commercial purposes.
- No personal data is collected beyond your level of Irish. You may stop at any time before submission.

- You will answer the following question: 

#### "Which Question–Answer pair exhibits a stronger command of Irish grammar and semantic coherence? Take the use of the reference text into account. If unsure, pick the one with a stronger display of Irish grammar. Choose A or B.". 

- Only base your decision on this question and not other factors.



Please confirm consent, select your annotator type and the source to evaluate, then press **Begin**.
"""

with gr.Blocks() as demo:
    # ---------- PAGE 1: Consent + Role + Source ----------
    with gr.Group(visible=True) as page1:
        gr.Markdown(CONSENT_MD)
        consent_chk = gr.Checkbox(label="I consent to take part and for my anonymised annotations to be open-sourced.", value=False)
        role_dd = gr.Dropdown(["Learner", "Native"], label="Annotator Type (required)", value=None)
        source_dd = gr.Dropdown(["Wiki", "Oireachtas"], label="Source (required)", value=None)
        begin_btn = gr.Button("Begin")
        gate_msg = gr.Markdown()

    # ---------- PAGE 2: Task ----------
    with gr.Group(visible=False) as page2:
        crit = gr.Markdown(QUESTION_MD)
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

    # ---------- State ----------
    annotator_type = gr.State("")   # Learner | Native
    source_state = gr.State(None)   # Wiki | Oireachtas
    comps_state = gr.State([])      # list of dicts
    idx_state = gr.State(0)

    # ---------- Handlers ----------
    def begin(consent, role, source):
        if not consent:
            return ("**Please tick the consent checkbox to proceed.**",
                    gr.update(visible=True), gr.update(visible=False),
                    "", "", "", "", "", "", "", "", "", "", "")
        if role not in ["Learner", "Native"]:
            return ("**Please select your annotator type.**",
                    gr.update(visible=True), gr.update(visible=False),
                    "", "", "", "", "", "", "", "", "", "", "")
        if source not in ["Wiki", "Oireachtas"]:
            return ("**Please select a source (Wikipedia/Oireachtas).**",
                    gr.update(visible=True), gr.update(visible=False),
                    "", "", "", "", "", "", "", "", "", "", "")

        comp_list = build_comparisons_k(source, K)
        if not comp_list:
            return ("**No items found for the selected source.**",
                    gr.update(visible=True), gr.update(visible=False),
                    "", "", "", "", "", "", "", "", "", "", "")

        i = 0
        item = comp_list[i]
        return ("",  # clear gate msg
                gr.update(visible=False), gr.update(visible=True),  # show page2
                f"{i+1} / {len(comp_list)}",
                item["text"], item["instruction_A"], item["response_A"],
                item["instruction_B"], item["response_B"],
                role, source, comp_list, i,
                gr.update(interactive=True), gr.update(interactive=True))

    begin_btn.click(
        begin,
        inputs=[consent_chk, role_dd, source_dd],
        outputs=[
            gate_msg, page1, page2,
            counter, ref_text, instA, respA, instB, respB,
            annotator_type, source_state, comps_state, idx_state,
            btnA, btnB
        ],
    )

    def choose(choice, role, source, comp_list, i):
        role = (role or "").strip()
        if not role or not comp_list:
            return ("**No comparisons loaded.**", gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                    gr.update(interactive=False), gr.update(interactive=False), i)

        item = comp_list[i]
        save_row(role, item, choice)

        i += 1
        if i >= len(comp_list):
            # Done: disable buttons, clear fields, lock progress at max
            return ("**Done — thank you!**",
                    f"{len(comp_list)} / {len(comp_list)}", "", "", "", "",
                    gr.update(interactive=False), gr.update(interactive=False), i)

        nxt = comp_list[i]
        return (f"Saved: {choice}",
                f"{i+1} / {len(comp_list)}",
                nxt["text"], nxt["instruction_A"], nxt["response_A"], nxt["instruction_B"], nxt["response_B"],
                gr.update(interactive=True), gr.update(interactive=True), i)

    btnA.click(
        lambda role, src, comps, i: choose("A", role, src, comps, i),
        inputs=[annotator_type, source_state, comps_state, idx_state],
        outputs=[status, counter, ref_text, instA, respA, instB, respB, btnA, btnB, idx_state],
    )
    btnB.click(
        lambda role, src, comps, i: choose("B", role, src, comps, i),
        inputs=[annotator_type, source_state, comps_state, idx_state],
        outputs=[status, counter, ref_text, instA, respA, instB, respB, btnA, btnB, idx_state],
    )

demo.launch(share=True)
