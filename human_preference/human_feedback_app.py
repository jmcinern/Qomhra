import gradio as gr
import pandas as pd
import json
import random
from datetime import datetime
from pathlib import Path
import os

from huggingface_hub import HfApi, hf_hub_download, create_repo
try:
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    # For older versions of huggingface_hub
    class HfHubHTTPError(Exception):
        pass

# --- Configuration ---
# Source data file containing instructions and responses
TRANSLATED_FILE = "translated_IRT_ga.jsonl"
# Local and remote filename for annotations
ANNOTATION_FILE = "DPO_annotations.csv"
# Hugging Face Hub details
HF_REPO_ID = "jmcinern/DPO_ga" # Your HF repo ID

HF_TOKEN = os.getenv("HF_TOKEN")

# Deterministic sampling settings
NUM_SAMPLES = 100
RANDOM_SEED = 42

# --- UI Content ---
CONSENT_MD = """
### Irish QA Pair Comparison (Master’s Thesis)

You are invited to take part in a study on Large Language Model Irish-language QA quality.
By continuing, you consent to the following:

- Your annotations are anonymised.
- The dataset (reference text + model outputs + your choices) will be released **open-source** for both research and commercial purposes.
- No personal data is collected. You may stop at any time.

- You will answer the following question:

#### Which answer, A or B, is better in terms of grammar, naturalness, and coherence?

- Only base your decision on this question and not other factors.

Please confirm consent, select your role, then press **Begin**.
"""

# --- Helper Functions ---
def load_master_samples() -> list:
    """Loads, shuffles deterministically, and returns the first 100 samples."""
    if not Path(TRANSLATED_FILE).exists():
        raise FileNotFoundError(f"Source file not found: {TRANSLATED_FILE}")
    with open(TRANSLATED_FILE, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # Shuffle with a fixed seed to get a deterministic "random" subset
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(data)
    return data[:NUM_SAMPLES]

def download_annotations() -> pd.DataFrame:
    """Downloads annotations from HF. If not found, returns an empty DataFrame."""
    try:
        local_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=ANNOTATION_FILE,
            repo_type="dataset",
            token=HF_TOKEN,
        )
        print(f"Downloaded existing annotations from {HF_REPO_ID}")
        return pd.read_csv(local_path)
    except HfHubHTTPError as e:
        # If the file doesn't exist on the Hub (404), it's the first run.
        if e.response.status_code == 404:
            print("No remote annotation file found. Creating a new one.")
            # Define the schema for the new CSV file, now including annotator_type
            return pd.DataFrame(columns=["hash", "annotator_type", "choice", "preferred_response", "timestamp"])
        else:
            raise  # Re-raise other HTTP errors

def upload_annotations(df: pd.DataFrame):
    """Saves a DataFrame locally and pushes it to the Hugging Face Hub."""
    if not HF_TOKEN:
        print("WARNING: No HF_TOKEN found. Skipping upload.")
        return

    # Save locally first
    df.to_csv(ANNOTATION_FILE, index=False)

    # Upload to Hub
    api = HfApi()
    create_repo(HF_REPO_ID, repo_type="dataset", exist_ok=True, token=HF_TOKEN)
    api.upload_file(
        path_or_fileobj=ANNOTATION_FILE,
        path_in_repo=ANNOTATION_FILE,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN,
        commit_message="Append new DPO annotation"
    )
    print(f"Successfully uploaded updated annotations to {HF_REPO_ID}")


# --- Gradio Core Logic ---

def prepare_tasks():
    """
    Loads master samples, downloads existing annotations, and prepares the
    list of un-annotated tasks for the current session.
    """
    master_samples = load_master_samples()
    annotations_df = download_annotations()
    completed_hashes = set(annotations_df['hash'].unique())

    to_do_samples = [s for s in master_samples if s['hash'] not in completed_hashes]

    tasks = []
    for sample in to_do_samples:
        # Shuffle response1 and response2 for unbiased presentation
        options = [('response1', sample['response1']), ('response2', sample['response2'])]
        random.shuffle(options)

        tasks.append({
            "hash": sample['hash'],
            "instruction": sample['instruction'],
            "response_A": options[0][1],
            "response_B": options[1][1],
            # Track which original response corresponds to A and B
            "shuffle_map": {'A': options[0][0], 'B': options[1][0]}
        })
    return tasks

def start_session(annotator_type):
    """
    Triggered by the 'Begin' button. Prepares tasks and loads the first one.
    """
    tasks = prepare_tasks()
    if not tasks:
        # All samples are already annotated
        return {
            consent_group: gr.update(visible=False),
            task_group: gr.update(visible=False),
            done_group: gr.update(visible=True),
            state_tasks: [],
            state_task_index: 0,
            state_annotator_type: ""
        }

    first_task = tasks[0]
    progress_str = f"Progress: 1 / {len(tasks)}"

    return {
        consent_group: gr.update(visible=False),
        task_group: gr.update(visible=True),
        done_group: gr.update(visible=False),
        state_tasks: tasks,
        state_task_index: 0,
        state_annotator_type: annotator_type,
        progress_counter: gr.update(value=progress_str),
        instruction_box: gr.update(value=first_task['instruction']),
        response_a_box: gr.update(value=first_task['response_A']),
        response_b_box: gr.update(value=first_task['response_B']),
    }

def record_choice(tasks, current_index, annotator_type, choice):
    """
    Records the user's choice, saves it, and loads the next task.
    """
    # 1. Get current task and determine which original response was preferred
    current_task = tasks[current_index]
    preferred_response_key = current_task['shuffle_map'][choice] # 'response1' or 'response2'

    # 2. Create a new annotation row, now including the annotator_type
    new_annotation = {
        "hash": current_task['hash'],
        "annotator_type": annotator_type,
        "choice": choice, # 'A' or 'B'
        "preferred_response": preferred_response_key,
        "timestamp": datetime.utcnow().isoformat()
    }

    # 3. Load existing annotations, append, and upload
    annotations_df = download_annotations()
    new_df = pd.concat([annotations_df, pd.DataFrame([new_annotation])], ignore_index=True)
    upload_annotations(new_df)

    # 4. Move to the next task
    next_index = current_index + 1
    if next_index >= len(tasks):
        # All tasks for this session are done
        return {
            task_group: gr.update(visible=False),
            done_group: gr.update(visible=True)
        }

    next_task = tasks[next_index]
    progress_str = f"Progress: {next_index + 1} / {len(tasks)}"

    return {
        state_task_index: next_index,
        progress_counter: gr.update(value=progress_str),
        instruction_box: gr.update(value=next_task['instruction']),
        response_a_box: gr.update(value=next_task['response_A']),
        response_b_box: gr.update(value=next_task['response_B']),
    }

def update_begin_button_status(consent_given, role_selected):
    """Enable the begin button only if consent is checked and a role is selected."""
    return gr.update(interactive=(consent_given and role_selected is not None))


# --- Gradio UI Layout ---

with gr.Blocks(theme=gr.themes.Soft(), title="DPO Annotation") as demo:
    # State management
    state_tasks = gr.State([])
    state_task_index = gr.State(0)
    state_annotator_type = gr.State("")

    # Page 1: Consent
    with gr.Group(visible=True) as consent_group:
        gr.Markdown(CONSENT_MD)
        with gr.Row():
            consent_checkbox = gr.Checkbox(label="I consent to the terms above")
            annotator_type_dropdown = gr.Dropdown(["Tester", "Native"], label="Select Your Role")
        begin_btn = gr.Button("Begin", interactive=False)

    # Page 2: Annotation Task
    with gr.Group(visible=False) as task_group:
        progress_counter = gr.Markdown("Progress: 0 / 0", elem_id="progress_counter")
        with gr.Column():
            instruction_box = gr.Textbox(label="Instruction", interactive=False, lines=3)
            with gr.Row():
                response_a_box = gr.Textbox(label="Answer A", interactive=False, lines=8)
                response_b_box = gr.Textbox(label="Answer B", interactive=False, lines=8)
            with gr.Row():
                choose_a_btn = gr.Button("A is Better", variant="primary")
                choose_b_btn = gr.Button("B is Better", variant="primary")

    # Page 3: Completion Message
    with gr.Group(visible=False) as done_group:
        gr.Markdown("## ✅ Thank You!\n\nAll available samples have been annotated. Your contribution is greatly appreciated.")


    # --- Event Handlers ---

    # Enable 'Begin' button only when consent is checked AND a role is selected
    consent_checkbox.change(
        fn=update_begin_button_status,
        inputs=[consent_checkbox, annotator_type_dropdown],
        outputs=begin_btn
    )
    annotator_type_dropdown.change(
        fn=update_begin_button_status,
        inputs=[consent_checkbox, annotator_type_dropdown],
        outputs=begin_btn
    )

    # Start the session when 'Begin' is clicked
    begin_btn.click(
        fn=start_session,
        inputs=[annotator_type_dropdown],
        outputs=[
            consent_group, task_group, done_group,
            state_tasks, state_task_index, state_annotator_type,
            progress_counter, instruction_box, response_a_box, response_b_box
        ]
    )

    # Handle choice A
    choose_a_btn.click(
        fn=record_choice,
        inputs=[state_tasks, state_task_index, state_annotator_type, gr.State('A')],
        outputs=[
            state_task_index, progress_counter,
            instruction_box, response_a_box, response_b_box,
            task_group, done_group
        ]
    )

    # Handle choice B
    choose_b_btn.click(
        fn=record_choice,
        inputs=[state_tasks, state_task_index, state_annotator_type, gr.State('B')],
        outputs=[
            state_task_index, progress_counter,
            instruction_box, response_a_box, response_b_box,
            task_group, done_group
        ]
    )

if __name__ == "__main__":
    # Ensure the source file exists before launching
    if not Path(TRANSLATED_FILE).exists():
        print(f"FATAL: Source data file '{TRANSLATED_FILE}' not found.")
        print("Please ensure the file is in the correct directory before running.")
    else:
        demo.launch()