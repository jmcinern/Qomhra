import choix
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, HfFileSystem
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# for displaying matrices
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# load annotations
HF_REPO = "jmcinern/Irish_Prompt_Response_Human_Feedback"
F_NAME_ANNNOTATIONS = "annotations_Wiki_Native.csv"#"LLM_K20.csv"

SCHEMA = [
    "annotator_type",   # Learner | Native | GPT-4o
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

# create comparisons of form (winner, loser), expected for choix Bradley-Terry

def load_existing(filename):
    api = HfApi()
    fs = HfFileSystem()
    path = f"datasets/{HF_REPO}/{filename}"
    local_path = api.hf_hub_download(repo_id=HF_REPO, filename=filename, repo_type="dataset")
    return pd.read_csv(local_path)

# get set of models
df_annotations = load_existing(F_NAME_ANNNOTATIONS)

# print number of rows in df 
print("#rows:", df_annotations.shape[0])
def dedup(df):
    # Remove duplicate rows based on all columns except timestamp and choice: same annotation => diff time/choice
    return df.drop_duplicates(subset=df.columns.difference(["timestamp", "choice"]))

df_annotations = dedup(df_annotations)

print("#rows (after deduplication):", df_annotations.shape[0])

models_set = df_annotations["model_A"].unique()
n_models = len(models_set)

# map model names to ints, (choix comparison expects this)
model_to_id = {model: i for i, model in enumerate(models_set)}
# ids to names
id_to_model = {i: model for model, i in model_to_id.items()}

# per annotator, model win proba matrix 
prob_mats = {}
model_labels = [id_to_model[i] for i in range(n_models)]

annotator_coutns = {}
# per annotator
for annotator, df_group in df_annotations.groupby("annotator_type"):
    annotator_coutns[annotator] = df_group.shape[0]
    if annotator == "Tester":
        continue
    print(f"\n\nAnnotator: {annotator}")
    comparisons = []
    # for each row store: model_A, model_B, choice
    for _, row in df_group.iterrows():
        model_A = row["model_A"]
        model_B = row["model_B"]
        choice = row["choice"]
        if choice == "A":
            comparisons.append((model_to_id[model_A], model_to_id[model_B]))
        else:
            comparisons.append((model_to_id[model_B], model_to_id[model_A]))


    # fit model [(ID, score)]
    model_ID_scores = choix.opt_pairwise(n_models, comparisons)

    # Convert skill scores into a ranking
    ranking = np.argsort(-model_ID_scores)
    print("\nRanking of models (best to worst):")
    for i in ranking:
        print(f"{id_to_model[i]}: {model_ID_scores[i]:.3f}")

    print(f"\n\n{annotator}: Pairwise win probabilities (rows = winner vs. columns):")

    # Collect scores into array in model order
    theta = np.array([model_ID_scores[i] for i in range(n_models)])

    # Build probability matrix
    prob_matrix = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                prob_matrix[i, j] = np.nan
            else:
                diff = theta[i] - theta[j]
                prob_matrix[i, j] = 1 / (1 + np.exp(-diff))

    # Wrap in DataFrame for readability
    prob_df = pd.DataFrame(prob_matrix, index=[id_to_model[i] for i in range(n_models)],
                        columns=[id_to_model[i] for i in range(n_models)])
    print(prob_df.round(3))
    prob_mats[annotator] = prob_df.reindex(index=model_labels, columns=model_labels)



'''
print("\n\nAnnotation counts by annotator type:")
for annotator, count in annotator_coutns.items():
    print(f"Annotator: {annotator}, Count: {count}")
'''


'''
# 3 matrices (inter-model win probabilities)
plot_annotators = [a for a in ["Native", "Learner", "GPT-4o"] if a in prob_mats]
if plot_annotators:
    n = len(plot_annotators)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for k, a in enumerate(plot_annotators):
        cbar = (k == n - 1)  # single colorbar on the last panel
        sns.heatmap(prob_mats[a].astype(float).round(3),
                    vmin=0, vmax=1, cmap="viridis", annot=True, fmt=".2f",
                    linewidths=.5, square=True, cbar=cbar,
                    cbar_kws={"label": "Win probability"}, ax=axes[k])
        axes[k].set_title(f"Win Probabilities — {a}")
        axes[k].set_xlabel("Opponent model")
        axes[k].set_ylabel("Winner model")
        axes[k].tick_params(axis="x", rotation=45, labelbottom=True)
    fig.suptitle("Inter-Model Win Probabilities by Annotator", y=1.02)
    plt.show()
'''


# Cohen's Kappa, align comparison
print(df_annotations["annotator_type"].unique())

# --- Function to compute Cohen's Kappa + confusion matrix for two annotators ---
def pairwise_agreement(df, annotator1, annotator2):
    # Canonical item_id (sorted models so A/B order doesn’t matter)
    def make_item_id(row):
        lo, hi = sorted([row["model_A"], row["model_B"]])
        return row["text"] + "||" + lo + "||" + hi

    df = df.copy()
    df["item_id"] = df.apply(make_item_id, axis=1)

    # Map choices into canonical frame (so choice always means "lo" vs "hi")
    def canon_choice(row):
        lo, hi = sorted([row["model_A"], row["model_B"]])
        winner = row["model_A"] if row["choice"] == "A" else row["model_B"]
        return "A" if winner == lo else "B"

    df["choice_canon"] = df.apply(canon_choice, axis=1)

    # Keep only the two annotators
    dfa = df[df["annotator_type"] == annotator1][["item_id", "choice_canon"]]
    dfb = df[df["annotator_type"] == annotator2][["item_id", "choice_canon"]]

    # Resolve duplicates (keep last occurrence per item_id)
    dfa = dfa.drop_duplicates(subset=["item_id"], keep="last").set_index("item_id")
    dfb = dfb.drop_duplicates(subset=["item_id"], keep="last").set_index("item_id")

    # Align only on common items
    aligned = dfa.join(dfb, how="inner", lsuffix=f"_{annotator1}", rsuffix=f"_{annotator2}")
    if aligned.empty:
        print(f"\nNo overlapping annotations between {annotator1} and {annotator2}")
        return None, None

    a = aligned[f"choice_canon_{annotator1}"]
    b = aligned[f"choice_canon_{annotator2}"]

    # Compute Cohen's Kappa
    kappa = cohen_kappa_score(a, b)

    # Confusion matrix
    labels = ["A", "B"]
    cm = confusion_matrix(a, b, labels=labels)
    cm_df = pd.DataFrame(cm,
                         index=[f"{annotator1} chose {l}" for l in labels],
                         columns=[f"{annotator2} chose {l}" for l in labels])

    return kappa, cm_df


annotators = ["Native", "Learner", "Aggregate_LLM"]
kappa_matrix = pd.DataFrame(index=annotators, columns=annotators, dtype=float)

for i, a1 in enumerate(annotators):
    for j, a2 in enumerate(annotators):
        if i == j:
            kappa_matrix.loc[a1, a2] = 1.0
        else:
            kappa, _ = pairwise_agreement(df_annotations, a1, a2)
            val = np.nan if kappa is None else kappa
            kappa_matrix.loc[a1, a2] = val
            kappa_matrix.loc[a2, a1] = val

print("\nInter-annotator Cohen's Kappa matrix:")
print(kappa_matrix.round(3))


# Plot heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(kappa_matrix.astype(float).round(3),
            annot=True, cmap="coolwarm", vmin=-1, vmax=1,
            linewidths=.5, square=True)
#plt.title("Inter-Annotator Agreement (Cohen's Kappa)")
plt.show()

