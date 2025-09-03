# Overview: practice python script to get familiar with libraries required for continued pre-trainiing
# txt -> tokenizer -> chunking -> trainer (CLM) (with datacollator (for batching)) -> model
# librsaries:
from transformers import DataCollatorForLanguageModeling, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TrainerCallback, TrainerControl
from transformers.trainer_utils import get_last_checkpoint
from datasets import Dataset, DatasetDict, load_from_disk #concatenate_datasets
#import torch
from sklearn.model_selection import train_test_split
import os
import math
import torch
import wandb
import random

print("Running the script")

 
model_size = "0.6"
model_test_name = "1156_FULL_DATASET_8_CPU-"+model_size+"B-CPT_ga_wandb_tests"
cache_path = "./cache/qwen3-"+model_size+"B"
model_name = "Qwen/Qwen3-"+model_size+"B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_path,
    trust_remote_code=True, 
    torch_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_path,
    trust_remote_code=True  
)

model.resize_token_embeddings(len(tokenizer))


wandb_api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=wandb_api_key)
import wandb


LR = 1e-4
config = {
    "model_size": f"{model_size}B",
    "epochs": 2,
    "learning_rate": LR,  
}
wandb.init(
    project="train-CPT",
    name=model_test_name,
    config=config,
    tags=["test-run", "qwen3", "irish", "deepspeed", "multi-gpu"]
)

if torch.cuda.is_available():
    print("CUDA is available!")
    print("Number of GPUs:", torch.cuda.device_count())
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available.")


# 1. read file
# 2. chunk
# 3. split
# 4. -> dataset
# 5. tokenize
def file_to_chunks(file_path, chunk_size=1000):
    # 1. read file
    with open(file_path, "r", encoding="utf-8") as f:
        file_text = f.read()
        file_words = file_text.split()#[:1_000_000]
        
    # 2. chunk
    chunks = [" ".join(file_words[i:i+chunk_size])
            for i in range(0, len(file_words), chunk_size)]
    
    # 3. split
    train, tmp = train_test_split(chunks, test_size=0.06, random_state=42, shuffle=True)

    # test and val set - 3% each 
    test, val = train_test_split(tmp, test_size=0.5, random_state=42, shuffle=True)
    
    # 4. -> dataset
    dataset = DatasetDict({
    "train": Dataset.from_dict({"text": train}),
    "validation": Dataset.from_dict({"text": val}),
    "test": Dataset.from_dict({"text": test}),
    })

    # 5. tokenize
    # b) helper function
    def tokenize_function(raw_chunk):
        return tokenizer(raw_chunk['text'])
    
    # c) tokenize dataset 
    dataset_tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"], num_proc=16)

    # tokenized -> model input size blocks
    block_size = 2048 

    # turns batch into chunks of block_sizea
    def group_texts(examples):
        # convert list of lists into a single list
        concatenated = sum(examples["input_ids"], [])
        # calculate max number of tokens given block size.
        total = len(concatenated) // block_size * block_size
        # cut up list by block size
        input_chunks = [concatenated[i:i+block_size] for i in range(0, total, block_size)]
        # need to have labels for the dataset batching 
        return {"input_ids": input_chunks, "labels": input_chunks} 

    
    # apply the function to the tokenized dataset
    dataset_chunks = dataset_tokenized.map(group_texts, 
                                                        batched=True, 
                                                        # attn padding not important for CPT
                                                        remove_columns=["attention_mask"],
                                                        num_proc=16
                                                        )
    return dataset_chunks
    
def shuffle(chunks_list):
    """Shuffle list of chunks in place and return"""
    shuffled = chunks_list.copy()  # Don't modify original
    random.shuffle(shuffled)
    return shuffled

def create_dataset_from_chunks(chunks_list, test_size=0.06, random_state=42):
    """Convert list of chunks back to DatasetDict format"""
    # Split shuffled chunks
    train, tmp = train_test_split(chunks_list, test_size=test_size, random_state=random_state, shuffle=True)
    test, val = train_test_split(tmp, test_size=0.5, random_state=random_state, shuffle=True)
    
    # Create DatasetDict
    dataset = DatasetDict({
        "train": Dataset.from_dict({"input_ids": train, "labels": train}),
        "validation": Dataset.from_dict({"input_ids": val, "labels": val}),
        "test": Dataset.from_dict({"input_ids": test, "labels": test}),
    })
    
    return dataset
    
def get_or_prepare_dataset(cache_path, chunk_size=10_000):
     # Check if already process and load from disk if so
    if os.path.exists(cache_path):
        print("Loading data set from disk")
        return load_from_disk(cache_path)
    # otherwise prepare dataset, tokenization, chunking etc...
    else:
        print("Preparing and processing dataset")
        # data dir with english and irish data
        data_dir = "./data/"
        # get file names in ./data/ from os
        data_file_paths = [f for f in os.listdir(data_dir)]

        # get bitext for initial alignment (parallel corpus)
        bitext_file = [f for f in data_file_paths if 'bitext' in f.lower()]
        bitext_path = os.path.join(data_dir, bitext_file[0])
        bitext_dataset = file_to_chunks(bitext_path, chunk_size=chunk_size)

        bitext_chunks = []
        bitext_chunks.extend(bitext_dataset['train']['input_ids'])
        bitext_chunks.extend(bitext_dataset['validation']['input_ids'])
        bitext_chunks.extend(bitext_dataset['test']['input_ids'])

        # other files
        other_files = [f for f in data_file_paths if 'bitext' not in f.lower()]
        all_chunks = []
        for f_name in other_files:
            print(f"Processing file: {f_name}")
            f_path = os.path.join(data_dir, f_name)
            file_chunks = file_to_chunks(f_path, chunk_size=chunk_size)
            # get all data
            all_chunks.extend(file_chunks['train']['input_ids'])
            all_chunks.extend(file_chunks['validation']['input_ids'])
            all_chunks.extend(file_chunks['test']['input_ids'])

        # mix files to prevent sequential training
        shuffled_mixed_chunks = shuffle(all_chunks)

        # propend bitext data to the mixed dataset
        combined_chunks = bitext_chunks + shuffled_mixed_chunks

        final_dataset = create_dataset_from_chunks(combined_chunks)
        final_dataset.save_to_disk(cache_path)
        print('Dataset saved')

        return final_dataset


dataset_path = "./cache/datset_processed"
final_dataset = get_or_prepare_dataset(dataset_path, chunk_size=10_000)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_path,
    trust_remote_code=True, 
    torch_dtype=torch.float16
)


# set up trainer with data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # CLM (autoregressive) 
)


training_args = TrainingArguments(
    learning_rate=LR,
    output_dir="./checkpoints/",
    overwrite_output_dir=True,
    num_train_epochs=2,
    save_steps=1,
    per_device_eval_batch_size=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,#gradient_checkpointing=True, # trick to save subsection of forward pass, prevents caching if True.
    logging_steps=1,
    do_eval= True,
    eval_strategy="steps",
    eval_steps=1,
    save_total_limit=2,
    prediction_loss_only=True,
    fp16=True,
    #report_to="wandb",  # enable wandb/hub
    deepspeed="./ds_config.json", # deepspeed config
    gradient_checkpointing=True, # trick to save subsection of forward pass, prevents caching if True.
)


# evaluate on the test set
def log_test_metrics_to_wandb(dataset, trainer):
    test_metrics = trainer.evaluate(eval_dataset=dataset['test'])

    if test_metrics:
        wandb.log({
            "final_test_loss": test_metrics.get("eval_loss"),
            "final_test_perplexity": test_metrics.get("eval_perplexity", math.exp(test_metrics.get("eval_loss", 0))),
        })

class ForceWandbLogging(TrainerCallback):
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs is not None:
            print(f"FORCING WANDB LOG: {logs}")  # Debug print
            # Force log everything to wandb
            wandb.log(logs, step=state.global_step)# set up training arguments

# pick up where bitext left off with mixed monolingual en and ga data. 
trainer = Trainer(
model=model,
args=training_args,
train_dataset=final_dataset['train'],
eval_dataset=final_dataset['validation'],
data_collator=data_collator,
callbacks=[ForceWandbLogging()] 
)


last_ckpt = get_last_checkpoint(training_args.output_dir)
if last_ckpt:
    print("Resuming from checkpoint:", last_ckpt)
    trainer.train(resume_from_checkpoint=True)
else:
    print("No checkpoint found, starting fresh training.")
    trainer.train(resume_from_checkpoint=False) # first run (False is default)
    
log_test_metrics_to_wandb(final_dataset, trainer)


wandb.finish()

# save the model
trainer.save_model("./checkpoints/"+model_test_name)

# CODE GRAVEYARD
'''''''''
class StopAfterFirstCheckpointCallback(TrainerCallback):
    def __init__(self):
        self.checkpoint_saved = False

    def on_save(self, args, state, control, **kwargs):
        if not self.checkpoint_saved:
            self.checkpoint_saved = True
            print("First checkpoint saved. Stopping training...")
            control.should_training_stop = True
        return control
'''''''''

# Explicitly log to WandB as report_to as trainer arg not working as expected (no train/val logs)



