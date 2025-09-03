from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm

# Load the dataset
ds = load_dataset("ReliableAI/Irish-Text-Collection")

# Thread-safe set and lock
unique_sources = set()

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

UCC_dict = {'gaparacrawl': [], 'culturax': [], 'gawiki': [], 'glot500': [], "unknown": []}


for item in tqdm(ds['train']):
    item_id = item['id']
    try:
        source, _ = item_id.split(':')
        UCC_dict[source].append(item['text'])
    except ValueError:
        UCC_dict["unknown"].append(item['text'])
        continue

for key in UCC_dict:
    with open(f"./data/UCC_{key}.txt", "w", encoding="utf-8") as f:
            SEP = "<|endoftext|>"
            string_with_sep = SEP.join(UCC_dict[key])
            print(string_with_sep[:1000])
            f.write(string_with_sep) 