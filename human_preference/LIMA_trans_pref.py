'''
Overview: create first of its kind instruction-tuning dataset for Irish using Gemini-2.5
1) 800 LIMA/100 Oireachtas/100 Wiki

'''
# Use LIMA for seeding the Oireachtas and Wiki Questions ./LIMA.jsonl
import json
import vertexai
from vertexai.preview.generative_models import GenerativeModel, GenerationConfig
import time
from typing import Dict, List, Optional, Tuple
import random
import os, json, time, random, hashlib, unicodedata, re
from tqdm import tqdm
import argparse
import asyncio


# limit for parsing for testing, then DPO subset, before full trans.
p = argparse.ArgumentParser()
p.add_argument("-n","--num", type=int, help="Max pairs to translate")
args = p.parse_args()


MAX_RETRIES = 2
RETRY_SLEEP_SEC = 2.0
RANDOM_SEED = 42

 # adjust to avoid 429s
CONCURRENCY = 100
async def _one(IRT, sem):
    async with sem:
        return IRT, await gemini_trans(model, IRT, translation_prompt)


random.seed(RANDOM_SEED)

# helpers to allow for deterministic hashing
def normalize_text(s):
    # Trim, collapse newlines a bit, NFC normalize
    s = s.strip()
    s = unicodedata.normalize("NFC", s)
    return s

def stable_hash(instruction, response):
    # Deterministic digest of normalized pair
    instr = normalize_text(instruction)
    resp = normalize_text(response)
    payload = "\x1e".join([instr, resp]).encode("utf-8")  # record-separator join
    return hashlib.sha256(payload).hexdigest()


# translate EN=>GA prompt
translation_prompt =    '''
Translate the following English Instruction and response into Irish. 

- response1 should be a natural, direct and fluent translation,
- response2 should be a weak alternative, it should be  unhelpful, not idiomatic, inaccurate, awkward.

- The contrast in quality of Irish should be very high. 

OUTPUT FORMAT (STRICT):
Return strict JSON with exactly:
{{
"instruction": "<instruction in Irish>",
"response1": "<much better response in Irish>",
"response2": "<much worse response in Irish>"
}}
The following is the English prompt-response pair: 
'''

# force JSON response from gemini
gen_cfg = GenerationConfig(
    response_mime_type="application/json",
    response_schema={
        "type": "OBJECT",
        "properties": {
            "instruction": {"type": "STRING"},
            "response1":    {"type": "STRING"},
            "response2":    {"type": "STRING"},
        },
        "required": ["instruction", "response1", "response2"],
        }
)

# Use LIMA for seeding the Oireachtas and Wiki Questions ./LIMA.jsonl
IRT_ga = []
with open("LIMA.jsonl", "r", encoding="utf-8") as f:
    for ln, raw in enumerate(f, 1):
        raw = raw.strip()
        if not raw:
            continue
        obj = json.loads(raw)                       # parse the JSON line
        conv = obj.get("conversations", [])
        if len(conv) >= 2 and isinstance(conv[0], str):
            prompt, response = conv[0], conv[1]     # two-string format
        elif len(conv) >= 2 and isinstance(conv[0], dict):
            # role-based mirrors (rare): pick text/content/value
            get = lambda d: d.get("value") or d.get("content") or d.get("text") or ""
            prompt, response = get(conv[0]), get(conv[1])
        else:
            raise ValueError(f"Line {ln}: unexpected conversations format")

        IRT_ga.append({"instruction": prompt, "response": response, "hash": stable_hash(prompt, response)})

# to call Google API
async def gemini_trans(model: GenerativeModel, pair_en: dict, prompt: str) -> Optional[str]:
          instruction_en = pair_en.get("instruction", "")
          response_en = pair_en.get("response", "")
          prompt = prompt + "\n\n" + "\n instruction_en: \n" + instruction_en + "\n response_en: \n" + response_en
          try:
              response = await model.generate_content_async(contents=prompt, generation_config=gen_cfg)
              #print(f"Gemini translation response: {response}")
              return response.text or None

          except:
              print(f"Gemini translation failed")
              return None
    

model = GenerativeModel('gemini-2.5-pro')
file_name = "translated_IRT_ga.jsonl"

# allow rerunning of pipeline buy hasing, read with append mode 
already_translated_hashes = []
with open(file_name, "r", encoding="utf-8") as f:
    already_translated_hashes = set()
    for line in f:
        obj = json.loads(line)
        already_translated_hashes.add(obj.get("hash"))

# IRT = IRT - already_translated_hashes
IRT_ga = [irt for irt in IRT_ga if irt.get("hash") not in already_translated_hashes]

gemini_project_id = "gen-lang-client-0817118952" 
gcloud_location = "us-central1"
vertexai.init(project=gemini_project_id, location=gcloud_location)


to_process = IRT_ga[:args.num] if args.num else IRT_ga

async def main():
    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = [_one(IRT, sem) for IRT in to_process]
    with open(file_name, "a", encoding="utf-8") as f:
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            IRT, translated = await fut
            if not translated:
                print("empty response")
                continue
            try:
                obj = json.loads(translated)
                obj["instruction_en"] = IRT["instruction"]
                obj["response_en"] = IRT["response"]
                obj["hash"] = IRT["hash"]
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            except Exception as e:
                print("JSON parse error:", e)
                print("Original response:", translated)

if __name__ == "__main__":
    asyncio.run(main())



