# convert_to_requests_jsonl.py
import json, sys, pathlib

SYSTEM_TEXT = ('''Translate the values of 'instruction', 'context', and 'response' into Irish (Gaeilge).
If 'context' is empty, leave it empty. Keep the key name 'category' and its value exactly as in the input.
Output only a valid JSON object with keys: instruction_ga, context_ga, response_ga, category. No extra keys, text, or formatting.'''
)

SCHEMA = {
    "type": "object",
    "properties": {
        "instruction_ga": {"type": "string"},
        "context_ga": {"type": "string"},
        "response_ga": {"type": "string"},
        "category": {"type": "string"},
    },
    "required": ["instruction_ga", "context_ga", "response_ga", "category"],
}

def make_request_line(obj):
    return {
        "request": {
            "systemInstruction": {"role": "system", "parts": [{"text": SYSTEM_TEXT}]},
            "contents": [
                {"role": "user", "parts": [{"text": json.dumps(obj, ensure_ascii=False)}]}
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": SCHEMA,
            },
        }
    }

def convert(in_path, out_path):
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            fout.write(json.dumps(make_request_line(obj), ensure_ascii=False) + "\n")

if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else "./Dolly15K_en.jsonl"
    out = sys.argv[2] if len(sys.argv) > 2 else "./Dolly15K_en_requests.jsonl"
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    convert(inp, out)
    print(f"Wrote {out}")
