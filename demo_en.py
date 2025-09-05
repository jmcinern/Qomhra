from transformers import AutoTokenizer
from openai import OpenAI
import re

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)

messages = [
    {"role": "system", "content": "Answer in English and put <|im_end|> at the end of your answer like the following examples."},

    # 1
    {"role": "user", "content": "What is the name of the largest planet in the Solar System?"},
    {"role": "assistant", "content": "Jupiter."},
    # 2
    {"role": "user", "content": "Name a famous poet from Ireland."},
    {"role": "assistant", "content": "W.B. Yeats."},
    # 3
    {"role": "user", "content": "In what year did Irish become an official language of the European Union?"},
    {"role": "assistant", "content": "2007."},
    # 4
    {"role": "user", "content": "How many counties are in the Republic of Ireland?"},
    {"role": "assistant", "content": "26."},
    # 5
    {"role": "user", "content": "What is the longest river in Ireland?"},
    {"role": "assistant", "content": "The Shannon."},
    # 6
    {"role": "user", "content": "Who is the President of Ireland?"},
    {"role": "assistant", "content": "Michael D. Higgins."},
    # 7
    {"role": "user", "content": "What currency is used in the Republic of Ireland?"},
    {"role": "assistant", "content": "The euro."},
    # 8
    {"role": "user", "content": "On what date is St. Patrick’s Day celebrated?"},
    {"role": "assistant", "content": "March 17."},
    # 9
    {"role": "user", "content": "What is the national language of Ireland?"},
    {"role": "assistant", "content": "Irish."},
    # 10
    {"role": "user", "content": "What is the highest mountain in Ireland?"},
    {"role": "assistant", "content": "Carrauntoohil."},

    # Prompt
    {"role": "user", "content": "What is the capital of Ireland?"}
]

#  as in training
prompt = tok.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)

prompt = re.sub(r"<think>.*?</think>\s*", "", prompt, flags=re.DOTALL)

client = OpenAI(base_url="https://enhancement-iraqi-structures-proper.trycloudflare.com/v1", api_key="-")

out = client.completions.create(
    model="qwen3-8B-ga/checkpoint-348",
    prompt=prompt,
    max_tokens=32,
    stop=["<|im_end|>"] 
)

text = out.choices[0].text
print(text)
