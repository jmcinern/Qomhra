from transformers import AutoTokenizer
from openai import OpenAI
import re

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)

messages = [
		{"role": "system", "content": "Answer in Irish and put <|im_end|> at the end of your answer like the following examples."},

        # 1
        {"role": "user", "content": "Cén t-ainm atá ar an bpláinéad is mó sa Ghrianchóras?"},
        {"role": "assistant", "content": "Iúpatar."},
        # 2
        {"role": "user", "content": "Luaigh file cáiliúil as Éirinn."},
        {"role": "assistant", "content": "W.B. Yeats."},
        # 3
        {"role": "user", "content": "Cén bhliain a tháinig an Ghaeilge chun bheith ina teanga oifigiúil san Aontas Eorpach?"},
        {"role": "assistant", "content": "2007."},
        # 4
        {"role": "user", "content": "Cé mhéad contae atá i bPoblacht na hÉireann?"},
        {"role": "assistant", "content": "26."},
        # 5
        {"role": "user", "content": "Cén abhainn is faide in Éirinn?"},
        {"role": "assistant", "content": "An tSionainn"},
        # 6
        {"role": "user", "content": "Cé hé Uachtarán na hÉireann?"},
        {"role": "assistant", "content": "Mícheál D. Ó hUiginn."},
        # 7
        {"role": "user", "content": "Cén airgeadra a úsáidtear sa Phoblacht?"},
        {"role": "assistant", "content": "An euro."},
        # 8
        {"role": "user", "content": "Cén dáta a cheiliúrtar Lá Fhéile Pádraig?"},
        {"role": "assistant", "content": "17 Márta."},
        # 9
        {"role": "user", "content": "Céard í teanga náisiúnta na hÉireann?"},
        {"role": "assistant", "content": "An Ghaeilge."},
        # 10
        {"role": "user", "content": "Cén sliabh is airde in Éirinn?"},
        {"role": "assistant", "content": "Corrán Tuathail."},

        # Prompt
        {"role": "user", "content": "Cad í príomhchathair na hÉireann?"}
]

# Render EXACTLY as in training
prompt = tok.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
prompt = re.sub(r"<think>.*?</think>\s*", "", prompt, flags=re.DOTALL)


client = OpenAI(base_url="https://enhancement-iraqi-structures-proper.trycloudflare.com/v1", api_key="-")

out = client.completions.create(
    model="qwen3-8B-ga/checkpoint-348",
    prompt=prompt,
    max_tokens=32,
    stop=["<|im_end|>"]  # match the template’s turn-end token
)

text = out.choices[0].text
text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
print(text)

