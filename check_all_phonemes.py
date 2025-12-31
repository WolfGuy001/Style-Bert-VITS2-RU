from style_bert_vits2.nlp.russian.g2p import g2p
from tqdm import tqdm
import os

esd_path = r"Data\Ruslan\esd.list"
if not os.path.exists(esd_path):
    print(f"Error: {esd_path} not found")
    exit(1)

unique_symbols = set()
with open(esd_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Checking {len(lines)} lines...")
for line in tqdm(lines):
    parts = line.strip().split("|")
    if len(parts) >= 4:
        text = parts[3]
        try:
            phones, _, _ = g2p(text)
            for p in phones:
                unique_symbols.add(p)
        except Exception as e:
            # print(f"Error on line: {text}\n{e}")
            pass

print("Unique symbols found:")
print(sorted(list(unique_symbols)))
