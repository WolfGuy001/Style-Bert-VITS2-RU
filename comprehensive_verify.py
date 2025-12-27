import os
import sys
from unittest.mock import MagicMock

# Set PHONEMIZER_ESPEAK_LIBRARY specifically for Windows espeak-ng
os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = r'C:/Program Files/eSpeak NG/libespeak-ng.dll'

# Add current directory to path
sys.path.append(os.getcwd())

# Mock BERT models
from style_bert_vits2.nlp import bert_models
class SimpleMockTokenizer:
    def __init__(self):
        self.text_words = []
    def __call__(self, text, add_special_tokens=False):
        self.text_words = text.split()
        return {'input_ids': list(range(len(self.text_words)))}
    def convert_ids_to_tokens(self, ids):
        # Map IDs back to the words we split
        return [self.text_words[i] for i in ids]

mock_tokenizer = SimpleMockTokenizer()
bert_models.load_tokenizer = MagicMock(return_value=mock_tokenizer)

from style_bert_vits2.nlp.russian.g2p import g2p
from style_bert_vits2.nlp.symbols import SYMBOLS

test_sentences = [
    "подсознание",
    "Это было очень превосходно и замечательно.",
    "Мы сидели на шине в чаще леса.",
    "Яблоко упало на траву.",
    "Хорошо, когда всё хорошо.",
    "Щегол запел песню.",
    "Цирк уехал, клоуны остались.",
    "Юбка и янтарь.",
    "жук молоко"
]

all_ok = True
all_phones_set = set()

print("Comprehensive G2P and Symbols Check:")
for sent in test_sentences:
    phones, tones, word2ph = g2p(sent)
    print(f"\nText: {sent}")
    print(f"Phones: {phones}")
    print(f"Tones:  {tones}")
    print(f"Sum word2ph: {sum(word2ph)}, Len phones: {len(phones)}")
    
    if sum(word2ph) != len(phones):
        print(f"ERROR: Alignment mismatch in '{sent}'!")
        all_ok = False
        
    missing = [p for p in phones if p not in SYMBOLS]
    if missing:
        print(f"ERROR: Missing symbols: {set(missing)}")
        all_ok = False
    
    # Check if we at least have some 1s in tones (unless sentence is empty/too short)
    if 1 not in tones and len(phones) > 5:
        print(f"WARNING: No stress marks found in '{sent}'!")
        # We won't fail yet, but it's suspicious
    
    for p in phones:
        all_phones_set.add(p)

print("\n--- Summary ---")
if all_ok:
    print("Alignment and Symbols check: PASSED")
else:
    print("Alignment and Symbols check: FAILED")

print(f"All phones used: {sorted(list(all_phones_set))}")
