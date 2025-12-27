import os
import sys

# Set PHONEMIZER_ESPEAK_LIBRARY specifically for Windows espeak-ng
os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = r'C:/Program Files/eSpeak NG/libespeak-ng.dll'

# Add current directory to path
sys.path.append(os.getcwd())

# Mock BERT models before importing g2p
from unittest.mock import MagicMock
from style_bert_vits2.nlp import bert_models

class MockTokenizer:
    def __call__(self, text, add_special_tokens=False):
        # Very simple splitter to mock Rubert behavior
        # "подсознание" -> ["под", "##созна", "##ние"]
        if text == "подсознание":
            ids = [1, 2, 3]
        elif text == "превосходно":
            ids = [4, 5]
        elif text == "мы шина чаща":
            ids = [6, 7, 8]
        else:
            ids = [9]
        return {'input_ids': ids}
    
    def convert_ids_to_tokens(self, ids):
        if ids == [1, 2, 3]: return ["под", "##созна", "##ние"]
        if ids == [4, 5]: return ["превосход", "##но"]
        if ids == [6, 7, 8]: return ["мы", "шина", "чаща"]
        return ["word"]

mock_tokenizer = MockTokenizer()
bert_models.load_tokenizer = MagicMock(return_value=mock_tokenizer)

from style_bert_vits2.nlp.russian.g2p import g2p
from style_bert_vits2.nlp.symbols import SYMBOLS

def test_g2p_case(text):
    print(f"\nTesting: '{text}'")
    phones, tones, word2ph = g2p(text)
    print(f"Phones: {phones}")
    print(f"Tones:  {tones}")
    print(f"Word2ph: {word2ph}")
    
    # Check if all phones are in SYMBOLS
    missing = [p for p in phones if p not in SYMBOLS]
    if missing:
        print(f"WARNING: Missing symbols in RU_SYMBOLS: {set(missing)}")
    else:
        print("All symbols are present in RU_SYMBOLS.")
        
    return phones, tones, word2ph

if __name__ == "__main__":
    # Case 1: Complex word with potential for subword reduction mismatch
    test_g2p_case("подсознание")
    
    # Case 2: Word with stress
    test_g2p_case("превосходно")
    
    # Case 3: Word with special symbols (ы, ш, ч)
    test_g2p_case("мы шина чаща")
    
    # Ensure word2ph sums up correctly (excluding start/end tokens)
    # Actually word2ph entries correspond to BERT tokens.
    # We should verify it matches what tokenizer gives.
