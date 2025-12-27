import os
import sys

# Set PHONEMIZER_ESPEAK_LIBRARY specifically for Windows espeak-ng
os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = r'C:/Program Files/eSpeak NG/libespeak-ng.dll'

from phonemizer import phonemize

def test_phonemize():
    # Checking for stress marks and specific symbols
    test_words = [
        "подсознание",
        "мы",       # should give ɨ
        "шина",     # should give ʂ
        "чаща",     # should give tɕ or ɕ
        "яблоко",
        "хорошо",   # multiple o's with reduction
    ]
    
    for w in test_words:
        p = phonemize(w, language='ru', backend='espeak', strip=False)
        # Replacing trailing space if any
        p = p.strip()
        print(f"Word: '{w}', Phonemes: '{p}'")

if __name__ == "__main__":
    test_phonemize()
