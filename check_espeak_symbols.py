import os
import subprocess

# Set PHONEMIZER_ESPEAK_LIBRARY specifically for Windows espeak-ng
os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = r'C:/Program Files/eSpeak NG/libespeak-ng.dll'

def get_espeak_ipa(text):
    # Using espeak-ng directly to see exactly what it outputs
    cmd = [r"C:\Program Files\eSpeak NG\espeak-ng.exe", "-v", "ru", "-q", "--ipa", text]
    result = subprocess.run(cmd, capture_output=True, text=False)
    # espeak-ng output might be in some encoding, let's try to decode carefully
    try:
        out = result.stdout.decode('utf-8').strip()
    except:
        out = result.stdout.decode('latin-1').strip()
    return out

test_words = [
    "подсознание",
    "мы",
    "шина",
    "чаща",
    "яблоко",
    "хорошо",
    "замок",
    "замок",
]

print("espeak-ng IPA output:")
for w in test_words:
    ipa = get_espeak_ipa(w)
    # The output might contain (en) or (ru) tags
    print(f"Word: {w} -> IPA: '{ipa}'")

# Also check phonemizer with strip=False
from phonemizer import phonemize

print("\nphonemizer (strip=False) output:")
for w in test_words:
    p = phonemize(w, language='ru', backend='espeak', strip=False).strip()
    print(f"Word: {w} -> Phonemes: '{p}'")
    print(f"Characters: {list(p)}")
