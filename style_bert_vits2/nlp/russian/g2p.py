from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages
from phonemizer import phonemize
import re

def g2p(text: str):
    tokenizer = bert_models.load_tokenizer(Languages.RU)
    # Get tokens and their mapping to words/original text
    encoded = tokenizer(text, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'])
    
    # We need to group tokens by "word" to phonemize them together
    # BERT tokens for the same word usually start with '##' (except the first one)
    # or follow the tokenizer's rules. rubert-base-cased uses '##' for subwords.
    
    word_tokens_groups = []
    if tokens:
        current_group = [tokens[0]]
        for token in tokens[1:]:
            if token.startswith("##"):
                current_group.append(token)
            else:
                word_tokens_groups.append(current_group)
                current_group = [token]
        word_tokens_groups.append(current_group)
    
    all_phones = []
    all_tones = []
    word2ph = []
    
    for group in word_tokens_groups:
        # Reconstruct word
        word = "".join([t.replace("##", "") for t in group])
        
        # Phonemize the whole word with strip=False to get stress marks
        # CRITICAL: with_stress=True is needed for Russian espeak to show stress ˈ
        ps = phonemize(word, language='ru', backend='espeak', strip=False, with_stress=True).strip()
        
        # Remove espeak-ng voice tags like (en) or (ru) if they appear
        ps = re.sub(r'\(.*?\)', '', ps)
        
        curr_phones = []
        curr_tones = []
        
        # Parse result for stress
        # espeak stress mark is ˈ (U+02C8)
        i = 0
        while i < len(ps):
            if ps[i] == 'ˈ':
                # Next character is stressed
                i += 1
                if i < len(ps):
                    curr_phones.append(ps[i])
                    curr_tones.append(1)
            elif ps[i] == 'ˌ':
                # Ignore secondary stress or mark as 0, but don't add to phones
                pass 
            elif ps[i] == ' ' or ps[i] == '\t':
                # Skip spaces if they escaped strip
                pass
            else:
                curr_phones.append(ps[i])
                curr_tones.append(0)
            i += 1
            
        all_phones.extend(curr_phones)
        all_tones.extend(curr_tones)
        
        # Alignment: Assign all phonemes to the first token of the word, 0 to others
        word2ph.append(len(curr_phones))
        for _ in range(len(group) - 1):
            word2ph.append(0)
            
    # Add start/end tokens
    all_phones = ["_"] + all_phones + ["_"]
    all_tones = [0] + all_tones + [0]
    word2ph = [1] + word2ph + [1]
    
    return all_phones, all_tones, word2ph