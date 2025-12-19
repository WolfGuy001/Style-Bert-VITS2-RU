from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages
from phonemizer import phonemize

def g2p(text: str):
    tokenizer = bert_models.load_tokenizer(Languages.RU)
    tokens = tokenizer.tokenize(text)
    
    phones = []
    tones = []
    word2ph = []
    
    for token in tokens:
        word = token.replace("##", "")

        ps = phonemize(word, language='ru', backend='espeak', strip=True)
        
        curr_phones = list(ps) 
        
        curr_tones = [0] * len(curr_phones)
        
        phones.extend(curr_phones)
        tones.extend(curr_tones)
        word2ph.append(len(curr_phones))
        
    phones = ["_"] + phones + ["_"]
    tones = [0] + tones + [0]
    word2ph = [1] + word2ph + [1]
    
    return phones, tones, word2ph