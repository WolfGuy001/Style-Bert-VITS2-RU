import re
from num2words import num2words

def normalize_text(text: str) -> str:
    # 1. Раскрытие чисел
    text = re.sub(r"\d+", lambda x: num2words(int(x.group(0)), lang='ru'), text)
    # 2. Унификация тире и кавычек
    text = text.replace("—", "-").replace("–", "-")
    text = text.replace("«", '"').replace("»", '"').replace("„", '"').replace("“", '"')
    return text