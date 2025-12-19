import re
from num2words import num2words

def normalize_text(text: str) -> str:
    # Пример простой нормализации
    # 1. Раскрытие чисел
    text = re.sub(r"\d+", lambda x: num2words(int(x.group(0)), lang='ru'), text)
    # 2. Замена пунктуации (если нужно)
    return text