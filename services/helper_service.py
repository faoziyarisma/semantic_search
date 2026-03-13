import re


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)  # hilangkan newline berlebih
    text = re.sub(r"\n+", " ", text)  # hilangkan line break
    return text.strip()
