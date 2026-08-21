import re

WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")
CYR_RE = re.compile(r"[А-Яа-яЁё]")


def normalize(text: str) -> str:
    return text.replace("ё", "е").replace("Ё", "Е")


class Lemmatizer:
    """Лемматизация русских слов (pymorphy3) с кэшем; латиница — просто lower()."""

    def __init__(self):
        self._morph = None
        self._cache: dict[str, str] = {}

    def _get_morph(self):
        if self._morph is None:
            import pymorphy3

            self._morph = pymorphy3.MorphAnalyzer()
        return self._morph

    def word(self, w: str) -> str:
        lw = w.lower()
        hit = self._cache.get(lw)
        if hit is not None:
            return hit
        if CYR_RE.search(lw):
            res = self._get_morph().parse(lw)[0].normal_form.replace("ё", "е")
        else:
            res = lw
        self._cache[lw] = res
        return res

    def text(self, text: str) -> str:
        return " ".join(self.word(w) for w in WORD_RE.findall(text))
