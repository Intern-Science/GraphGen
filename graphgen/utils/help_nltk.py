import os
from typing import Dict, List, Optional
import nltk
import jieba

resource_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources")


class NLTKHelper:
    _stopwords: Dict[str, Optional[List[str]]] = {
        "english": None,
        "chinese": None,
    }

    def __init__(self):
        jieba.initialize()

    def get_stopwords(self, lang: str) -> List[str]:
        nltk.data.path.append(os.path.join(resource_path, "nltk_data"))
        if self._stopwords[lang] is None:
            try:
                nltk.data.find("corpora/stopwords")
            except LookupError:
                nltk.download("stopwords", download_dir=os.path.join(resource_path, "nltk_data"))

            self._stopwords[lang] = nltk.corpus.stopwords.words(lang)
        return self._stopwords[lang]

    @staticmethod
    def word_tokenize(text: str, lang: str) -> List[str]:
        if lang == "zh":
            return jieba.lcut(text)
        nltk.data.path.append(os.path.join(resource_path, "nltk_data"))
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", download_dir=os.path.join(resource_path, "nltk_data"))

        return nltk.word_tokenize(text)
