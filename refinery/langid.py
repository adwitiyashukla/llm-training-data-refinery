import importlib.resources


class LangId:
    def __init__(self, model=None):
        if model is None:
            import fasttext

            path = importlib.resources.files("fast_langdetect") / "resources" / "lid.176.ftz"
            with importlib.resources.as_file(path) as file:
                model = fasttext.load_model(str(file))
        self.model = model

    def __call__(self, text):
        line = text.replace("\n", " ").strip()
        if not line:
            return "", 0.0
        labels, probs = self.model.predict(line, k=1)
        return labels[0].removeprefix("__label__"), min(1.0, float(probs[0]))


ISO3 = {
    "eng": "en", "deu": "de", "fra": "fr", "spa": "es", "rus": "ru", "jpn": "ja", "zho": "zh", "por": "pt",
    "ita": "it", "nld": "nl", "pol": "pl", "tur": "tr", "vie": "vi", "ind": "id", "kor": "ko", "ces": "cs",
    "swe": "sv", "ara": "ar", "fas": "fa", "ukr": "uk", "hun": "hu", "ell": "el", "ron": "ro", "dan": "da",
    "fin": "fi", "nor": "no", "heb": "he", "tha": "th", "hin": "hi", "bul": "bg", "slk": "sk", "cat": "ca",
    "hrv": "hr", "lit": "lt", "srp": "sr", "slv": "sl", "ben": "bn", "lav": "lv", "est": "et", "msa": "ms",
    "tam": "ta", "urd": "ur", "mar": "mr", "tel": "te", "kat": "ka", "aze": "az", "kaz": "kk", "bos": "bs",
    "glg": "gl", "eus": "eu", "isl": "is", "mkd": "mk", "sqi": "sq", "hye": "hy", "fil": "tl", "tgl": "tl",
    "swa": "sw", "afr": "af", "mal": "ml", "kan": "kn", "guj": "gu", "pan": "pa", "nep": "ne", "sin": "si",
    "khm": "km", "mya": "my", "mon": "mn", "uzb": "uz", "bel": "be", "lat": "la", "epo": "eo", "cym": "cy",
    "gle": "ga", "nob": "no", "nno": "nn", "zsm": "ms", "lao": "lo", "amh": "am", "som": "so", "yor": "yo",
}


def from_header(header):
    first = header.split(",")[0].strip().lower()
    return ISO3.get(first, first)
