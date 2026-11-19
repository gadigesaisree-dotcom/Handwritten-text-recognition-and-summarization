

import re
from spellchecker import SpellChecker


class TextPostProcessor:
    CORRECTION_FREQ_THRESHOLD = 1e-7   # tunable

    def __init__(self):
        self.spell = SpellChecker()



    def process(self, text: str) -> str:
        """Full post-processing pipeline: normalise → correct → return."""
        norm = self._normalize_text(text)
        corrected = self._correct_spelling(norm)
        return corrected

    
    def _normalize_text(self, text: str) -> str:
        """Strip junk characters from OCR output without over-filtering."""
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^A-Za-z0-9\s\.,'\"\?!\-:;/()]", "", text)
        return text.strip()

    def _correct_spelling(self, text: str) -> str:
        """Conservative spell-correction that preserves proper nouns / numbers."""
        words = text.split()
        corrected_words = []

        misspelled = self.spell.unknown(words)

        for word in words:
            core = word.strip(".,!?\"';:()")

            if (
                not core
                or not core.isalpha()
                or core[0].isupper()          # proper noun heuristic
                or len(core) > 20
                or core.lower() not in misspelled
            ):
                corrected_words.append(word)
                continue

            correction = self.spell.correction(core.lower())
            if correction and correction != core.lower():
                freq = self.spell.word_frequency.dictionary.get(correction, 0)
                if freq >= self.CORRECTION_FREQ_THRESHOLD:
                    word = word.replace(core, correction)

            corrected_words.append(word)

        return " ".join(corrected_words)



if __name__ == "__main__":
    processor = TextPostProcessor()
    tests = [
        "The quick brwn f0x jmped over the lazy d0g.",
        "Dr. Smith visitd New York on Tuesdy for a confrence.",
        "Artifical inteligence has transfrmd modern computng.",
    ]
    for t in tests:
        print("IN :", t)
        print("OUT:", processor.process(t))
        print()
