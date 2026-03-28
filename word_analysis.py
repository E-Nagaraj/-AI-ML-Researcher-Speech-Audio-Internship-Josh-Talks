from collections import Counter
import re

def extract_words(texts):
    words = set()
    for text in texts:
        for w in text.split():
            words.add(w)
    return list(words)

def build_freq(refs):
    word_freq = Counter()
    for text in refs:
        for w in text.split():
            word_freq[w] += 1
    return word_freq

def is_hindi(word):
    return bool(re.match(r'^[\u0900-\u097F]+$', word))

def classify_word(word, word_freq):
    freq = word_freq[word]

    if freq > 5 and is_hindi(word):
        return "correct", "high"

    if freq > 2:
        return "correct", "medium"

    return "incorrect", "low"

def run_word_analysis(refs):
    all_words = extract_words(refs)
    word_freq = build_freq(refs)

    results = []
    for word in all_words:
        label, confidence = classify_word(word, word_freq)
        results.append((word, label, confidence))

    low_conf = [r for r in results if r[2] == "low"]

    print("Low confidence words:", len(low_conf))
    print(low_conf[:20])

    return results