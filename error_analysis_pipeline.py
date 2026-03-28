import re
from collections import Counter
from asr_assignment.error_analysis import ErrorTaxonomyBuilder, systematic_sample

def is_urdu(text):
    return bool(re.search(r"[\u0600-\u06FF]", text))

def is_devanagari(text):
    return bool(re.search(r"[\u0900-\u097F]", text))

def classify_advanced(case):
    ref = case.reference
    hyp = case.hypothesis

    if is_urdu(hyp) and is_devanagari(ref):
        return "script-mismatch"

    if case.error_type == "english-borrowing":
        return "english-borrowing"

    if case.error_type == "deletion":
        return "deletion"
    if case.error_type == "insertion":
        return "insertion"

    return "phonetic-error"

def run_error_analysis(refs, preds):
    rows = []
    for i in range(len(refs)):
        rows.append({
            "sample_id": str(i),
            "reference": refs[i],
            "hypothesis": preds[i]
        })

    builder = ErrorTaxonomyBuilder()
    cases = builder.build_cases(rows)

    sampled_cases = systematic_sample(cases, sample_size=25)

    labels = []
    for case in sampled_cases:
        labels.append(classify_advanced(case))

    print(Counter(labels))

    return sampled_cases, labels