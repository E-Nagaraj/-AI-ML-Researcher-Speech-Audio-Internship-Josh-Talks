import torch
import re
import evaluate
from tqdm import tqdm
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Dataset
dataset = load_dataset("google/fleurs", "hi_in")

train_ds = dataset["train"].select(range(200))
test_ds = dataset["test"].select(range(50))

train_ds = train_ds.cast_column("audio", Audio(sampling_rate=16000))
test_ds = test_ds.cast_column("audio", Audio(sampling_rate=16000))

# Model
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="hindi",
    task="transcribe"
)

model.generation_config.suppress_tokens = []

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Preprocess
def prepare_dataset(batch):
    audio = batch["audio"]

    inputs = processor(
        audio["array"],
        sampling_rate=16000,
        return_tensors="pt"
    )

    batch["input_features"] = inputs.input_features[0]

    batch["labels"] = processor.tokenizer(
        batch["transcription"],
        return_tensors="pt",
        padding=True
    ).input_ids[0]

    return batch

train_ds = train_ds.map(prepare_dataset, remove_columns=train_ds.column_names)
test_ds = test_ds.map(prepare_dataset, remove_columns=test_ds.column_names)

print(train_ds[0])

# Inference
def generate_predictions(dataset):
    preds = []
    refs = []

    for sample in tqdm(dataset):
        input_features = torch.tensor(sample["input_features"]).unsqueeze(0).to(device)

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                max_length=225,
                num_beams=1
            )

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        preds.append(transcription)
        refs.append(processor.tokenizer.decode(sample["labels"], skip_special_tokens=True))

    return preds, refs

baseline_preds, refs = generate_predictions(test_ds)

# Cleaning
def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[।!?.,]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def keep_hindi_only(text):
    return re.sub(r"[^\u0900-\u097F\s]", "", text)

cleaned_preds = [
    normalize_text(keep_hindi_only(p))
    for p in baseline_preds
]

cleaned_refs = [
    normalize_text(keep_hindi_only(r))
    for r in refs
]

# Debug samples
for i in range(5):
    print("REF:", refs[i])
    print("PRED:", baseline_preds[i])
    print("-"*50)

# WER
wer_metric = evaluate.load("wer")

cleaned_wer = wer_metric.compute(
    predictions=cleaned_preds,
    references=cleaned_refs
)

print("Final Cleaned WER:", cleaned_wer)

baseline_wer = wer_metric.compute(
    predictions=baseline_preds,
    references=refs
)

print("Baseline WER:", baseline_wer)
print("After cleanup WER:", cleaned_wer)