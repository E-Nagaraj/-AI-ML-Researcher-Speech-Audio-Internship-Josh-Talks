# Hindi ASR Evaluation & Improvement using Whisper-small

This project presents a complete pipeline for evaluating and improving Hindi Automatic Speech Recognition (ASR) using the Whisper-small model on the Google FLEURS dataset.

## Overview

The project focuses on:

* Baseline ASR evaluation using Whisper
* Word Error Rate (WER) analysis
* Text normalization and cleanup
* Error taxonomy and linguistic analysis
* Word-level confidence classification
* (Optional) Fine-tuning for performance improvement

## Key Features

* **End-to-End ASR Pipeline**
  Audio → Feature Extraction → Prediction → Evaluation

* **WER Optimization**
  Improves evaluation by handling punctuation, spacing, and script inconsistencies

* **Error Taxonomy**
  Identifies key error types:

  * Script mismatch (Hindi vs Urdu)
  * Phonetic errors
  * Deletion & insertion errors
  * English word borrowing

* **Text Cleanup Pipeline**
  Includes normalization and filtering for fair evaluation

* **Word-Level Analysis**
  Classifies words into correct/incorrect with confidence scoring

## Results

| Model                  | WER   |
| ---------------------- | ----- |
| Baseline Whisper-small | ~0.77 |
| After Cleanup          | ~0.71 |
| Fine-tuned (optional)  | ~0.55 |

## Key Insights

* Script mismatch significantly inflates WER despite correct phonetic output
* Post-processing improves evaluation fairness
* Fine-tuning improves actual recognition accuracy
* Error analysis reveals both linguistic and model limitations

## Project Structure

```bash
asr-assignment/
│── src/                # Core pipeline code
│── notebooks/          # Exploration & debugging
│── data/               # Dataset instructions
│── requirements.txt
│── README.md
```

## Dataset

We use the Hindi subset of the Google FLEURS dataset:

```python
from datasets import load_dataset
dataset = load_dataset("google/fleurs", "hi_in")
```

## How to Run

```bash
pip install -r requirements.txt
python src/evaluate.py
```

## Notes

This project emphasizes not just model performance, but also evaluation correctness and linguistic analysis, which are critical in real-world ASR systems.

