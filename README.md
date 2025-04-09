


# WAR REPORTER
## RAG-Based Summarization with DeepSeek-AI & Sentence Transformers

[![Watch the video](https://img.youtube.com/vi/YJuqR8Gn9oY/maxresdefault.jpg)](https://www.youtube.com/watch?v=YJuqR8Gn9oY)

## Overview

This project focuses on Retrieval-Augmented Generation (RAG) summarization using `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` and `sentence-transformers/all-mpnet-base-v2`. Additionally, it includes text-to-speech conversion using `gtts`.

## Key Technologies

* **RAG Summarization:** Implements Retrieval-Augmented Generation for text summarization.
* **DeepSeek AI Model:** Utilizes `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` for high-quality summarization.
* **Text-to-Speech:** Converts generated summaries into speech using `gtts`.

## Demo Video

Click the thumbnail above or watch here: [YouTube Video](https://www.youtube.com/watch?v=YJuqR8Gn9oY)
##   Technical Approach

* **CUDA 12.4 Setup:**
    * Instructions for downgrading from CUDA 12.5 to 12.4 to ensure compatibility.
* **Torch Compilation:**
    * Building PyTorch for CUDA 12.4 to ensure compatibility.
* **Model Usage:**
    * DeepSeek-R1-Distill-Qwen-1.5B for high-quality summarization.
    * all-mpnet-base-v2 for efficient text retrieval.
* **Text-to-Speech:**
    * Generating speech output from summarized text using gtts.

##   Project Description

This project showcases the integration of advanced AI models and real-time text summarization using RAG techniques.

[Image of the project or a relevant diagram]

(You can replace the above line with an actual image link if you have one)

## Model Explanation
### DeepSeek-R1-Distill-Qwen-1.5B vs. Facebook/BART-Large-CNN

| Feature                      | DeepSeek-R1-Distill-Qwen-1.5B | Facebook/BART-Large-CNN |
|------------------------------|--------------------------------|-------------------------|
| **Model Size**               | 1.5B parameters (Distilled)  | 406M parameters        |
| **Performance**              | Faster and more efficient    | Slightly slower        |
| **Summarization Quality**    | More coherent and contextual | Can sometimes be redundant |
| **Context Handling**         | Handles longer contexts better | Struggles with very long documents |
| **Computational Efficiency** | Lower resource requirements | Requires more compute power |

- **DeepSeek-R1-Distill-Qwen-1.5B** is a distilled version of a large transformer-based model optimized for summarization. It provides **better coherence**, **more relevant summarization**, and **handles long-form documents efficiently**.
- **Facebook/BART-Large-CNN** is a well-known transformer-based summarizer but can sometimes produce **generic or repetitive summaries**, especially for longer contexts.

Given these advantages, DeepSeek-R1-Distill-Qwen-1.5B is preferred in this pipeline to ensure high-quality, efficient, and meaningful summaries.

## Installation
```bash
pip install gtts pypdf transformers torch
```

## Usage
```python
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from gtts import gTTS

# Convert JPG to PDF
def jpg_to_pdf(image_path, pdf_path):
    image = Image.open(image_path)
    image.save(pdf_path, "PDF")

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    images = convert_from_path(pdf_path)
    text = "\n".join([pytesseract.image_to_string(img) for img in images])
    return text

# Summarization using DeepSeek
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def summarize_text(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(**inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Convert text to speech
def text_to_speech(text, output_path):
    tts = gTTS(text)
    tts.save(output_path)

# Example Usage
image_path = "sample.jpg"
pdf_path = "output.pdf"
jpg_to_pdf(image_path, pdf_path)
text = extract_text_from_pdf(pdf_path)
summarized_text = summarize_text(text)
text_to_speech(summarized_text, "output.mp3")
```
## **Sentence Transformer Architecture**
![senetnce transfoemer](https://github.com/user-attachments/assets/8519b318-9a37-44bc-aac5-52b251a1b949)


## Conclusion
This pipeline automates the process of extracting text from images, summarizing it efficiently using `DeepSeek-R1-Distill-Qwen-1.5B`, and converting it into speech using `gTTS`, making it useful for accessibility applications and content consumption.
