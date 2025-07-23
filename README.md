# Yoruba Roman Numerals Translation System

This project is a fine-tuned sequence-to-sequence model that translates **Roman numerals** into their **Yoruba language representations**, trained on over 6000 samples. It includes both the model training code and a deployed web demo using Flask and Hugging Face.

---

## 🔍 Project Overview

- **Model**: [Davlan/byt5-base-eng-yor-mt](https://huggingface.co/Davlan/byt5-base-eng-yor-mt) fine-tuned on Roman-to-Yoruba numeral pairs.
- **Dataset**: 6,000+ entries mapping Roman numerals to Yoruba equivalents.
- **Goal**: Build an expert system that automatically translates Roman numerals (including extended Unicode variants like `V̅`) into Yoruba language.
- **Approach**: 
  - Fine-tune a pretrained multilingual model (ByT5).
  - Deploy the trained model via Flask as a local web app and Hugging Face Space.

---

## 🚀 Online Demo

- 🧠 Model: [Hugging Face Model Page](https://huggingface.co/Emeritus-21/yorubanumerals-expertsystem)
- 🌍 Live Web App: [Hugging Face Spaces Demo](https://huggingface.co/spaces/Emeritus-21/Yoruba-roman-numerals)

---

## 🧠 Model Training

- Framework: PyTorch + Hugging Face Transformers
- Tokenizer: `AutoTokenizer` from `Davlan/byt5-base-eng-yor-mt`
- Loss: Cross-entropy
- Optimizer: AdamW
- Evaluation: BLEU, ROUGE, and exact match accuracy
- Features:
  - Gradient clipping
  - Early stopping
  - Train-validation split
  - Performance tracking with loss plots

### 📈 Sample Metrics:
- **Validation Accuracy**: ~`XX%` *(Replace with actual value)*
- **BLEU Score**: `X.XXX`
- **ROUGE-L**: `X.XXXX`

---

## 🛠 Local Setup & Usage

### 🔧 Requirements

- Python 3.8+
- PyTorch
- Transformers
- Flask
- Evaluate (for metrics)

### 🔨 Run Locally

1. Clone this repo:
    ```bash
    git clone https://github.com/YOUR_USERNAME/yoruba-roman-numerals.git
    cd yoruba-roman-numerals
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Start the Flask app:
    ```bash
    python app.py
    ```

4. Visit `http://localhost:5000` in your browser.

---

## 📡 API Usage

**Endpoint:** `/convert`

**Method:** `POST`

**Request (JSON):**
```json
{
  "input": "XIV"
}
