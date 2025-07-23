from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re
import unicodedata

# Initialize Flask app
app = Flask(__name__)

# Load model and tokenizer from local path
MODEL_PATH = "yoruba_byt5_trained"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, local_files_only=True)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Regex to validate Roman numerals including V̅
EXTENDED_ROMAN_PATTERN = re.compile(r"^[IVXLCDMV̅]+$", re.IGNORECASE)

def is_roman_numeral(s):
    return bool(EXTENDED_ROMAN_PATTERN.fullmatch(s.strip()))

# Home route
@app.route('/')
def home():
    return render_template("index.html")

# JSON API endpoint
@app.route('/convert', methods=['POST'])
def convert():
    data = request.get_json()
    roman_input = data.get("input", "").strip()
    roman_input = unicodedata.normalize("NFC", roman_input)

    if not roman_input:
        return jsonify({"error": "No input provided"}), 400

    if not is_roman_numeral(roman_input):
        return jsonify({"error": "Please enter a valid Roman numeral (I,V,X,L,C,D,M,V̅ only)"}), 400

    inputs = tokenizer(roman_input, return_tensors="pt", truncation=True, max_length=256).to(device)
    output = model.generate(**inputs, max_length=256)

    result = tokenizer.decode(output[0], skip_special_tokens=True)

    return jsonify({"input": roman_input, "yoruba": result})

# HTML form endpoint
@app.route('/convert-form', methods=['POST'])
def convert_form():
    roman_input = request.form.get("input", "").strip()
    roman_input = unicodedata.normalize("NFC", roman_input)

    if not roman_input:
        return render_template("index.html", error="No input provided.", input_text=roman_input)

    if not is_roman_numeral(roman_input):
        return render_template(
            "index.html",
            error="Please enter a valid Roman numeral (I,V,X,L,C,D,M,V̅ only).",
            input_text=roman_input
        )

    inputs = tokenizer(roman_input, return_tensors="pt", truncation=True, max_length=256).to(device)
    output = model.generate(**inputs, max_length=256)

    result = tokenizer.decode(output[0], skip_special_tokens=True)

    return render_template("index.html", result=result, input_text=roman_input)

# Run app
if __name__ == '__main__':
    app.run(debug=True)
