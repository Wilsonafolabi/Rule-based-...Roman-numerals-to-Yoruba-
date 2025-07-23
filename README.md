Yoruba Roman Numerals Expert System 🇳🇬🔢
This project is an expert system that translates Roman numerals (e.g., I, V, X, XIV, V̅M, up to 6000) into their Yoruba text equivalents using a fine-tuned transformer model. The model is based on google/byt5-small and fine-tuned with a dataset of Roman numerals and their Yoruba translations. The project includes a Flask-based web application for easy interaction via a web interface or API, and is integrated with Resend for sending email notifications of conversion results.
Features

Translates Roman numerals (including extended notation like V̅ for thousands) into Yoruba text.
Supports Roman numerals up to 6000.
Provides a web interface for user input and a JSON API for programmatic access.
Validates input to ensure only valid Roman numerals are processed.
Sends conversion results via email using the Resend API.
Deployed on Hugging Face Spaces for easy access and testing.

Model Details

Model: Fine-tuned from google/byt5-small.
Dataset: Custom dataset (Yorubanumbers1-6000.csv) containing Roman numerals and their Yoruba translations.
Training: Fine-tuned using PyTorch with the Davlan/byt5-base-eng-yor-mt tokenizer, optimized with AdamW, gradient clipping, and early stopping.
Evaluation Metrics:
BLEU Score: Measures translation quality.
ROUGE Score: Evaluates text similarity.
Exact Match Accuracy: Ensures precise translations.


Model Size: 582M parameters.
Hosted on: Hugging Face Model Hub

Demo
Try the live demo on Hugging Face Spaces.
Installation

Clone the repository:
git clone https://github.com/your-username/yoruba-roman-numerals.git
cd yoruba-roman-numerals


Install dependencies:
pip install -r requirements.txt


Download the fine-tuned model and tokenizer from Hugging Face or use the local yoruba_byt5_trained directory if already downloaded.

Ensure you have PyTorch and CUDA (optional, for GPU support) installed.

Set up the Resend API:

Obtain a Resend API key from Resend.
Create a .env file in the project root and add your Resend API key:RESEND_API_KEY=your_resend_api_key





Usage
Running the Flask App

Start the Flask application:
python app.py


Access the web interface at http://localhost:5000 or use the API endpoint at http://localhost:5000/convert.


Example API Usage
Send a POST request to the /convert endpoint:
curl -X POST http://localhost:5000/convert -H "Content-Type: application/json" -d '{"input": "V̅M", "email": "user@example.com"}'

Response:
{
  "input": "V̅M",
  "yoruba": "ẹgbẹ̀rún márùn-ún",
  "email_status": "Conversion result sent to user@example.com"
}

Example Code Usage
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import requests

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("yoruba_byt5_trained")
model = AutoModelForSeq2SeqLM.from_pretrained("yoruba_byt5_trained")

# Example input
roman_input = "V̅M"
inputs = tokenizer(roman_input, return_tensors="pt", truncation=True, max_length=256)
outputs = model.generate(**inputs, max_length=256)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Input: {roman_input}, Yoruba: {result}")

# Send result via Resend API
resend_api_key = "your_resend_api_key"
response = requests.post(
    "https://api.resend.com/emails",
    headers={"Authorization": f"Bearer {resend_api_key}"},
    json={
        "from": "your-app@resend.dev",
        "to": "user@example.com",
        "subject": "Yoruba Roman Numeral Conversion",
        "text": f"Input: {roman_input}\nYoruba: {result}"
    }
)
print("Email sent:", response.json())

Project Structure
yoruba-roman-numerals/
│
├── app.py                    # Flask application for web interface and API
├── yoruba_byt5_trained/      # Directory containing the fine-tuned model and tokenizer
├── Yorubanumbers1-6000.csv   # Dataset used for training
├── requirements.txt          # Python dependencies
├── templates/
│   └── index.html            # HTML template for web interface
├── .env                      # Environment file for Resend API key
└── README.md                 # Project documentation

Training Details

Dataset: Yorubanumbers1-6000.csv contains Roman numerals and their Yoruba translations.
Tokenizer: Davlan/byt5-base-eng-yor-mt for tokenizing inputs and targets.
Training Setup:
Batch size: 8
Learning rate: 5e-5
Epochs: Up to 25 with early stopping (patience=3)
Optimizer: AdamW with weight decay (0.01)
Gradient clipping: max_norm=1.0


Evaluation:
Training and validation loss plotted for monitoring.
Metrics: BLEU, ROUGE, and exact match accuracy calculated on validation set.



Dependencies

transformers
torch
flask
pandas
tqdm
matplotlib
evaluate
rouge_score
requests
python-dotenv

Install them using:
pip install transformers torch flask pandas tqdm matplotlib evaluate rouge_score requests python-dotenv

Contributing
Contributions are welcome! Please open an issue or submit a pull request with improvements or bug fixes.
License
This project is licensed under the MIT License.
Acknowledgments

Built with Hugging Face Transformers.
Fine-tuned using Google Colab.
Email functionality powered by Resend.
Thanks to the Yoruba language community for inspiration and support.
