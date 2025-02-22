import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
import os
import pdfplumber

load_dotenv()

app = Flask(__name__)

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

model = genai.GenerativeModel('gemini-1.5-flash')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

# Function to shorten text if needed
def shorten_text(text, max_tokens=4096):
    """Shortens text while preserving key details using Google Gemini API."""
    if len(text.split()) <= max_tokens:
        return text  # If text is within limit, return as is

    # Prompt for Gemini to summarize while keeping key details
    prompt = f"""
    Summarize the following document while keeping all key information such as names, dates, technical terms, and important insights intact.
    Keep it concise but meaningful. Limit to {max_tokens} tokens.

    Document:
    {text}
    """

    try:
        response = genai.generate_text(prompt=prompt)
        summarized_text = response.text.strip()
        return summarized_text
    except Exception as e:
        print("Error summarizing text:", e)
        return text  # Return original text if Gemini fails

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    pdf_path = f"./uploads/{file.filename}"
    file.save(pdf_path)
    
    text = extract_text_from_pdf(pdf_path)
    context = shorten_text(text)
    
    os.remove(pdf_path)
    print(f"Deleted file: {pdf_path}")
    
    return jsonify({"message": "File uploaded", "context": context})

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.json
    question = data.get("question")
    context = data.get("context")
    
    if not question or not context:
        return jsonify({"error": "Missing question or context"}), 400
    
    prompt = f"Based on this document: {context}\nAnswer this question : {question}"
    response = model.generate_content(prompt)
    
    return jsonify({"answer": response.text})

if __name__ == "__main__":
    app.run(debug=True)
