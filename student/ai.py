from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import re
import torch
import logging
import os

# Initialize Flask app
app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes
logging.basicConfig(level=logging.INFO)

# Configuration
MODEL_NAME = "google/flan-t5-small"
MAX_LENGTH = 300
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize AI model
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    math_tutor = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        device=DEVICE
    )
    app.logger.info(f"Model loaded successfully on {DEVICE}")
except Exception as e:
    math_tutor = None
    app.logger.error(f"Model loading failed: {str(e)}")

# Helper functions
def _build_cors_preflight_response():
    response = jsonify({'status': 'preflight'})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")
    return response

def _corsify_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

def validate_equation(equation):
    """Validate equation format"""
    equation = equation.replace(" ", "")
    if not re.match(r'^[\dxX\+\-\=]+$', equation):
        return False
    if equation.count("=") != 1:
        return False
    return True

def solve_linear_equation(equation):
    try:
        if not validate_equation(equation):
            return "Invalid equation format"
            
        equation = equation.replace(" ", "")
        left, right = equation.split("=")
        
        a = b = 0
        
        # Process left side terms
        for term in re.findall(r'[+-]?\d*x?', left):
            if 'x' in term:
                coeff = term.replace('x', '') or '1'
                a += float(coeff)
            else:
                b -= float(term or 0)
        
        # Process right side terms
        for term in re.findall(r'[+-]?\d*x?', right):
            if 'x' in term:
                coeff = term.replace('x', '') or '1'
                a -= float(coeff)
            else:
                b += float(term or 0)
                
        if a == 0:
            return "No solution exists"
            
        solution = round(-b / a, 2)
        return f"x = {solution}"
        
    except Exception as e:
        return f"Error solving equation: {str(e)}"

# Routes
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/')
def home():
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            return render_template_string(f.read())
    except Exception as e:
        app.logger.error(f"Failed to load index.html: {str(e)}")
        return "Application loading failed", 500

@app.route('/explain', methods=['POST', 'OPTIONS'])
def explain_concept():
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()
        
    if not math_tutor:
        return _corsify_response(jsonify({'error': 'AI service unavailable'})), 503
        
    try:
        data = request.get_json()
        if not data or 'query' not in data or not isinstance(data['query'], str):
            return _corsify_response(jsonify({'error': 'Invalid request format'})), 400
            
        prompt = f"""Explain this math concept to a junior high student in Ghana:
                  {data['query']}
                  - Use simple language
                  - Include local examples
                  - Break into short paragraphs"""
                  
        result = math_tutor(prompt, max_length=400)
        return _corsify_response(jsonify({
            'response': result[0]['generated_text'],
            'status': 'success'
        }))
        
    except Exception as e:
        app.logger.error(f"Explanation error: {str(e)}")
        return _corsify_response(jsonify({'error': 'Failed to generate explanation'})), 500

@app.route('/solve', methods=['POST', 'OPTIONS'])
def solve_equation():
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()
        
    try:
        data = request.get_json()
        if not data or 'equation' not in data or not isinstance(data['equation'], str):
            return _corsify_response(jsonify({'error': 'Invalid request format'})), 400
            
        equation = data['equation'].strip()
        
        # First try custom solver
        solution = solve_linear_equation(equation)
        if not solution.startswith("Error"):
            return _corsify_response(jsonify({
                'response': solution,
                'status': 'success'
            }))
            
        # Fallback to AI solver
        if not math_tutor:
            return _corsify_response(jsonify({'error': 'Equation solver unavailable'})), 503
            
        prompt = f"""Solve this equation step-by-step for a student:
                   {equation}
                   - Show each step clearly
                   - Explain the reasoning
                   - Verify the solution"""
                   
        result = math_tutor(prompt, max_length=400)
        return _corsify_response(jsonify({
            'response': result[0]['generated_text'],
            'status': 'success'
        }))
        
    except Exception as e:
        app.logger.error(f"Solving error: {str(e)}")
        return _corsify_response(jsonify({'error': 'Failed to solve equation'})), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)