from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
CORS(app)

class QuestionSolver:
    def __init__(self):
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        
    def solve_question(self, question, doubt):
        headers = {
            'Authorization': f'Bearer {self.groq_api_key}',
            'Content-Type': 'application/json'
        }
        
        prompt = f"""Solve this academic problem in exactly 500 words or less in markdown format:

Question: {question}
Student's Doubt: {doubt}

Provide:
1. Complete step-by-step solution
2. All relevant formulas
3. Key concepts explained
4. Final answer

Format in clean markdown. Be concise but comprehensive."""

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "model": "llama3-8b-8192",
            "temperature": 0.3,
            "max_tokens": 1000,
            "top_p": 1,
            "stream": False
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except requests.exceptions.RequestException as e:
            return f"API Error: {str(e)}"
        except KeyError as e:
            return f"Response parsing error: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"

solver = QuestionSolver()

@app.route('/solve', methods=['POST'])
def solve_question():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        question = data.get('question', '').strip()
        doubt = data.get('doubt', '').strip()
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
            
        if not doubt:
            return jsonify({'error': 'Doubt is required'}), 400
            
        solution = solver.solve_question(question, doubt)
        
        return jsonify({
            'success': True,
            'solution': solution,
            'question': question,
            'doubt': doubt
        })
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    if not os.getenv('GROQ_API_KEY'):
        print("Warning: GROQ_API_KEY environment variable not set")
    
    app.run(debug=True, host='0.0.0.0', port=3002)