
from flask import Flask, request, jsonify, render_template
import os, requests
from dotenv import load_dotenv
from datetime import datetime

# Load API key
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

app = Flask(__name__, template_folder='.')

# OpenRouter API endpoint
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '').lower()

    
    if "date" in user_message:
        bot_response = f"Today's date is {datetime.now().strftime('%B %d, %Y')}."
    elif "time" in user_message:
        bot_response = f"The current time is {datetime.now().strftime('%H:%M:%S')}."
    elif "add" in user_message or "subtract" in user_message or "multiply" in user_message or "divide" in user_message:
        try:
            # WARNING: only use eval with safe input in production
            bot_response = str(eval(user_message.replace("x", "*")))
        except:
            bot_response = "Sorry, I couldn't calculate that."
    else:
        # Fallback to AI model
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "openrouter/free",
            "messages": [{"role": "user", "content": user_message}],
            "temperature": 0.7,
            "max_tokens": 300
        }
        try:
            response = requests.post(OPENROUTER_URL, headers=headers, json=data)
            result = response.json()
            bot_response = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not bot_response:
                bot_response = "Sorry, I couldn't process that."
        except Exception as e:
            print("Error:", e)
            bot_response = "Sorry, I couldn't process that."

    return jsonify({"reply": bot_response})

if __name__ == "__main__":
    app.run(debug=True)