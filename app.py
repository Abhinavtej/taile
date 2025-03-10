import os
from flask import Flask, request, render_template, jsonify
import requests
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from dotenv import load_dotenv

def generate_story(user_input, genre, language):
    """Generate a story using Llama-3.2 without pipeline."""
    tokens = word_tokenize(user_input)
    tagged_words = pos_tag(tokens)
    keywords = [word for word, tag in tagged_words if tag in ["NN", "NNS", "NNP", "NNPS", "JJ"]]
    
    prompt = f"""
    You are a creative and skilled storyteller, capable of crafting immersive and engaging narratives.
    
    **Task:**  
    Write a captivating **{genre}** story in **{language}** based on the following key elements:  
    - **Keywords:** {', '.join(keywords)}

    **Guidelines:**  
    - Ensure the story has a **clear beginning, middle, and end**.  
    - Maintain a **coherent flow** and **engaging storytelling style**.  
    - Use **vivid descriptions**, **realistic dialogues**, and **strong character development**.  
    - Keep the story **exciting, emotionally engaging, and original**.  
    - Make it **concise** yet **impactful**, ensuring it fits within the given constraints.

    **Output Format:**  
    Return only the final story without any explanations or unnecessary text.  
    
    Now, begin the story.
    """
    
    headers = {
        'Content-Type': 'application/json',
    }
    
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    
    response = requests.post(
        f"{GEMINI_ENDPOINT}?key={GEMINI_API_KEY}",
        headers=headers,
        json=data
    )
    
    if response.status_code != 200:
        raise Exception(f"Error {response.status_code}: {response.text}")

    response_json = response.json()
    story = response_json['candidates'][0]['content']['parts'][0]['text']
    return story

app = Flask(__name__)

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    """Handle story generation requests."""
    data = request.json
    user_input = data.get("keywords", "").strip()
    genre = data.get("genre", "fiction")
    language = data.get("language", "english")
    
    if not user_input:
        return jsonify({"error": "No keywords provided"}), 400
    
    story = generate_story(user_input, genre, language)
    return jsonify({"story": story})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)