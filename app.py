import os
import time
from flask import Flask, request, render_template, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import pinecone
import spacy
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

def get_embedding(text):
    """Generate embedding for given text."""
    vector = embed_model.encode(text).tolist()
    return vector

def upsert_story_vectors(story_id, text, genre, language):
    """Store story embeddings in Pinecone."""
    try:
        vector = get_embedding(text)
        index.upsert(
            vectors=[
                {
                    "id": story_id,
                    "values": vector,
                    "metadata": {
                        "genre": genre,
                        "language": language,
                        "content": text,
                    },
                }
            ]
        )
    except Exception as e:
        print(f"Error upserting to Pinecone: {e}")

def retrieve_relevant_data(keywords, genre, language):
    """Retrieve relevant story context from Pinecone."""
    try:
        query_vector = get_embedding(" ".join(keywords))
        results = index.query(
            vector=query_vector,
            top_k=3,
            include_metadata=True,
            filter={"genre": {"$eq": genre}, "language": {"$eq": language}},
        )
        retrieved_texts = [match["metadata"].get("content", "") for match in results.get("matches", [])]
        return " ".join(retrieved_texts) if retrieved_texts else ""
    except Exception as e:
        print(f"Error retrieving from Pinecone: {e}")
        return ""

def generate_story(user_input, genre, language):
    """Generate a story using Llama-3.2 without pipeline."""
    doc = nlp(user_input)
    keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN", "ADJ"]]
    context = retrieve_relevant_data(keywords, genre, language)
    
    prompt = f"""
    You are a creative and skilled storyteller, capable of crafting immersive and engaging narratives.
    
    **Task:**  
    Write a captivating **{genre}** story in **{language}** based on the following key elements:  
    - **Keywords:** {', '.join(keywords)}  
    - **Context (if relevant):** {context}  

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
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    output_tokens = model.generate(**inputs, max_length=2048, do_sample=True, temperature=0.7)
    story = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    upsert_story_vectors(str(hash(story)), story, genre, language)
    return story

app = Flask(__name__)

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")

nlp = spacy.load("en_core_web_sm")

embed_model = SentenceTransformer("intfloat/multilingual-e5-large")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", token=HF_TOKEN)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables.")

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index_name = "story-generator"

if index_name not in pc.list_indexes().names():
    print("Creating Pinecone index...")
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    time.sleep(10)

index = pc.Index(index_name)

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
    app.run(host="0.0.0.0", port=8000 ,debug=True)