from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
import chromadb
from chromadb.utils import embedding_functions
import os
import json
import sys
sys.path.append("./pipeline")
from config import config

app = Flask(__name__)

# Initialize ChromaDB
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=config.sentence_model_path
)
chroma_client = chromadb.PersistentClient(path="db")
collection = chroma_client.get_collection(
    name="all_files", embedding_function=sentence_transformer_ef
)


# Store the last search results
last_search_ids = []

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/Files/<path:path>')
def serve_file(path):
    # Split the path to handle IMG_DIR
    parts = path.split('/')
    if parts[0] == 'IMG_DIR':
        return send_from_directory('Files/IMG_DIR', '/'.join(parts[1:]))
    return send_from_directory('Files', path)

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get('query', '')
    
    results = collection.query(
        query_texts=[query], n_results=config.n_results
    )
    
    global last_search_ids
    last_search_ids = results["ids"][0]
    # Clean each path in the list
    last_search_ids = [path.replace("./FILES", "") for path in last_search_ids]
    
    # Format the response
    slides = []
    for idx, slide_id in enumerate(last_search_ids):
        try:
            print("This is the slide id")
            print(slide_id)
            with open(f"{slide_id}.desc.txt", "r", encoding='latin-1') as f:
                description = f.read()
        except Exception as e:
            description = "No description available"
        
        # Ensure the image URL is relative to the Files directory
        image_path = slide_id.replace('Files/', '')
        slides.append({
            'imageUrl': image_path,
            'description': description,
            #'presentationName': presentation_name,
        })
    
    return jsonify({'slides': slides})

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question')
    
    if not last_search_ids:
        return jsonify({'answer': "Please search for slides first before asking a question."})
    
    # Read descriptions for all found slides
    text_images = []
    for slide_id in last_search_ids:
        try:
            with open(f"{slide_id}.desc.txt", "r", encoding='latin-1') as f:
                text_images.append(f.read())
        except Exception as e:
            continue
    
    # Construct the prompt
    prompt = "Here are descriptions of Images:\n\n"
    for idx, desc in enumerate(text_images):
        prompt += f"Description {idx + 1}:\n{desc.strip()}\n\n"
    prompt += f"Here is the Question:\n{question}\n"
    prompt += "Please answer the question using only the context from the descriptions that are relevant to the question."

    # Generate answer using Llama
    answer = ""
    
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7680, debug=True) 
