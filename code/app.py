from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
import json

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Qdrant configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Initialize Qdrant client
if QDRANT_API_KEY:
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
else:
    qdrant = QdrantClient(host="localhost", port=6333)

# Google Gemini configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash-001"

# Initialize other clients
model = SentenceTransformer("all-MiniLM-L6-v2")
genai_client = genai.Client(api_key=GOOGLE_API_KEY)

# Get available collections
def get_available_collections():
    collections = [c.name for c in qdrant.get_collections().collections]
    return collections

# Perform semantic search
def semantic_search(query_text, collections, limit=5, similarity_threshold=0.0):
    query_vector = model.encode(query_text).tolist()
    all_results = []
    for collection_name in collections:
        try:
            search_results = qdrant.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
                score_threshold=similarity_threshold
            )
            for idx, result in enumerate(search_results):
                formatted_result = {
                    "collection": collection_name,
                    "score": result.score,
                    "text": result.payload.get("text", ""),
                    "metadata": {},
                    "ref_id": f"REF_{idx+1}"
                }
                if "metadata" in result.payload:
                    formatted_result["metadata"] = result.payload["metadata"]
                else:
                    for key in result.payload:
                        if key != "text":
                            formatted_result["metadata"][key] = result.payload[key]
                all_results.append(formatted_result)
        except Exception as e:
            print(f"Error searching collection {collection_name}: {e}")
    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:limit]

# Generate response using Google Gemini
def generate_gemini_response(query, context_chunks, temperature=0.7):
    try:
        formatted_chunks = []
        for idx, chunk in enumerate(context_chunks):
            metadata = chunk.get('metadata', {})
            author = metadata.get('author', 'Unknown')
            filename = metadata.get('filename', '')

            source_info = f"Collection: {chunk['collection']}, Author: {author}, File: {filename}"
            formatted_chunks.append(f"[REF_{idx+1}] Source [{source_info}]: {chunk['text']}")

        formatted_context = "\n\n".join(formatted_chunks)

        system_instruction = "You are a helpful research assistant, assisting in research for blog posts."

        prompt = (
            f"Context:\n{formatted_context}\n\n"
            f"Question: {query}\n\n"
            "Answer the question strictly based on the above context."
        )

        token_response = genai_client.models.count_tokens(
            model=GEMINI_MODEL,
            contents=prompt
        )
        input_token_count = token_response.total_tokens

        response = genai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=system_instruction,
                max_output_tokens=1024,
            ),
        )

        output_token_response = genai_client.models.count_tokens(
            model=GEMINI_MODEL,
            contents=response.text
        )
        output_token_count = output_token_response.total_tokens

        return {
            "text": response.text,
            "token_info": {
                "input_tokens": input_token_count,
                "output_tokens": output_token_count,
                "total_tokens": input_token_count + output_token_count
            }
        }
    except Exception as e:
        return {"text": f"Error generating response: {str(e)}", "token_info": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}}

@app.route('/')
def index():
    collections = get_available_collections()
    return render_template('index.html', collections=collections)

@app.route('/api/query', methods=['POST'])
def query():
    data = request.json
    query_text = data.get('query', '')
    selected_collections = data.get('collections', [])
    chunk_limit = int(data.get('chunk_limit', 5))
    temperature = float(data.get('temperature', 0.7))
    similarity_threshold = float(data.get('similarity_threshold', 0.0))

    if not query_text:
        return jsonify({"error": "Query is required"}), 400

    if not selected_collections:
        return jsonify({"error": "At least one collection must be selected"}), 400

    search_results = semantic_search(
        query_text,
        selected_collections,
        limit=chunk_limit,
        similarity_threshold=similarity_threshold
    )

    gemini_response = generate_gemini_response(query_text, search_results, temperature)

    return jsonify({
        "query": query_text,
        "chunks": search_results,
        "response": gemini_response.get("text", "Error generating response"),
        "token_info": gemini_response.get("token_info", {})
    })

if __name__ == '__main__':
    if not GOOGLE_API_KEY:
        print("WARNING: GOOGLE_API_KEY is not set. Please add it to your .env file.")

    try:
        collections = get_available_collections()
        print(f"Available collections: {collections}")
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")

    app.run(debug=True)


#To run:
#/Users/mason/opt/anaconda3/envs/psai/bin/python /Users/mason/Desktop/Technical_Projects/PYTHON_Projects/ResearchAI/code/app.py