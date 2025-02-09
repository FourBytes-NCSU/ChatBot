from flask import Flask, render_template, request, jsonify
from flask_cors import CORS  # Enable CORS for embedding in an iframe
import requests
import os, re, random, json
from datetime import datetime, timedelta

# Google Calendar API imports
from google.oauth2 import service_account
from googleapiclient.discovery import build

# LangChain and related imports for the RAG setup
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Import for custom LLM
from langchain.llms.base import LLM
from pydantic import Field
from typing import Optional, List

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['SECRET_KEY'] = 'supersecretkey'

# Global variable to store available food listings
available_food_cache = []
# Global cache for QA responses
qa_cache = {}

# Updated main server URL (your working ngrok URL)
MAIN_SERVER_URL = "https://8ac2-152-7-255-197.ngrok-free.app"

##############################
# Google Calendar Setup
##############################
SCOPES = ['https://www.googleapis.com/auth/calendar']
SERVICE_ACCOUNT_FILE = 'service-account.json'  # Ensure this file is in your project root
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
# Use your specific calendar ID here instead of "primary"
calendar_service = build('calendar', 'v3', credentials=credentials)

##############################
# Custom Ollama LLM Implementation using Pydantic fields
##############################
class OllamaLLM(LLM):
    model: str = Field(default="llama3.1")
    temperature: float = Field(default=0.7)

    @property
    def _llm_type(self) -> str:
        return "ollama"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
        }
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            raise Exception(f"Error calling Ollama API: {response.text}")
        try:
            lines = response.text.strip().split("\n")
            output = ""
            for line in lines:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        output += obj.get("response", "")
                    except Exception as e:
                        print(f"Error parsing line: {line}\nError: {e}")
            return output
        except Exception as e:
            raise Exception(f"Error processing NDJSON from Ollama API. Response text: {response.text}") from e

##############################
# RAG (Retrieval-Augmented Generation) Setup using multiple PDF documents
##############################
def setup_rag():
    pdf_folder = "docs"
    all_documents = []
    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(pdf_folder, filename)
            try:
                loader = PyPDFLoader(filepath)
                documents = loader.load()
                all_documents.extend(documents)
                print(f"Loaded {len(documents)} documents from {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    if not all_documents:
        raise ValueError("No PDF documents loaded. Please check your docs folder and PDF files.")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(all_documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embeddings)
    # Limit retriever results for faster responses (return only top 3 documents)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=OllamaLLM(model="llama3.1", temperature=0.7),
        chain_type="stuff",
        retriever=retriever
    )
    return qa_chain

qa_chain = setup_rag()

##############################
# Helper function to fetch and filter food listings by locality
##############################
def fetch_food_listings(locality: str):
    try:
        resp = requests.get(f"{MAIN_SERVER_URL}/order/available-food", timeout=10, verify=False)
    except Exception as e:
        return None, f"Request error: {str(e)}"
    if resp.status_code != 200:
        return None, f"Could not fetch food listings (status code: {resp.status_code})"
    try:
        food_list = resp.json()
    except Exception as e:
        try:
            food_list = json.loads(resp.text)
        except Exception as e:
            return None, f"Failed to parse food listings JSON: {str(e)}"
    # Use substring matching (case-insensitive) to filter by locality
    filtered_food = [food for food in food_list if locality.lower() in food.get('city', '').lower()]
    print("filtered_food", filtered_food)
    return filtered_food, None

##############################
# Endpoints
##############################
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chatbot/api/food', methods=['GET'])
def get_food():
    global available_food_cache
    locality = request.args.get('locality', '')
    filtered_food, error = fetch_food_listings(locality)
    if error:
        return jsonify({"error": error}), 500
    available_food_cache = filtered_food
    return jsonify(filtered_food)

@app.route('/chatbot/schedule-pickup', methods=['POST'])
def schedule_pickup():
    data = request.json
    main_response = requests.post(f"{MAIN_SERVER_URL}/pickup-request", json=data)
    if main_response.status_code not in [200, 201]:
        return jsonify({"error": "Failed to send pickup request to main server"}), 500
    event = {
        'summary': f"Food Pickup for food id {data.get('food_id')}",
        'location': data.get('address'),
        'description': f"Pickup scheduled for food listing id: {data.get('food_id')}",
        'start': {
            'dateTime': data.get('pickup_time_start'),
            'timeZone': 'America/Los_Angeles',
        },
        'end': {
            'dateTime': data.get('pickup_time_end'),
            'timeZone': 'America/Los_Angeles',
        },
    }
    try:
        created_event = calendar_service.events().insert(
            calendarId='3f6fe2962bedfdc87176d43218f12c9cc421266c891cb2cd7f14d79a5b3edac7@group.calendar.google.com',
            body=event
        ).execute()
    except Exception as e:
        return jsonify({"error": f"Google Calendar error: {e}"}), 500
    event_link = created_event.get('htmlLink')
    return jsonify({
        "message": f"Pickup scheduled successfully for food id {data.get('food_id')}. Event created: <a href='{event_link}' target='_blank'>View Event</a>"
    })

@app.route('/chatbot/query', methods=['POST'])
def query_handler():
    global available_food_cache, qa_cache
    data = request.json
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "No query provided"}), 400
    lower_query = query.lower()
    
    # 1. Scheduling branch: if the query contains scheduling keywords
    if any(keyword in lower_query for keyword in ["pickup", "schedule", "book", "collect"]):
        match = re.search(r'\b(\d+)\b', query)
        if match:
            food_id = int(match.group(1))
            selected_food = next((food for food in available_food_cache if int(food.get('id')) == food_id), None)
            if not selected_food:
                return jsonify({
                    "flow": "pickup_scheduling",
                    "message": f"Food with id {food_id} not found in available listings."
                })
            expiry_str = selected_food.get('expiry_date')
            try:
                if " " in expiry_str:
                    expiry_dt = datetime.strptime(expiry_str, '%Y-%m-%d %H:%M:%S')
                else:
                    expiry_dt = datetime.strptime(expiry_str, '%Y-%m-%d') + timedelta(hours=23, minutes=59, seconds=59)
            except Exception as e:
                expiry_dt = datetime.now() + timedelta(hours=1)
            now = datetime.now()
            if now >= expiry_dt:
                pickup_start = now
            else:
                delta = expiry_dt - now
                random_seconds = random.randrange(int(delta.total_seconds()))
                pickup_start = now + timedelta(seconds=random_seconds)
            pickup_end = pickup_start + timedelta(minutes=30)
            pickup_address = selected_food.get('address')
            user_email = "user@example.com"
            event = {
                'summary': f"Food Pickup for food id {food_id}",
                'location': pickup_address,
                'description': f"Pickup scheduled for food listing id: {food_id} from provider {selected_food.get('provider_id')}",
                'start': {
                    'dateTime': pickup_start.isoformat(),
                    'timeZone': 'America/Los_Angeles',
                },
                'end': {
                    'dateTime': pickup_end.isoformat(),
                    'timeZone': 'America/Los_Angeles',
                },
            }
            try:
                created_event = calendar_service.events().insert(
                    calendarId='3f6fe2962bedfdc87176d43218f12c9cc421266c891cb2cd7f14d79a5b3edac7@group.calendar.google.com',
                    body=event
                ).execute()
            except Exception as e:
                return jsonify({
                    "flow": "pickup_scheduling",
                    "message": f"Google Calendar error: {e}"
                })
            event_link = created_event.get('htmlLink')
            return jsonify({
                "flow": "pickup_scheduling",
                "message": f"Pickup scheduled successfully for food id {food_id}. Event created: <a href='{event_link}' target='_blank'>View Event</a>"
            })
        else:
            return jsonify({
                "flow": "pickup_scheduling",
                "message": "Could not extract food id from your query. Please include a number (the food id)."
            })
    
    # 2. Food listings branch: if the query is asking for available food listings
    elif any(keyword in lower_query for keyword in ["available food", "listing", "donation", "provider"]):
        default_locality = "Raleigh"
        filtered_food, error = fetch_food_listings(default_locality)
        if error:
            return jsonify({"error": error}), 500
        # Limit the results to only 3 food pickups
        limited_food = filtered_food[:3]
        available_food_cache = filtered_food
        return jsonify({
            "flow": "food_listings",
            "message": f"Here are the available food listings in {default_locality.title()} (showing 3 results):",
            "data": limited_food
        })
    
    # 3. RAG Q&A branch: if the query is for help/information about local resources
    else:
        # Use caching for repeated queries to speed up response times
        if query in qa_cache:
            answer = qa_cache[query]
        else:
            answer = qa_chain.run(query)
            qa_cache[query] = answer
        return jsonify({
            "flow": "rag_qa",
            "message": "Here is the information you requested:",
            "answer": answer
        })

if __name__ == '__main__':
    app.run(debug=True, port=5001)
