# MrBZU Langchain API
This project provides a Flask-based API that integrates Langchain with Firebase Storage and Google Cloud Storage to build a chatbot knowledge base from various document formats. It uses OpenAI’s GPT models for question-answering over documents uploaded to Firebase.

🛠️ Features
Knowledge Base Construction
Loads PDFs, CSVs, and TXT files from local directories or Firebase Storage, splits text into chunks, creates vector embeddings with OpenAI, and builds a retriever-based QA model.

Firebase & Google Cloud Storage Integration
Connects to Firebase Storage to upload, download, list, and delete files programmatically.

REST API Endpoints

Query the chatbot with user questions

List files stored in Firebase buckets

Download file content dynamically from Firebase (supporting PDF, CSV, TXT)

Delete files both locally and from Firebase

Update/download files from Firebase storage to local directories

Rebuild the chatbot knowledge base on demand

Security Best Practices
Uses environment variables to store sensitive credentials like OpenAI API key and Firebase credentials, avoiding hardcoding secrets in the source code.

🔧 Technologies Used
Python 3

Flask (REST API)

Firebase Admin SDK

Google Cloud Storage Client

Langchain (Text splitting, embeddings, vector stores)

OpenAI GPT-3.5-turbo

Pandas, PyPDF2 for file processing

🚀 Getting Started
Set up environment variables (e.g., in .env):

ini
نسخ
تحرير
OPENAI_API_KEY=your_openai_api_key
GOOGLE_APPLICATION_CREDENTIALS=/path/to/firebase-adminsdk.json
Install required packages:

bash
نسخ
تحرير
pip install -r requirements.txt
Run the Flask app:

bash
نسخ
تحرير
python main.py
Use the API endpoints to interact with the knowledge base and files.

📝 API Endpoints
GET /api/query/<query> — Query the chatbot with a user question.

GET /get-files-name-with-id/<folder_id> — List files in Firebase storage under specific folders.

GET /file/<filename> — Get content of a file (PDF, CSV, TXT) from Firebase storage.

DELETE /file/<filename> — Delete a file locally and from Firebase storage.

GET /updateData/<folder_id> — Download files from Firebase to local folders.

POST /rebuild-knowladgeBase/<data_type> — Rebuild the chatbot knowledge base.
