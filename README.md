💬 MrBZU Langchain API
This project provides a Flask-based REST API that integrates Langchain, OpenAI, and Firebase Cloud Storage to build a smart, document-aware chatbot. It processes local or cloud-based PDF, TXT, and CSV files to create a searchable knowledge base using vector embeddings.

🛠️ Features
📚 Knowledge Base Construction
Loads PDF, CSV, and TXT files from local directories or Firebase Storage.

Splits text into chunks using RecursiveCharacterTextSplitter.

Generates embeddings with OpenAI.

Creates a retriever-powered QA chain using Langchain.

☁️ Firebase & Google Cloud Integration
Upload, download, list, and delete files from Firebase storage buckets.

🌐 REST API Capabilities
Query the chatbot with questions based on your documents.

Dynamically access or delete files from cloud or local storage.

Rebuild and update the knowledge base on demand.


Supports file format detection and content extraction (PDF, TXT, CSV).

🔐 Security Best Practices
Sensitive credentials (OpenAI API keys, Firebase service account JSON) are stored using environment variables.

Secrets are not hardcoded or committed to the repository.

🔧 Technologies Used
Python 3

Flask (REST API)

Firebase Admin SDK

Google Cloud Storage Python Client

Langchain (embeddings, text splitting, retriever chains)

OpenAI GPT-3.5-turbo

PyPDF2, Pandas for file parsing

🚀 Getting Started
1. Set Environment Variables
Create a .env file or export environment variables directly:
OPENAI_API_KEY=your_openai_api_key
GOOGLE_APPLICATION_CREDENTIALS=/path/to/firebase-adminsdk.json
Ensure your Firebase service account JSON file is not committed to the repository.

2. Install Required Packages
pip install -r requirements.txt
3. Run the Flask App
python main.py
