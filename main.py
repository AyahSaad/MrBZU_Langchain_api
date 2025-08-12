import os
import io
import firebase_admin
import pandas as pd
import PyPDF2
from flask import Flask, jsonify
from flask_cors import CORS
from pathlib import Path
from firebase_admin import credentials, storage
from google.cloud import storage as gcs
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
import json
import threading
import signal
import sys

app = Flask(__name__)
CORS(app)

# Set the API key as an environment variable
api_key = os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "GOOGLE_APPLICATION_CREDENTIALS"

# Define pqa globally
qa = None

history = []


# Function to initialize and connect firebase storage
def initialize_firebase():
    cred = credentials.Certificate("")
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'mr-bzu-v1.appspot.com'
    })
    print("------------FIREBASE INITIALIZE DONE -----------")


# function to read the pdf files from local
def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text


# function to read the csv files from local
def read_csv(file_path):
    df = pd.read_csv(file_path)
    text = df.to_string(index=False)
    return text


# function to load data and preparing it to build chatbot knowledge base
class EnhancedDirectoryLoader:
    def __init__(self, directory_path, glob_pattern):
        self.directory = Path(directory_path)
        self.glob_pattern = glob_pattern

    def load(self):
        documents = []
        for file_path in self.directory.glob(self.glob_pattern):
            if file_path.suffix.lower() == '.pdf':
                text = read_pdf(file_path)
                documents.append(Document(page_content=text))
            elif file_path.suffix.lower() == '.txt':
                with open(file_path, encoding="utf-8") as file:
                    text = file.read()
                    documents.append(Document(page_content=text))
            elif file_path.suffix.lower() == '.csv':
                text = read_csv(file_path)
                documents.append(Document(page_content=text))
        return documents


def MrBZU_knowladgeBase():
    global qa
    try:
        # define model parameters
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=1900,
            n=1,
            openai_api_key=api_key)

        directory_path = "./studentData"
        p_loader = EnhancedDirectoryLoader(directory_path, '*.*')
        p_data = p_loader.load()

        p_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        p_split_data = p_text_splitter.split_documents(p_data)
        p_embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        p_vector_data = Chroma.from_documents(p_split_data, p_embeddings)

        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=p_vector_data.as_retriever())
        print("Person QA model initialized successfully.")
    except Exception as e:
        print(f"Error initializing Person QA model: {e}")


# get method to return a response for user query
@app.route('/api/query/<query>', methods=['GET'])
def handle_query(query):
    global qa

    print("------------------ QA ------------------")
    print(qa)

    print("------------------ QUERY ------------------")
    print(query)
    # Assuming history management can be improved; for now, it's always reset.
    history = []
    chat_list = list(sum(history, ()))
    chat_list.append(query)
    query_str = ' '.join(chat_list)

    response = qa({"query": query_str})

    # Remove newline characters from the response
    cleaned_response = response["result"].replace('\n', '')

    print("------------------ RESPONSE ------------------")
    print(cleaned_response)
    # Format the output strings

    return jsonify({"response": cleaned_response})


# function to get files name with ids from firebase
@app.route('/get-files-name-with-id/<folder_id>', methods=['GET'])
def get_files_names_with_id(folder_id):
    # Initialize a client for Google Cloud Storage
    client = gcs.Client()

    # Define the prefixes based on the folder_id
    prefixes = ['data/'] if folder_id == '1' else ['data/', 'studentsData/']

    # Get the bucket object
    bucket = client.bucket('mr-bzu-v1.appspot.com')

    # Create a list to store file data
    files_data = []

    # List and collect file data for each prefix
    for prefix in prefixes:
        blobs = bucket.list_blobs(prefix=prefix)
        for blob in blobs:
            file_name = blob.name.split('/')[-1]  # Get the file name
            file_id = blob.id.split('/')[-1]  # Extract the file ID from the blob.id string
            file_uri = f"gs://{bucket.name}/{blob.name}"  # Construct the GCS URI
            file_data = {
                'file_name': file_name,
                'file_id': file_id,
                'file_uri': file_uri
            }
            files_data.append(file_data)

    # Return JSON response
    return jsonify(files_data)


# function to rebuild the knowledge base
@app.route('/rebuild-knowladgeBase/<string:data_type>', methods=['POST'])
def rebuild_KnowladgeBase(data_type):
    if data_type == 'student':
        MrBZU_knowladgeBase()
        print("Students Knowledge base updated !")
    else:
        MrBZU_knowladgeBase()
        print("Person Knowledge base updated !")

    return "Update Mr. BZU Knowladge Base Done!"

# function to get file content from firebase according to its name
@app.route('/file/<filename>', methods=['GET'])
def get_file_content(filename):
    # Initialize a client for Google Cloud Storage
    client = gcs.Client()
    # Get the bucket object
    bucket = client.bucket('mr-bzu-v1.appspot.com')

    # Directories to check
    directories = ['data', 'studentsData']
    blob = None

    # Check each directory for the file
    for directory in directories:
        blob_path = f"{directory}/{filename}"
        temp_blob = bucket.blob(blob_path)
        if temp_blob.exists():
            blob = temp_blob
            break

    # Check if the file exists
    if not blob.exists():
        return f"File '{filename}' not found."

    # Determine the file extension
    file_extension = os.path.splitext(filename)[1].lower()

    text_content = ""

    if file_extension == '.pdf':
        # Download the file content as bytes
        content = blob.download_as_bytes()

        # Create a BytesIO object from the file content
        pdf_file = io.BytesIO(content)

        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        # Get the number of pages in the PDF
        num_pages = len(pdf_reader.pages)

        # Initialize an empty string to store the text content
        text_content = ""

        # Iterate over each page and extract the text
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text_content += page.extract_text()

    elif file_extension == '.txt':
        # Handle TXT file
        content = blob.download_as_string()
        text_content = content.decode("utf-8")
    elif file_extension == '.csv':
        # Handle CSV file
        content = blob.download_as_string()
        csv_file = io.StringIO(content.decode("utf-8"))
        df = pd.read_csv(csv_file)
        text_content = df.to_string(index=False)
    else:
        return f"Unsupported file extension '{file_extension}'"

    return text_content

# function to delete file  from firebase according to its name.
@app.route('/file/<filename>', methods=['DELETE'])
def delete_file(filename):
    # Initialize a client for Google Cloud Storage
    client = gcs.Client()
    # Get the bucket object
    bucket = client.bucket('mr-bzu-v1.appspot.com')

    # Directories to check
    cloud_directories = ['data', 'studentsData']
    local_directories = ['./studentData', './data']

    cloud_file_deleted = False
    local_file_deleted = False

    # Check each cloud directory for the file and attempt deletion
    for directory in cloud_directories:
        blob_path = f"{directory}/{filename}"
        blob = bucket.blob(blob_path)
        try:
            if blob.exists():
                blob.delete()
                cloud_file_deleted = True
                print(f"File '{filename}' deleted successfully from cloud storage '{directory}'.")
        except Exception as e:
            print(f"Failed to delete '{filename}' from cloud storage '{directory}': {e}")

    # Check each local directory for the file and attempt deletion
    for directory in local_directories:
        file_path = os.path.join(directory, filename)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                local_file_deleted = True
                print(f"File '{filename}' deleted successfully from local directory '{directory}'.")
        except Exception as e:
            print(f"Failed to delete '{filename}' from local directory '{directory}': {e}")

    # Prepare the response based on deletion status
    if cloud_file_deleted or local_file_deleted:
        return f"File '{filename}' deleted successfully from all relevant directories.", 200
    else:
        return f"File '{filename}' not found in any specified directories.", 404


# function to update chatbot data from firebase
@app.route('/updateData/<folder_id>', methods=['GET'])
def updateData(folder_id):
    directory = "data" if folder_id == '1' else "studentData"
    prefixes = ['data/'] if folder_id == '1' else ['data/', 'studentsData/']

    # Ensure the local directory exists
    local_data_dir = f'./{directory}'
    os.makedirs(local_data_dir, exist_ok=True)

    # Delete local files in the specified directory
    for filename in os.listdir(local_data_dir):
        file_path = os.path.join(local_data_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                print(f"Removed {file_path}")
        except Exception as e:
            return jsonify(message=f"Failed to delete {file_path}: {e}"), 500

    # Initialize a client for Google Cloud Storage
    client = gcs.Client()

    # Get the bucket object
    bucket = client.bucket('mr-bzu-v1.appspot.com')

    # Download files from the specified prefixes
    for prefix in prefixes:
        blobs = bucket.list_blobs(prefix=prefix)
        for blob in blobs:
            if not blob.name.endswith('/') and (folder_id != '1' or 'studentsData/' not in blob.name):
                file_name = blob.name.split('/')[-1]  # Get the file name without the prefix

                # Download the file content
                local_file_path = os.path.join(local_data_dir, file_name)
                blob.download_to_filename(local_file_path)
                print(f"Downloaded {file_name} to {local_file_path}")

    return 'Files have been downloaded successfully', 200

def signal_handler(sig, frame):
    print('Shutting down gracefully...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":

    # -------------- Build knowledge-base -------------- #
    MrBZU_knowladgeBase()

    # -------------- Initialize FireBase -------------- #
    initialize_firebase()

    app.run(host="0.0.0.0", port=80, debug=True, use_reloader=False)