from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings, load_doc
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

embeddings = download_hugging_face_embeddings()


splits = load_doc(list_file_path=['data\Medical_book.pdf'], chunk_size=300, chunk_overlap=50)
     

# Index

vectorstore = Chroma.from_documents(documents=splits,
                                    embedding=HuggingFaceEmbeddings())

