from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings, load_doc
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.constants import llm_model , temperature, max_tokens, top_k
from src.prompt import *
import os

app = Flask(__name__)

embeddings = download_hugging_face_embeddings()

print('embeddings loaded')
splits = load_doc(list_file_path=['data\Medical_book.pdf'], chunk_size=300, chunk_overlap=50)
     

# Index

vectordb = Chroma.from_documents(documents=splits,embedding=HuggingFaceEmbeddings())
print('vector database made')
index_name="medical-bot"

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}

llm = HuggingFaceEndpoint(
            repo_id=llm_model,
            # model_kwargs={"temperature": temperature, "max_new_tokens": max_tokens, "top_k": top_k, "load_in_8bit": True}
            temperature = temperature,
            max_tokens = max_tokens,
            top_k = top_k,
            load_in_8bit = True,)

qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=vectordb.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)