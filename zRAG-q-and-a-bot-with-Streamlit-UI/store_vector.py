from langchain.embeddings.openai import OpenAIEmbeddings
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.vectorstores import Chroma

os.environ["OPENAI_API_KEY"]="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"

embeddings = OpenAIEmbeddings(disallowed_special=())

def get_pdf_text():
    document=[]
    for file in os.listdir("docs"):
        if file.endswith(".pdf"):
            pdf_path="./docs/"+file
            loader=PyPDFLoader(pdf_path)
            document.extend(loader.load())
        elif file.endswith('.docx') or file.endswith('.doc'):
            doc_path="./docs/"+file
            loader=Docx2txtLoader(doc_path)
            document.extend(loader.load())
        elif file.endswith('.txt'):
            text_path="./docs/"+file
            loader=TextLoader(text_path)
            document.extend(loader.load())
    return document
def get_text_chunks(document):
    document_splitter=CharacterTextSplitter(separator='\n', chunk_size=500, chunk_overlap=50)
    document_chunks=document_splitter.split_documents(document)
    return document_chunks

def get_vector_store(document_chunks):
    global embeddings
    vectordb=Chroma.from_documents(document_chunks,embedding=embeddings, persist_directory='./openai1')
    vectordb.persist()
    return vectordb

raw_text = get_pdf_text()
text_chunks = get_text_chunks(raw_text)
vector_store = get_vector_store(text_chunks)
