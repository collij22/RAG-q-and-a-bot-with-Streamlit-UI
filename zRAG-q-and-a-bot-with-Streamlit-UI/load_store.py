from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
import os
os.environ["OPENAI_API_KEY"]="api_key here"

embeddings = OpenAIEmbeddings()

#embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
db = Chroma(persist_directory="data\\",embedding_function=embeddings)
results = db.similarity_search_with_score("what is your name", k=3)
print(results)
