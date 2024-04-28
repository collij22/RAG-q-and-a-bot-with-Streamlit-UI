import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
#from langchain_google_genai import ChatGoogleGenerativeAI

import os

os.environ["OPENAI_API_KEY"]="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"
#embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
#instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",model_kwargs={"device": "cpu"})
embeddings = OpenAIEmbeddings()
vector_store = Chroma(persist_directory="openai1\\",embedding_function=embeddings)
GOOGLE_API_KEY="api key"

def get_conversational_chain(vector_store):
    llm=ChatOpenAI(temperature=0.7, model_name='gpt-4')
    #llm = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True, google_api_key=GOOGLE_API_KEY)
    memory = ConversationBufferMemory(memory_key = "chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                             retriever=vector_store.as_retriever(search_kwargs={'k':3}),
                                             verbose=False, memory=memory)
    return conversation_chain

def user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chatHistory = response['chat_history']
    for i, message in enumerate(st.session_state.chatHistory):
        if i%2 == 0:
            st.write("Human: ", message.content)
        else:
            st.write("Bot: ", message.content)
def main():
    st.set_page_config("Chat with Multiple PDFs")
    st.header("Chat with Multiple PDF 💬")
    user_question = st.text_input("Ask a Question from the PDF Files")
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = None
    if user_question:
        user_input(user_question)
    with st.sidebar:
        st.title("Settings")
        st.subheader("Upload your Documents")
        st.session_state.conversation = get_conversational_chain(vector_store)
        st.success("Done")



if __name__ == "__main__":
    main()
