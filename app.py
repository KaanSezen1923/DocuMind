__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import tempfile
import streamlit as st
from dotenv import find_dotenv,load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


load_dotenv(find_dotenv())
gemini_api_key=os.getenv("GEMINI_API_KEY")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=gemini_api_key)
llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash",api_key=gemini_api_key)
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question.If answer doesn't exist context answer the own data."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

def read_split_pdf(uploaded_files):
    if uploaded_files is None:
        return None

    all_splits=[]

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_filepath = tmp_file.name
        loader=PyPDFLoader(temp_filepath)
        data=loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(data)
        all_splits.extend(splits)
    return all_splits
        
    
def create_retriever(documents,embedding,):
    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings,persist_directory="./chroma_db")
    retriever = vectorstore.as_retriever()
    return retriever

def create_chain(llm,prompt,retriever):
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain
    
    

st.set_page_config("DocuMind")
st.header("RAG based Chat with PDF")

if "retriever" not in st.session_state:
    st.session_state.retriever = None


if "messages" not in st.session_state:
    st.session_state["messages"]=[]
    st.session_state["messages"].append({"role":"ai","content":"Hi. I'm DocuMind.Upload PDF files and ask me questions!"})
    
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])



with st.sidebar:
    st.title("Menü")
    filename=st.sidebar.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",accept_multiple_files=True,type="PDF")
    if st.button("Submit and Process  button"):
        with st.spinner("Processing..."):
            data=read_split_pdf(filename)
            if data:
                st.session_state.retriever = create_retriever(data, embeddings)
                st.success("Processing completed!")
            else:
                st.error("Please upload PDF files first")
            
user_query=st.chat_input("Enter your query:")

    
            
if user_query and st.session_state.retriever :
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state["messages"].append({"role":"user","content":user_query})
    chain=create_chain(llm,prompt,st.session_state.retriever)
    results=chain.invoke({"input":user_query})
    response=results["answer"]
    with st.chat_message("ai"):
        st.write(response)
    st.session_state["messages"].append({"role":"ai","content":response})
