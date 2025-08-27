import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_experimental.agents.agent_toolkits import create_csv_agent
import streamlit as st
import glob
import os
import csv

# Constant
INDEX_DIR = "faiss"
CSV_INDEX_DIR = "datasets/*.csv"
TXT_INDEX_DIR = "datasets/*.txt"
PDF_INDEX_DIR = "datasets/*.pdf"
llm = OllamaLLM(model="gemma3:1b", temperature=0.2)  # Load LLM Model
combined_data = []
template = """
You are a helpful assistant that answers questions based on the provided text, PDF document and/or CSV/EXCEL files.
Use only the context provided to answer the question. If you don't know the answer or can't find it in the context, say so. Please provide longer but pleasant to read answer, user likely doesn't like really short answer.
Context: {context}

Question: {question}

Detailed Answer:
"""

CUSTOM_PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

@st.cache_resource # to make sure this only run once
def rag_chain_func():
    # Preparing vector database
    # Convert the chunks into embeddings and store in FAISS or load existing FAISS index
    # Alternatively you could use: https://huggingface.co/CompendiumLabs/bge-base-en-v1.5-gguf
    # embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    if os.path.exists('/datasetan'):
        print("Loading existing FAISS index...")
        vectorDB = FAISS.load_local(
            INDEX_DIR, embedding_model, allow_dangerous_deserialization=True
        )
    else:
        print("Creating new FAISS index from file(s)...")

        # Load textfile for finetuning the LLM using this tutorial:
        # https://medium.com/@anil.goyal0057/building-a-simple-rag-system-with-langchain-faiss-and-ollama-mistral-model-822b3245e576
        # With some reference from:
        # https://huggingface.co/blog/ngxson/make-your-own-rag
        # https://blog.dailydoseofds.com/p/hands-on-rag-over-excel-sheets
        

        # Loading CSV File
        # https://dev.to/jhparmar/rag-with-chromadb-llama-index-ollama-csv-23f7 [not used, reserved for later]
        # https://medium.com/@satyadeepbehera/how-to-query-csv-and-excel-files-using-langchain-9d59dde42c5f
        # Loads tons of CSVs
        # https://towardsdev.com/efficiently-reading-multiple-csv-files-without-pandas-03373b52166e
        # https://www.machinelearningplus.com/gen-ai/build-a-simple-rag-system-with-csv-files-step-by-step-guide-for-beginners/
        csv_file_paths = glob.glob(CSV_INDEX_DIR)
        pdf_file_paths = glob.glob(PDF_INDEX_DIR)
        txt_file_paths = glob.glob(TXT_INDEX_DIR)

        for filename in csv_file_paths:
            loader = CSVLoader(file_path=filename, encoding="utf-8-sig", csv_args={'delimiter': ';'})
            combined_data.extend(loader.load())
        
        for filename in txt_file_paths:
            loader = TextLoader(file_path=filename, encoding='utf-8')
            combined_data.extend(loader.load())

        for filename in pdf_file_paths:
            loader = PyPDFLoader(file_path=filename)
            combined_data.extend(loader.load())
            
        if not combined_data:
            raise FileNotFoundError("No CSV, TXT, or PDF files found in the 'data' directory.")


        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        docs = text_splitter.split_documents(combined_data)
        # docs = text_splitter.create_documents(combined_data)

        vectorDB = FAISS.from_documents(docs, embedding_model)
        vectorDB.save_local(INDEX_DIR)

    # Convert vector DB into retriever object
    # so it can return the top 3 of most similiar documents or chunk based on user query
    retriever = vectorDB.as_retriever(search_kwargs={"k": 3})  # hence the 3 in k
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, chain_type="stuff", return_source_documents=True, chain_type_kwargs={'prompt': CUSTOM_PROMPT}
    )

    return rag_chain


# Streamlit interface
st.title("Prooza - AI Assisstant for your pricing needs")
rag_chain = rag_chain_func()

# Calling session
if 'messages' not in st.session_state:
    st.session_state.messages = []

if rag_chain is not None:
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    user_query = st.chat_input("Ask me anything")

    if user_query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate and display bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = rag_chain.invoke({"query": user_query})
                bot_response = response['result']
                st.markdown(bot_response)
        
        # Add bot response to chat history
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
