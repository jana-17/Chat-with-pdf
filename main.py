import os
import pickle
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from utils import calculate_file_hash
from config import QA_PROMPT

with st.sidebar:
    st.title('Chat With PDF 📄')
    st.markdown('''
    ## About
    This is a PDF chat application     
 
    ''')


pdf_save_path = "doc_files"
vec_store_path = "vector_stores"

if not os.path.exists(pdf_save_path):
    os.makedirs(pdf_save_path)

if not os.path.exists(vec_store_path):
    os.makedirs(vec_store_path)


st.header('Chat with PDF')
pdf = st.file_uploader("Upload your PDF here",type='pdf')

if pdf is not None:
    # Save the uploaded file to the specified path
    file_path = os.path.join(pdf_save_path, pdf.name)
    
    with open(file_path, "wb") as f:
        f.write(pdf.getbuffer())
    
    st.success(f"PDF uploaded successfully!! \nYou can start chating." )

st.sidebar.header("Available PDFs")
pdfs = [f for f in os.listdir(pdf_save_path) if f.endswith('.pdf')]

# Select a PDF from the sidebar
selected_pdf = st.sidebar.selectbox("Choose a PDF", pdfs)

# Display selected PDF content (if needed)
if selected_pdf:
    st.write(f"You selected: {selected_pdf}")


llm = ChatGroq(model="llama3-8b-8192")


def documents_to_splits(file_path):
    # Load and chunk 
    loader = PyPDFLoader(file_path, extract_images=False)
    docs = loader.lazy_load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    return splits

def store_embeddings(chunks, embeddings, store_name):
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    with open(f"vector_stores/{store_name}.pkl", "wb") as f:
        pickle.dump(vectorstore, f)
    return vectorstore

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.getenv('HF_API_KEY'), model_name="sentence-transformers/all-MiniLM-l6-v2"
)

if selected_pdf:
    hashvalue = calculate_file_hash(os.path.join(pdf_save_path,selected_pdf))

    # creating vectorstores
    if hashvalue in [file[:file.rfind('.')]for file in os.listdir('vector_stores')]:
        with open(f"vector_stores/{hashvalue}.pkl", "rb") as f:
            vectorstore = pickle.load(f)
    else:
        chuncks = documents_to_splits(os.path.join(pdf_save_path,selected_pdf))
        vectorstore = store_embeddings(chuncks, embeddings, hashvalue)


    prompt = PromptTemplate.from_template(QA_PROMPT)

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # retrieved_docs = retriever.invoke("Why one should start a company?")

    def format_docs(docs: list) -> str:
        """
        Formats the output from the retriver. (list[str] -> str)
        """
        return "\n\n".join(doc.page_content for doc in docs)

    # our RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Chat interface
    st.subheader("Chat Interface")
    user_input = st.text_input("Type your message:")
    
    if user_input:
        llm_answer = rag_chain.stream(user_input)
        st.write_stream(llm_answer)

else:
    st.write("Upload or Select a PDF")

