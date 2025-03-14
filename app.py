# import streamlit as st
# import os
# from langchain_groq import ChatGroq
# from langchain_community.document_loaders import WebBaseLoader
# from langchain.embeddings import OllamaEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# import time

# from dotenv import load_dotenv
# load_dotenv()

# ## load the Groq API key
# groq_api_key = os.environ['GROQ_API_KEY']

# # Initialize session state for embeddings and vectors
# if "vectors" not in st.session_state:
#     st.session_state.embeddings = OllamaEmbeddings()
#     st.session_state.vectors = None

# st.title("ChatGroq Demo")

# # User input for document URL
# document_url = st.text_input("Enter the URL of the document you want to load:")

# if document_url:
#     if "loaded_url" not in st.session_state or st.session_state.loaded_url != document_url:
#         st.session_state.loaded_url = document_url
#         st.session_state.loader = WebBaseLoader(document_url)
#         st.session_state.docs = st.session_state.loader.load()

#         st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
#         st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

# prompt_template = ChatPromptTemplate.from_template(
#     """
#     Answer the questions based on the provided context only.
#     Please provide the most accurate response based on the question
#     <context>
#     {context}
#     <context>
#     Questions:{input}
#     """
# )

# document_chain = create_stuff_documents_chain(llm, prompt_template)

# if st.session_state.vectors:
#     retriever = st.session_state.vectors.as_retriever()
#     retrieval_chain = create_retrieval_chain(retriever, document_chain)

#     prompt = st.text_input("Input your prompt here")

#     if prompt:
#         start = time.process_time()
#         response = retrieval_chain.invoke({"input": prompt})
#         print("Response time :", time.process_time() - start)
#         st.write(response['answer'])

#         # With a streamlit expander
#         with st.expander("Document Similarity Search"):
#             # Find the relevant chunks
#             for i, doc in enumerate(response["context"]):
#                 st.write(doc.page_content)
#                 st.write("--------------------------------")
# else:
#     st.write("Please enter a valid document URL to load the document.")



# import streamlit as st
# import os
# from langchain_groq import ChatGroq
# from langchain_community.document_loaders import WebBaseLoader
# from langchain.embeddings import OllamaEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# import time

# from dotenv import load_dotenv
# load_dotenv()

# ## load the Groq API key
# groq_api_key = os.environ['GROQ_API_KEY']

# # Initialize session state for embeddings and vectors
# if "vectors" not in st.session_state:
#     st.session_state.embeddings = OllamaEmbeddings()
#     st.session_state.vectors = None

# st.title("ChatGroq Demo")

# # User input for document URL
# document_url = st.text_input("Enter the URL of the document you want to load:")

# if document_url:
#     if "loaded_url" not in st.session_state or st.session_state.loaded_url != document_url:
#         st.session_state.loaded_url = document_url
#         st.session_state.loader = WebBaseLoader(document_url)
#         st.session_state.docs = st.session_state.loader.load()

#         st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
#         st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

# prompt_template = ChatPromptTemplate.from_template(
#     """
#     Answer the questions based on the provided context only.
#     Please provide the most accurate response based on the question
#     <context>
#     {context}
#     <context>
#     Questions:{input}
#     """
# )

# document_chain = create_stuff_documents_chain(llm, prompt_template)

# if st.session_state.vectors:
#     retriever = st.session_state.vectors.as_retriever()
#     retrieval_chain = create_retrieval_chain(retriever, document_chain)

#     prompt = st.text_input("Input your prompt here")

#     if prompt:
#         start = time.process_time()
#         response = retrieval_chain.invoke({"input": prompt})
#         print("Response time :", time.process_time() - start)
#         st.write(response['answer'])

#         # With a streamlit expander
#         with st.expander("Document Similarity Search"):
#             # Find the relevant chunks
#             for i, doc in enumerate(response["context"]):
#                 st.write(doc.page_content)
#                 st.write("--------------------------------")
# else:
#     st.write("Please enter a valid document URL to load the document.")

import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from deep_translator import GoogleTranslator  # Replace googletrans with deep-translator

from dotenv import load_dotenv
load_dotenv()

## Load the Groq API key
groq_api_key = os.environ['GROQ_API_KEY']

# Initialize session state for embeddings, vectors, and conversation history
if "vectors" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings()
    st.session_state.vectors = None

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

st.title("ChatGroq Demo")

# User input for document URL
document_url = st.text_input("Enter the URL of the document you want to load:")

if document_url:
    if "loaded_url" not in st.session_state or st.session_state.loaded_url != document_url:
        try:
            st.session_state.loaded_url = document_url
            st.session_state.loader = WebBaseLoader(document_url)
            st.session_state.docs = st.session_state.loader.load()

            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
            st.success("Document loaded successfully!")
        except Exception as e:
            st.error(f"Error loading document: {e}")

# Multi-language support
st.sidebar.header("Multi-language Support")
target_language = st.sidebar.selectbox("Select target language for responses:", ["en", "es", "fr", "de", "zh-cn"])

def translate_text(text, dest_language):
    """Translate text to the target language using Google Translate."""
    try:
        translated = GoogleTranslator(source='auto', target=dest_language).translate(text)
        return translated
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text  # Return original text if translation fails

# Initialize ChatGroq
llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

# Define the prompt template
prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

document_chain = create_stuff_documents_chain(llm, prompt_template)

if st.session_state.vectors:
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    prompt = st.text_input("Input your prompt here")

    if prompt:
        try:
            start = time.process_time()
            response = retrieval_chain.invoke({"input": prompt})
            print("Response time :", time.process_time() - start)

            # Translate the response if a target language is selected
            if target_language != "en":
                response['answer'] = translate_text(response['answer'], target_language)

            st.write(response['answer'])

            # Add the conversation to the history
            st.session_state.conversation_history.append((prompt, response['answer']))

            # Display document similarity search
            with st.expander("Document Similarity Search"):
                for i, doc in enumerate(response["context"]):
                    st.write(doc.page_content)
                    st.write("--------------------------------")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Display conversation history
st.sidebar.header("Conversation History")
if st.session_state.conversation_history:
    for i, (user_prompt, bot_response) in enumerate(st.session_state.conversation_history):
        with st.sidebar.expander(f"Conversation {i + 1}"):
            st.write(f"**You:** {user_prompt}")
            st.write(f"**Bot:** {bot_response}")
else:
    st.sidebar.write("No conversation history yet.")

# Error handling for missing document URL
if not document_url:
    st.warning("Please enter a valid document URL to load the document.")
    
