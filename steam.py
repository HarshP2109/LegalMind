from getRAG import user_input
from getSummarize import summarize
from situationRAG import situation_input
from getEmbedd import generateEmbedding
import streamlit as st
from PyPDF2 import PdfReader
import time
import asyncio
import shutil
import os

# To delete Faiss After done
def deleteFaiss():
    try:
        shutil.rmtree("faiss_index")
        print("Directory 'faiss' deleted successfully.")
    except FileNotFoundError:
        print("Directory 'faiss' not found.")


async def getRag(text):
    # summarizer = pipeline("summarization")
    # summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return await user_input(text)

async def createEmbedding(file_text):
    res = await generateEmbedding(file_text)
    if res is None:
        ValueError("Error in Creating Embedding")
    else:
        return res
    


# Below Frontend

# Initialize session state for API key and messages
# if 'file' not in st.session_state:
#     st.session_state.file = None
if 'situation' not in st.session_state:
    st.session_state.situation = ""
if 'SitRag' not in st.session_state:
    st.session_state.SitRag = []
if 'file_text' not in st.session_state:
    st.session_state.file_text = ""
if 'db_path' not in st.session_state:
    st.session_state.db_path = ""
if 'Rag' not in st.session_state:
    st.session_state.Rag = []
# if 'new' not in st.session_state:
#     st.session_state.new = False
if 'page' not in st.session_state:
    st.session_state.page = "Home"

# Function to display chat interface
def chat_interface():

        # Display chat history
    for chat in st.session_state.Rag:
        if chat["role"] == "user":
            with st.chat_message("user"):
                st.markdown(f"**You:** {chat['content']}")
        else:
            with st.chat_message("assistant"):
                st.markdown(f"**Assistant:** {chat['content']}")

    # Accept user input
    prompt = st.chat_input("Say something")

    if prompt:

        # Add user message to chat history
        st.session_state.Rag.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(f"**You:** {prompt}")

        # Display loading message
        loading_message_placeholder = st.empty()
        loading_message_placeholder.markdown("**Loading...**")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = loop.run_until_complete(getRag(prompt))
            st.markdown(f"**LegalAssist:** {response}")

        # Clear loading message and display response
        loading_message_placeholder.empty()
        st.session_state.Rag.append({"role": "assistant", "content": response})


# Function to display chat interface
def situation_interface():

    situat = st.text_area("Enter the situation of at least 50 words here:")
    st.session_state.situation = situat

    if st.button("Save") or st.session_state.situation is not "":
            # Display chat history
        for chat in st.session_state.SitRag:
            if chat["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(f"**You:** {chat['content']}")
            else:
                with st.chat_message("assistant"):
                    st.markdown(f"**Assistant:** {chat['content']}")

        # Accept user inputs
        prompt = st.chat_input("Say something")

        if prompt:

            # Add user message to chat history
            st.session_state.SitRag.append({"role": "user", "content": prompt})

            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(f"**You:** {prompt}")

            # Display loading message
            loading_message_placeholder = st.empty()
            loading_message_placeholder.markdown("**Loading...**")

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                response = loop.run_until_complete(situation_input(situat,prompt))
                st.markdown(f"**LegalAssist:** {response}")

            # Clear loading message and display response
            loading_message_placeholder.empty()
            st.session_state.SitRag.append({"role": "assistant", "content": response})

    if st.button("Reset"):   
        st.session_state.situation = ""
        st.success("Data reset! Please provide a new scenario on this page.")

    # st.write(messages_key)

# Summarisation Page
def summarisation():
    st.title("Case Summarization")
    # st.write("Please wait, summarisation in progress...")

    # Lazy loading simulation
    if st.button("Summarize"):
        # Display loading message
        loading_message_placeholder = st.empty()
        loading_message_placeholder.markdown("**Sumarizing...**")

        # Display assistant response in chat message container
        # with st.chat_message("assistant"):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(summarize())
        # st.write("Summarisation complete!")
        st.text_area("Summary", response, height=700)
        # Clear loading message and display response
        loading_message_placeholder.empty()

        # st.write("### Summary :")
        

# Main application
st.set_page_config(page_title="QA RAG App", layout="wide")

# Sidebar for navigation
if  st.session_state.file_text:
    
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", ["Home", "Summarisation", "QA Chat", "Situation Based Chat"])

else:
    # st.session_state.new = False
    page = "Home"
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", ["Home", "Situation Based Chat"])

if page == "Home":
    st.title("Welcome to LawGPT")
    st.write("LawGPT is an AI-driven solution designed to assist legal professionals with the following features:")
    st.write("""
        - **Case Summarisation:** Get concise summaries of long legal documents.
        - **RAG (Retrieval-Augmented Generation):** Perform question-answering on your uploaded case files.
        - **Case Suggestions:** Receive suggestions based on past similar cases.
    """)

        # File upload or text input (min 50 words)
    option = st.selectbox("Choose an input method:", ["Upload a PDF", "Enter text (at least 50 words)"])

    if option == "Upload a PDF":
        uploaded_file = st.file_uploader("Upload a legal case PDF", type=["pdf"])
    else:
        case_text = st.text_area("Enter at least 50 words of legal text here:")

    # Input to save API key
    # api_key_input = st.text_input("Enter Gemini API key:", type="password")

    if st.button("Save"):
        deleteFaiss()
        if option == "Upload a PDF":
            reader = PdfReader(uploaded_file)
            case_text = ""
            for page in reader.pages:
                case_text += page.extract_text()
            st.session_state.file_text = case_text
        st.session_state.page = "QA Chat"

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        st.session_state.db_path = loop.run_until_complete(createEmbedding(case_text))

        # os.environ["GOOGLE_API_KEY"]=api_key_input
        # st.session_state.new = True
        st.success("File saved! You can now access the chat pages from the sidebar.")
        st.rerun() # Rerun to update the page state

    if st.session_state.file_text:
        if st.button("Reset"):
            # st.session_state.file = None
            deleteFaiss()
            st.session_state.file_text = ""
            st.session_state.db_path = ""
            st.session_state.page = "Home"
            st.success("File Data reset! Please upload a new file on the Home page.")
            st.rerun()  # Rerun to update the page state

elif page == "Summarisation":
    summarisation()

elif page == "QA Chat":
    st.title("Case Chat")
    st.session_state.page = "Rag"
    # st.session_state.new = False
    chat_interface()

elif page == "Situation Based Chat":
    st.title("Situation Based Chat")
    st.session_state.page = "Situation"
    # st.session_state.new = False
    situation_interface()


    
