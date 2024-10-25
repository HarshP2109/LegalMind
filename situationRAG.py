from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
# import google.generativeai as genai
from langchain.prompts import PromptTemplate
import re
from dotenv import load_dotenv
import os

load_dotenv()

def removeBold(string):
  return string.replace("**", "").replace("###", "").replace("##", "")

async def get_conversational_chain():
    # Define a prompt template for asking questions based on a given context
    prompt_template = """
    Answer the question as like you are a legal professional as detailed as possible with the help of lawbook provided, make sure to provide all the details,
    don't provide the wrong answer\n\n
    LawBook:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    # Initialize a ChatGoogleGenerativeAI model for conversational AI
    # model = ChatVertexAI(model="gemini-pro", temperature=0.3)
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

    # Create a prompt template with input variables "context" and "question"
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Load a question-answering chain with the specified model and prompt
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

async def situation_input(context, user_question):
    
    question = f"""
    Context:\n {context}?\n
    Question: \n{user_question}\n
    """
    print(question)
    # Create embeddings for the user question using a Google Generative AI model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # print(1)
    # Load a FAISS vector database from a local file
    new_db = FAISS.load_local("CrimeBook", embeddings, allow_dangerous_deserialization=True)
    # print(2)
    # Perform similarity search in the vector database based on the user question
    docs = new_db.similarity_search(question, k=3)
    print("embeddings search done")
    # print(3)
    # Obtain a conversational question-answering chain
    chain = await get_conversational_chain()
    print("Got the chain")
    # print(4)
    # Use the conversational chain to get a response based on the user question and retrieved documents
    response = chain(
        {"input_documents": docs, "question": question}, return_only_outputs=True
    )
    print("got response")
    # print(5)
    # Print the response to the console
    # print(response["output_text"])

    return removeBold(response["output_text"])

    # Display the response in a Streamlit app (assuming 'st' is a Streamlit module)
    # st.write("Reply: ", response["output_text"])