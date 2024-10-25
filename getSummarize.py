from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
# import google.generativeai as genai
from langchain.prompts import PromptTemplate
import re
from dotenv import load_dotenv
import os

load_dotenv()

caseCategory = {
  "Civil Law": [
    "Contract Disputes",
    "Property Disputes",
    "Family Law (Divorce, Child Custody, etc.)",
    "Tort Law (Personal Injury, Negligence, Defamation, etc.)",
    "Consumer Rights",
    "Employment Law",
    "Landlord-Tenant Disputes",
    "Intellectual Property (Patents, Trademarks, Copyright)",
    "Wills and Probate"
  ],
  "Criminal Law": [
    "Theft (Burglary, Robbery, etc.)",
    "Assault and Battery",
    "Homicide (Murder, Manslaughter)",
    "Drug-Related Offenses",
    "Cybercrime (Hacking, Fraud)",
    "Sexual Offenses (Rape, Child Abuse)",
    "White-Collar Crimes (Fraud, Embezzlement, Money Laundering)",
    "Terrorism",
    "Environmental Crimes"
  ],
  "Constitutional Law": [
    "Fundamental Rights Violation",
    "Judicial Review",
    "Election Law",
    "Separation of Powers",
    "Freedom of Speech and Expression",
    "Religious Rights",
    "Equality before Law"
  ],
  "Administrative Law": [
    "Government Contracts",
    "Zoning Regulations",
    "Licensing and Permits",
    "Public Health and Safety Regulations",
    "Education Law",
    "Environmental Regulations"
  ],
  "Corporate/Commercial Law": [
    "Corporate Governance",
    "Mergers and Acquisitions",
    "Securities and Investments",
    "Bankruptcy and Insolvency",
    "Commercial Contracts",
    "Antitrust and Competition Law"
  ],
  "Family Law": [
    "Divorce and Separation",
    "Child Custody and Visitation",
    "Adoption",
    "Domestic Violence",
    "Alimony and Child Support"
  ],
  "Environmental Law": [
    "Pollution Control",
    "Wildlife Protection",
    "Land Use and Zoning",
    "Natural Resource Management",
    "Climate Change Litigation",
    "Hazardous Waste Disposal"
  ],
  "Labor and Employment Law": [
    "Worker's Compensation",
    "Wrongful Termination",
    "Discrimination (Age, Gender, Disability, etc.)",
    "Workplace Safety",
    "Wage and Hour Disputes",
    "Employee Benefits and Pensions"
  ],
  "Tax Law": [
    "Income Tax Disputes",
    "Corporate Tax Disputes",
    "Tax Fraud",
    "Property Tax Issues",
    "GST/VAT"
  ],
  "Intellectual Property Law": [
    "Patent Infringement",
    "Trademark Disputes",
    "Copyright Violation",
    "Trade Secrets",
    "Licensing Agreements"
  ],
  "Real Estate Law": [
    "Land Disputes",
    "Zoning and Land Use",
    "Property Ownership and Transfer",
    "Landlord-Tenant Issues",
    "Construction Disputes"
  ],
  "International Law": [
    "Treaties and Agreements",
    "Extradition Cases",
    "International Trade Disputes",
    "Human Rights Law",
    "Diplomatic Immunity"
  ],
  "Human Rights Law": [
    "Discrimination",
    "Refugee and Asylum Rights",
    "Prisoner's Rights",
    "Gender Equality",
    "Freedom of Speech and Association"
  ],
  "Cyber Law": [
    "Data Privacy",
    "Cybersecurity Breaches",
    "Online Fraud and Identity Theft",
    "Intellectual Property in Digital Space",
    "Cyberbullying and Harassment"
  ],
  "Health Law": [
    "Medical Malpractice",
    "Public Health Regulations",
    "Medical Licensing",
    "Health Insurance Disputes",
    "Pharmaceutical Regulations"
  ],
  "Education Law": [
    "School Discipline",
    "Special Education Law",
    "Student Rights",
    "Teacher Employment Disputes"
  ],
  "Immigration Law": [
    "Visa and Residency Issues",
    "Deportation Cases",
    "Citizenship and Naturalization",
    "Asylum Cases",
    "Refugee Law"
  ],
  "Insurance Law": [
    "Life Insurance Claims",
    "Health Insurance Disputes",
    "Property and Casualty Insurance",
    "Motor Vehicle Insurance",
    "Insurance Fraud"
  ],
  "Maritime Law": [
    "Shipping and Trade Disputes",
    "Vessel Collisions",
    "Marine Insurance",
    "Salvage Rights",
    "Environmental Damage at Sea"
  ],
  "Entertainment and Media Law": [
    "Defamation and Libel",
    "Copyright Issues",
    "Artist Contracts",
    "Broadcasting Rights",
    "Film and Television Production Disputes"
  ]
}

def removeBold(string):
  return string.replace("**", "").replace("###", "").replace("##", "")


async def get_conversational_chain():
    # Define a prompt template for asking questions based on a given context
    prompt_template = """
    You are an expert in law and legal case classification. Given the following court case document text, please:
    Answer the question as detailed as possible from the provided context, make sure to provide all the details in a proper format,
    if the answer is not in the provided context just say dont answer that part, \n\n
    Context:\n {context}?\n
    caseCategories: \n{caseCategory}\n,
    Format:     1. Identify the main category from caseCategories for this case (e.g., Criminal, Civil, Constitutional).
                2. Identify the subcategory from caseCategories for this case(e.g., Homicide, Theft, Contract Dispute).
                3. Extract the year the judgment was passed.
                4. Provide a brief title for the case.
                5. Summarize the document in 1500-2000 words.
    Answer:
    """

    # Initialize a ChatGoogleGenerativeAI model for conversational AI
    # model = ChatVertexAI(model="gemini-pro", temperature=0.3)
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

    # Create a prompt template with input variables "context" and "question"
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context","caseCategory"]
    )

    # Load a question-answering chain with the specified model and prompt
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

async def summarize():
    # Create embeddings for the user question using a Google Generative AI model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # print(1)
    # Load a FAISS vector database from a local file
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    # print(2)
    # Perform similarity search in the vector database based on the user question
    docs = new_db.similarity_search("Summarize this document", k=3)
    # print(3)
    # Obtain a conversational question-answering chain
    chain = await get_conversational_chain()
    # print(4)
    # Use the conversational chain to get a response based on the user question and retrieved documents
    response = chain(
        {"input_documents": docs, "caseCategory":caseCategory}, return_only_outputs=True
    )
    # print(5)
    # Print the response to the console
    print(response["output_text"])

    return removeBold(response["output_text"])

    # Display the response in a Streamlit app (assuming 'st' is a Streamlit module)
    # st.write("Reply: ", response["output_text"])