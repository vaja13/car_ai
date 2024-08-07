
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_vector_stores(docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_stores = FAISS.from_documents(docs, embedding=embeddings)
    vector_stores.save_local("vehicle_troubleshoot")

def get_conversational_chain():
    # template = '''You are an expert in answering question regarding car troubleshooting from the context you are provided. 
    # Make sure that if answer is relevant then give detailed explanation and if answer is not available
    # then check whether it is related to vehicle if it is not related to vehicle then give response
    # as i dont have this domain expertise.
    # Context : \n{context}\n
    # Question : \n{question}\n
    # Answer :
    # If question asked is not from vehicle troubleshooting and vehicle information related
    # and if it is regarding navigation or direction related then give response as "0"
    # '''
    template = '''You are an expert in answering question regarding vehicle parts and
    vehicle troubleshooting.
    If question asked is regarding navigation or direction related then give response as "0"
    if question is other than this above mention domain then return "1"
    Domain knowledge additional : \n{context}\n
    Question : \n{question}\n
    Answer :
    '''
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.4)
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    db = FAISS.load_local("/Users/akshatvaja/Documents/Tata_Project/vehicle_troubleshoot",
                           embeddings,
                            allow_dangerous_deserialization=True)
    retriever = db.as_retriever()
    retrievalQA = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return retrievalQA

def get_response(user_question):
    retrievalQA = get_conversational_chain()
    response = retrievalQA.invoke({"query": user_question})
    return response["result"]

# def main():
    # st.set_page_config("Chat PDF")
    # st.header("Chat with PDF using Gemini💁")

    # btn = st.button("Enter")
    # # user_question = st.text_input("Ask a Question from the PDF Files")

    # # Uncomment these lines if you need to load and index PDFs initially
    # if btn:
    #     loader = PyPDFDirectoryLoader("/Users/akshatvaja/Documents/Tata_Project/books")
    #     documents = loader.load()
    #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    #     final_documents = text_splitter.split_documents(documents)
    #     get_vector_stores(final_documents)
    #     st.success("PDFs indexed successfully")

    # if user_question:
    #     response = get_response(user_question)
    #     st.write("Reply:", response)

# if __name__ == "__main__":
#     main()












# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_community.document_loaders import PyPDFDirectoryLoader,PyPDFLoader
# from langchain.chains.question_answering import load_qa_chain
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# import os

# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# def get_vector_stores(docs):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_stores = FAISS.from_documents(docs,embedding=embeddings)
#     vector_stores.save_local("vehicle_index")

# def get_conversational_chain():
#     template = '''You are an expert in answering question from the context you are provided,Make Sure that if answer is relavent then give detailed explaination and if answer is not available then just answer it is not available in context.
#     context : \n{context}\n
#     Question : \n{question}\n

#     Answer :
#     '''
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.4)
#     prompt = PromptTemplate(template=template , input_variables=["context","question"])
#     # chain = load_qa_chain(model , chain_type="stuff",prompt = prompt)
#     db = FAISS.load_local("vehicle_index",embeddings,allow_dangerous_deserialization=True)
#     retriever = db.as_retriever()
#     retrievalQA = RetrievalQA.from_chain_type(
#         llm=model,
#         chain_type="stuff",
#         retriever=retriever,
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": prompt}
#     )
#     return retrievalQA


# def get_response(user_quetion):
#     # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#     # db = FAISS.load_local("vehicle_index",embeddings,allow_dangerous_deserialization=True)
#     # docs = db.similarity_search()
#     # retriver = db.as_retriever()

#     retrievalQA = get_conversational_chain()
   
#     response = retrievalQA.invoke(input)
#     # return response['result']
#     # response = chain(
#     #     {"input_documents":docs , "question":user_quetion},
#     #     return_only_outputs=True
#     # )

#     # print(response)
#     st.write("Reply",response['result'])

# def main():
#     st.set_page_config("Chat PDF")
#     st.header("Chat with PDF using Gemini💁")

#     user_question = st.text_input("Ask a Question from the PDF Files")

#     # loader=PyPDFDirectoryLoader("/Users/akshatvaja/Documents/Tata_Project/vehicle_info")

#     # documents=loader.load()

#     # text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

#     # final_documents=text_splitter.split_documents(documents)
#     # print(final_documents)

#     # get_vector_stores(final_documents)
#     # print("Done")
#     # # st.success("Done")

#     if user_question:
#         get_response(user_question)





# if __name__ == "__main__":
#     main()

