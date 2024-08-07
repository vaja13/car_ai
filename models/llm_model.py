import google.generativeai as genai
import os
import streamlit as st
from langchain_astradb import AstraDBVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader,PyPDFLoader
import PyPDF2 as pdf
import cassio
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough,Runnable
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

os.environ["ASTRA_DB_API_ENDPOINT"] ="https://f391f9ba-54da-4f88-af00-97b5441e213f-us-east1.apps.astra.datastax.com"
os.environ["ASTRA_DB_APPLICATION_TOKEN"]=os.getenv("ASTRA_DB_APPLICATION_TOKEN")
os.environ["HUGGING_FACE_API"] = "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"


def vector_store_config():
    embeddings = HuggingFaceEmbeddings()
    # embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vstore = AstraDBVectorStore(
        collection_name="vehicle",
        embedding=embeddings,
        token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
        api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
    )
    # print("Astra vector store configured")

    # loader=PyPDFLoader("/Users/akshatvaja/Documents/Tata_Project/vehicle_info/SAU1307.pdf")
    # # reader = pdf.PdfReader("/Users/akshatvaja/Documents/Tata_Project/vehicle_info/Automotive training.pdf")
    # # text = ""
    # # for page in range(len(reader.pages)):
    # #     page = reader.pages[page]
    # #     text += str(page.extract_text())

    # documents=loader.load()
    # # documents = text
    # text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

    # final_documents=text_splitter.split_documents(documents)
    # print(final_documents)

    # vstore.add_documents(final_documents)
    # # Checks your collection to verify the documents are embedded.
    # print(vstore.astra_db.collection("vehicle").find())
    return vstore

    



def get_response(input):
    
    vstore = vector_store_config()
    retriever = vstore.as_retriever(search_kwargs={"k": 3})

    prompt_template = """
    Answer the question based only on the supplied context. If you don't know the answer, say you don't know the answer.
    Context: {context}
    Question: {question}
    Your answer:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    model = genai.GenerativeModel(model_name="gemini-pro")

    # class MyRunnable(Runnable):
    #     def __init__(self, obj):
    #         self.obj = obj

    #     def invoke(self, input_data):
    #         # Here you need to define how your object should be invoked.
    #         # Assuming retriever has a method like 'search' that you want to call.
    #         return self.obj.search(input_data)

    # runnable_retriever = MyRunnable(retriever)

    # chain = (
    #     {"context": retriever, "question": RunnablePassthrough()}
    #     | prompt
    #     | model
    #     | StrOutputParser()
    # )

    # response = chain.invoke(input)
    # return response
    
    retrievalQA = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        memory = ConversationBufferMemory()
    )
    
    response = retrievalQA.invoke(input)
    return response['result']


    

# # if __name__ == "__main__":
# #     vector_store_config()

# import google.generativeai as genai
# import os
# import streamlit as st
# from langchain_astradb import AstraDBVectorStore
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
# import PyPDF2 as pdf
# import cassio
# from langchain.prompts import ChatPromptTemplate
# from langchain.schema.output_parser import StrOutputParser
# from langchain.schema.runnable import RunnablePassthrough, Runnable
# from langchain.chains import RetrievalQA
# from langchain.llms.base import LLM

# # Configure Generative AI with the API key
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Environment variables for AstraDB
# os.environ["ASTRA_DB_API_ENDPOINT"] = "https://f391f9ba-54da-4f88-af00-97b5441e213f-us-east1.apps.astra.datastax.com"
# os.environ["ASTRA_DB_APPLICATION_TOKEN"] = os.getenv("ASTRA_DB_APPLICATION_TOKEN")

# def vector_store_config():
#     embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vstore = AstraDBVectorStore(
#         collection_name="vehicle",
#         embedding=embedding,
#         token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
#         api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
#     )
#     return vstore

# class GoogleGenerativeModelWrapper(LLM, Runnable):
#     def __init__(self, model_name):
#         self.model = genai.GenerativeModel(model_name=model_name)

#     def _call(self, prompt, stop=None):
#         response = self.model(prompt)
#         return response["text"]

#     def invoke(self, input_data):
#         return self._call(input_data)

# def get_response(input_text):
#     vstore = vector_store_config()
#     retriever = vstore.as_retriever(search_kwargs={"k": 3})

#     prompt_template = """
#     Answer the question based only on the supplied context. If you don't know the answer, say you don't know the answer.
#     Context: {context}
#     Question: {question}
#     Your answer:
#     """
#     prompt = ChatPromptTemplate.from_template(template=prompt_template)

#     model = GoogleGenerativeModelWrapper(model_name="gemini-pro")

#     retrievalQA = RetrievalQA.from_chain_type(
#         llm=model,
#         chain_type="stuff",
#         retriever=retriever,
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": prompt}
#     )

#     response = retrievalQA.invoke(input_text)
#     return response

# # Example usage in a Streamlit app
def main():
    text = st.text_input("Enter your question:")
    if st.button("Get Response"):
        response = get_response(text)
        st.write(response)

if __name__ == "__main__":
    main()
