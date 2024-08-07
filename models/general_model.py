import streamlit as st
# import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

# loading the environment
load_dotenv()

#configuring the api key
# genai.configure(api_key=os.getenv('GOOGLE_API_KEY'));
llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.environ["GOOGLE_API_KEY"])

#model intialisation

# calling the gemini response
def gemini_response(question):
    prompt_template = """
    You are an expert in information and troubleshooting in vehicle-related tasks. Now give a brief answer for the asked question.
    Question: {question}
    Your answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=['question'])
    chain = prompt | llm
    response = chain.invoke(question)
    return response

