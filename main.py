from language_conversion.speech_to_text import speech_to_text as speech2text
from language_conversion.text_to_speech import text_to_speech as text2speech
from models.rag import get_response 
from models.classify import model_load,pred_function
from models.general_model import gemini_response
import streamlit as st
import json
import string

def main():
    st.set_page_config("Car Troubelshooter")
    st.header("Ask Anything")
    # st.title("Speech-to-Text and Text-to-Speech App")
    # if st.button("Load_models"):
    #     model_load()
    #     st.success("Speech recognition successful!")
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    st.write("Click the button below and start speaking:")
    if st.button("Start Recording"):
        st.write("Recording...")


        text = speech2text()
        # text = "Hello"
        if text:
            st.success("Speech recognition successful!")
            st.write("You said: " + text)
            # text2speech(text)
            # prediction = pred_function(text)
            # if prediction == 1:
            response = get_response(text)
            if response and response == "0":
                text2speech("finding best route to required location")
                pass
            elif response and response == "1":
                # st.write("Sorry, but this question is not in my domain on which i am trained")
                response = gemini_response(text)
                st.write("Response: " + response)

                punctuation=string.punctuation
                mapping=str.maketrans("","",punctuation)
                response = response.translate(mapping)

                text2speech(response)
            elif response:
                text2speech("searching my knowledge sphere")
                st.write("Response: " + response)
                punctuation=string.punctuation
                mapping=str.maketrans("","",punctuation)
                response = response.translate(mapping)
                text2speech(response)
            else:
                punctuation=string.punctuation
                mapping=str.maketrans("","",punctuation)
                response = response.translate(mapping)
                text2speech(response)
            

            st.session_state.chat_history.append({"user": text, "bot": response})
           
        else:
            st.warning("No speech detected or recognition failed.")
    

    if st.session_state.chat_history:
        chat_history_json = json.dumps(st.session_state.chat_history, indent=4)
        st.download_button(
            label="Download Chat History",
            data=chat_history_json,
            file_name="chat_history.json",
            mime="application/json"
        )
    # st.header("Text-to-Speech")
    # text_input = st.text_input("Enter text to convert to speech:")
    # if st.button("Convert to Speech"):
    #     if text_input:
    #         st.write("Playing speech...")
            
    #     else:
    #         st.write("Playing speech...")
    #         text2speech()
    #         # st.warning("Please enter some text to convert.")

if __name__ == "__main__":
    main()