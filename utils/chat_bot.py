import streamlit as st
import google.generativeai as genai

api_key="AIzaSyC0HQTg-oaShAG_0_GgxIqUCTTjtqRRK9E"

genai.configure(api_key=api_key)

def my_chatbot():
    model =  genai.GenerativeModel("gemini-pro")

    chat = model.start_chat(history=[])


    def get_gemini_response(question):
        response= chat.send_message(question, stream=True)

        return response
    
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    
    input = st.text_input("Enter Question", key="input")

    submit= st.button("Ask Question")

    if submit and input:
        response = get_gemini_response(input)

        st.session_state["chat_history"].append(("You", input))

        st.subheader("Response:")
        for chunk in response:
            st.write(chunk.text)

            st.session_state["chat_history"].append(("Bot", chunk.text))
        
        st.subheader("History")
        
        for role, text in st.session_state["chat_history"]:
            st.markdown(f"""### {role}""")
            st.write(text)








