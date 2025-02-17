
import google.generativeai as genai
import streamlit as st

genai.configure(api_key="AIzaSyCQZ4_tSrQ8MNt9K5UoqLm2iZ9DmqGYDY0")

st.title("An AI Code Review")

user_prompt = st.text_area("Enter your Python code here:")

def get_code_review_and_fix(user_code):
    model = genai.GenerativeModel(model_name='models/gemini-2.0-flash-exp',
                                #   'models/gemini-1.5-pro-latest',
                                  system_instruction="""Analyze the submitted code and identify bugs, errors or areas of improvement.
                                                           Provide the fixed code snippets.
                                                           """)

    if user_code:
        response = model.generate_content(user_code)
        return response.text

if st.button("Generate"):
    if user_prompt:
        review_response = get_code_review_and_fix(user_prompt)

        st.subheader("Code Review:")
        st.write(review_response)
