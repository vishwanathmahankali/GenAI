import google.generativeai as genai
import streamlit as st

genai.configure(api_key="AIzaSyCQZ4_tSrQ8MNt9K5UoqLm2iZ9DmqGYDY0")

def initialize_memory():
    if 'conversation_history' not in st.session_state:
        st.session_state['conversation_history'] = []

def get_user_input():
    st.title("AI Conversational Data Science Tutor")
    user_question = st.text_input("Ask me anything about Data Science:")
    
    if st.button("Submit"):
        return user_question
    return None

def generate_answer(user_question):
    model = genai.GenerativeModel(
        model_name='models/gemini-2.0-flash-exp',  
        system_instruction = """
            You are a highly knowledgeable Data Science tutor. Your primary focus is to answer questions about the following topics only:
            - Machine Learning (including supervised learning, unsupervised learning, reinforcement learning, deep learning, neural networks, and algorithms like decision trees, random forests, and k-means).
            - Statistics (including descriptive statistics, hypothesis testing, statistical significance, regression, probability theory, and Bayesian statistics).
            - Data Analysis (including data cleaning, feature engineering, data exploration, data wrangling, and preprocessing).
            - Programming languages and libraries used in data science, particularly Python, R, pandas, NumPy, scikit-learn, TensorFlow, and similar tools.
            - Data Visualization (including tools like matplotlib, seaborn, and visualization types such as scatter plots, histograms, and box plots).
            - Big Data technologies (Hadoop, Spark, MapReduce, ETL processes, distributed systems).
            - Algorithms, optimization, and complexity in the context of data science.

            Only answer questions related to the above fields and avoid discussing any unrelated topics such as general knowledge, unrelated industries, or non-technical subjects. Keep answers concise, clear, and focused on providing high-quality explanations of data science concepts. Keep each response short and ensure the response remains on-topic.
            """
    )
    
    conversation_context = "\n".join(st.session_state['conversation_history']) + f"\nUser: {user_question}\nAI:"
    
    response = model.generate_content(conversation_context)
    answer = response.text.strip()

    st.session_state['conversation_history'].append(f"User: {user_question}")
    st.session_state['conversation_history'].append(f"AI: {answer}")

    return answer

def main():
    initialize_memory()

    user_question = get_user_input()
    
    if user_question:
        st.write(f"Fetching answer for your question: {user_question}...")

        with st.spinner('Thinking...'):
            answer = generate_answer(user_question)
        
        st.write(answer)
    
    if st.session_state['conversation_history']:
        st.write("Conversation so far:")
        for exchange in st.session_state['conversation_history']:
            st.write(exchange)

if __name__ == "__main__":
    main()
