import google.generativeai as genai
import streamlit as st

genai.configure(api_key="AIzaSyCQZ4_tSrQ8MNt9K5UoqLm2iZ9DmqGYDY0")  

def get_user_input():
    st.title("AI Powered Travel Planner")
    source = st.text_input("Enter Source Location:")
    destination = st.text_input("Enter Destination Location:")

    if st.button("Get Travel Options"):
        return source, destination
    return None, None

def generate_travel_query(source, destination):
    model = genai.GenerativeModel(
        model_name='models/gemini-2.0-flash-exp',  
        system_instruction="""You are a travel assistant. Based on the user's input, provide travel options such as cab, bus, 
                              train, and flights. Include estimated costs for each travel method."""
    )
    
    prompt = f"Provide travel options from {source} to {destination} including cab, bus, train, and flight with estimated costs."
    
    if source and destination:
        response = model.generate_content(prompt)
        return response.text

def main():
    source, destination = get_user_input()
    if source and destination:
        st.write(f"Fetching travel options from {source} to {destination}...")
        
        # Show a spinner while waiting for the result
        with st.spinner('Fetching travel options...'):
            travel_recommendations = generate_travel_query(source, destination)
        
        st.write(travel_recommendations)

if __name__ == "__main__":
    main()
