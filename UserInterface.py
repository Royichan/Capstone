import streamlit as st
import requests

# Function to send features to an API for localization prediction
def sendSOP(text):
    # Create the request body
    body = {"text":text}
    try:
        # Send a POST request to the localization prediction API
        response = requests.post(url="http://localhost:105/sopPrediction", json=body)
    except requests.exceptions.ConnectionError as e:
         # Handle connection error to the API
        st.text("API Connection Failed")
        return []

    if response.status_code == 200:
        Score = response.json()['score']
        print(response.json())
        return Score
    else:
        # Handle API call failure
        st.text("API Call Failed")

def main():
     # Streamlit app title and user input section
    st.title("SOP Analysis")
    text = st.text_area("Enter the SOP Text", height=200)
    
    
    # Button to trigger localization prediction
    if st.button("Score"):
        if len(text) != 0:
            score = sendSOP(text) # Call the prediction function
            if score == 1.0:
                st.write(f"The SOP will Probably get Accepted")
            else:
                st.write(f"The SOP will Probably get Rejected")
        else:
            st.write(f"Input doesn't meet prerequisite, Input length is {len(text)}")
if __name__ == "__main__":
    main()
