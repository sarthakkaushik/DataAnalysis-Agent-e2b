import streamlit as st
import os
from csv_query_system import query_csv_system

def main():
    st.title("CSV Query System")
    st.write("""# My first app Hello *world!*""")
    # OpenAI API Key input
    openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")

    # File uploader
    uploaded_files = st.file_uploader("Choose CSV files", accept_multiple_files=True, type="csv")

    if uploaded_files and openai_api_key:
        csv_files = []
        for uploaded_file in uploaded_files:
            # Save uploaded file temporarily
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            csv_files.append(uploaded_file.name)

        # Display uploaded files
        st.write(f"Uploaded files: {', '.join(csv_files)}")

        # Query input
        query = st.text_input("Enter your question about the CSV files:")

        if query:
            if not csv_files:
                st.error("Please upload at least one CSV file before querying.")
            else:
                with st.spinner('Processing your query...'):
                    try:
                        # Process the query
                        result = query_csv_system(query, csv_files, openai_api_key)

                        # Display the answer
                        st.subheader("Answer:")
                        st.write(result)
                    except Exception as e:
                        st.error(f"An error occurred while processing your query: {str(e)}")

        # Clean up temporary files
        for file in csv_files:
            if os.path.exists(file):
                os.remove(file)

if __name__ == "__main__":
    main()