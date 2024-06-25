import streamlit as st
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import os
from pandas_agent import create_agent, process_query

st.title("📊 Lung Diseases Analysizer")

# Sidebar for OpenAI API Key and Model Selection
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
model = st.sidebar.selectbox("Select OpenAI Model", ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview","gpt-4-0613"])

# File uploader for multiple CSV files
uploaded_files = st.file_uploader("Choose CSV files", accept_multiple_files=True, type="csv")

if api_key and uploaded_files:
    os.environ["OPENAI_API_KEY"] = api_key
    
    # Read and concatenate all uploaded CSV files
    combined_df = [pd.read_csv(file) for file in uploaded_files]
    # combined_df = combine_dataframes(combined_dfs)
    
    st.write("Combined DataFrame:")
    # st.dataframe(combined_df)
    
    # Create the pandas dataframe agent
    agent = create_agent(api_key, model, combined_df)
    
    # Text area for user query
    user_query = st.text_area("Enter your query about the data:")
    
    if st.button("Run Query"):
        if user_query:
            with st.spinner("Processing query..."):
                response = process_query(agent, user_query)
                st.write("Response:")
                st.write(response)
        else:
            st.warning("Please enter a query.")
    
    # Add example queries
    st.sidebar.markdown("---")
    st.sidebar.subheader("Example Queries:")
    example_query1 = """Can you rank the top 5 countries Based on the scenario where patient population is weighted at 50% 
    and competition weighted at 50%?"""
    example_query2 = """Can you weigh Patient Population at 25%, Competition at 25%, 
    and Country Operations at 50% for Obstructive Lung Diseases to give me the prioritized country list? Please in proper format"""
    
    if st.sidebar.button("Use Example Query 1"):
        st.text_area("Query", value=example_query1)
    if st.sidebar.button("Use Example Query 2"):
        st.text_area("Query", value=example_query2)

else:
    st.warning("Please enter your OpenAI API Key and upload at least one CSV file to proceed.")
