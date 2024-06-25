from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import pandas as pd

def create_agent(api_key, model, df):
    """
    Create a pandas dataframe agent.
    
    :param api_key: OpenAI API key
    :param model: OpenAI model to use
    :param dataframe: Pandas DataFrame to analyze
    :return: Langchain agent
    """
    print( "Mode:", model)
    print("Open API Key:", api_key)
    llm = ChatOpenAI(api_key=api_key, model=model, temperature=0)
  
    agent = create_pandas_dataframe_agent(llm, df, verbose=True,allow_dangerous_code=True,)
    # create_pandas_dataframe_agent(llm, dataframe, verbose=True,allow_dangerous_code=True, handle_parsing_errors=True)
    return agent

def process_query(agent, query):
    """
    Process a query using the pandas dataframe agent.
    
    :param agent: Langchain agent
    :param query: Query string
    :return: Query result or error message
    """
    try:
        result = agent.run(query)
        return result
    except Exception as e:
        return f"An error occurred: {str(e)}"