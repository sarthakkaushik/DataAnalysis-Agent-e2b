from typing import Dict, TypedDict, List, Tuple
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage
from langchain.tools import Tool
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolInvocation
import os
from dotenv import load_dotenv


class State(TypedDict):
    messages: List[BaseMessage]
    query: str
    current_csv: str
    results: Dict[str, str]
    continue_querying: str

def create_single_csv_agent(file_path, openai_api_key):
    return create_csv_agent(
        ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", openai_api_key=openai_api_key),
        file_path,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True,
        handle_parsing_errors=True
    )

def create_csv_tools(csv_files, openai_api_key):
    tools = []
    for file in csv_files:
        agent = create_single_csv_agent(file, openai_api_key)
        tool = Tool(
            name=f"Query_{file}",
            func=lambda q, file=file, agent=agent: agent.run(q),
            description=f"Useful for querying the {file} dataset"
        )
        tools.append(tool)
    return tools

def decide_csv(state, csv_files, openai_api_key):
    query = state['query']
    prompt = ChatPromptTemplate.from_messages([
        ("human", "Based on the following query, which CSV file should I query? Options are: {csv_files}. Query: {query}"),
        ("human", "Respond with just the name of the CSV file, nothing else.")
    ])
    model = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
    response = model.invoke(prompt.format(csv_files=", ".join(csv_files), query=query))
    return {"current_csv": response.content}

def query_csv(state, tools):
    current_csv = state['current_csv']
    query = state['query']
    tool = next((tool for tool in tools if tool.name == f"Query_{current_csv}"), None)
    if tool is None:
        return {"results": {current_csv: "Error: CSV file not found"}}
    result = tool.func(query)
    return {"results": {**state['results'], current_csv: result}}

def decide_if_done(state, openai_api_key):
    results = state['results']
    query = state['query']
    prompt = ChatPromptTemplate.from_messages([
        ("human", "Based on the following query and results, do we need to query more CSV files? Query: {query}\nResults: {results}"),
        ("human", "Respond with either 'YES' if we need to query more files, or 'NO' if we have sufficient information.")
    ])
    model = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
    response = model.invoke(prompt.format(query=query, results=results))
    return {"continue_querying": "YES" if response.content.strip().upper() == "YES" else "NO"}

def compile_answer(state, openai_api_key):
    results = state['results']
    query = state['query']
    prompt = ChatPromptTemplate.from_messages([
        ("human", "Based on the following query and results from multiple CSV files, provide a comprehensive answer. Query: {query}\nResults: {results}"),
        ("human", "Provide a detailed answer that synthesizes information from all relevant CSV files.")
    ])
    model = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
    response = model.invoke(prompt.format(query=query, results=results))
    return {"messages": state['messages'] + [response]}

def create_graph(csv_files, tools, openai_api_key):
    workflow = StateGraph(State)
    workflow.add_node("decide_csv", lambda state: decide_csv(state, csv_files, openai_api_key))
    workflow.add_node("query_csv", lambda state: query_csv(state, tools))
    workflow.add_node("decide_if_done", lambda state: decide_if_done(state, openai_api_key))
    workflow.add_node("compile_answer", lambda state: compile_answer(state, openai_api_key))
    workflow.set_entry_point("decide_csv")
    workflow.add_edge("decide_csv", "query_csv")
    workflow.add_edge("query_csv", "decide_if_done")
    workflow.add_conditional_edges(
        "decide_if_done",
        lambda x: x["continue_querying"],
        {
            "YES": "decide_csv",
            "NO": "compile_answer"
        }
    )
    workflow.add_edge("compile_answer", END)
    return workflow.compile()

def query_csv_system(query, csv_files, openai_api_key):
    tools = create_csv_tools(csv_files, openai_api_key)
    graph = create_graph(csv_files, tools, openai_api_key)
    result = graph.invoke({
        "messages": [],
        "query": query,
        "current_csv": "",
        "results": {},
        "continue_querying": "YES"
    })
    return result['messages'][-1].content