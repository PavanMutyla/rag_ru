import sys
import os

from langgraph.graph import START, END, StateGraph
from langchain_openai import OpenAIEmbeddings
from chains import simple_chain, llm_with_tools
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from typing import TypedDict, Optional, Dict, List, Union, Annotated
from langchain_core.messages import AnyMessage #human or AI message
from langgraph.graph.message import add_messages # reducer in langgraph 
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langgraph.checkpoint.memory import MemorySaver
import json
import langchain
from tools import json_to_table, goal_feasibility, rag_tool, save_data
import re

from dotenv import load_dotenv
load_dotenv()

memory = MemorySaver()
config = {"thread_id":"sample"}
tools = [json_to_table, rag_tool]
#tool_executor = ToolExecutor([json_to_table, goal_feasibility])
json_to_table_node = ToolNode([json_to_table])

rag_tool_node = ToolNode([rag_tool])
class Graph(TypedDict):
    query: Annotated[list[AnyMessage], add_messages]
    #chat_history : List[BaseMessage]
    user_data : Dict
    allocations : Dict 
    #data : str 
    output : Dict
    retrieved_context: str

def chat(state):
    inputs = {
        "query": state["query"],
        "user_data": state["user_data"],
        "allocations": state["allocations"],
        #"data": state["data"],
        "chat_history": state["query"],  # If you treat `query` as history
        "retrieved_context": state.get("retrieved_context", "")
    }

    result = simple_chain.invoke(inputs)
    #print(result)

    return {
        "query": state["query"],
        "user_data": state["user_data"],
        "allocations": state["allocations"],
        #"data": state["data"],
        "retrieved_context": "",  # clear after use
        "output": result.content
    }

def json_to_table_node(state):
    tool_output = json_to_table(state["allocations"])  # Or whatever your input is
    return AIMessage(content=tool_output)

def tools_condition(state):
    last_message = state["query"][-1]  # Last user or AI message
    if isinstance(last_message, AIMessage):
        tool_calls = getattr(last_message, "tool_calls", None)
        
        # Check if tool calls exist and handle them
        if tool_calls:
            tool_name = tool_calls[0].get('name', '')  # Safely access the tool name
            
            if tool_name == "json_to_table":
                return "show_allocation_table"
            
            elif tool_name == "rag_tool":
                return "query_rag"
            else:
                return "tools"  # Fallback in case of unknown tool names
    return "END"  # End the flow if no tool calls are found


# ---- GRAPH SETUP ----
graph = StateGraph(Graph)

# Nodes
graph.add_node("chat", chat)
graph.add_node("show_allocation_table", json_to_table_node)
#graph.add_node("save_data_info", save_data_node)
graph.add_node("query_rag", rag_tool_node)
graph.add_node("tool_output_to_message", lambda state: AIMessage(content=state["tool_output"]))


#graph.add_node("tools", ToolNode(tools))  # fallback for other tools


# Main flow
graph.add_edge(START, "chat")
graph.add_conditional_edges("chat", tools_condition)

# Each tool goes back to chat
graph.add_edge("show_allocation_table", "chat")
#graph.add_edge("save_data_info", "chat")
graph.add_edge("query_rag", "chat")

# End after a loop
graph.add_edge("chat", END)


# Compile
app = graph.compile(checkpointer=memory)

'''
with open('/home/pavan/Desktop/FOLDERS/RUBIC/RAG_without_profiler/RAG_rubik/sample_data/sample_alloc.json', 'r') as f:
    data = json.load(f)
with open('/home/pavan/Desktop/FOLDERS/RUBIC/RAG_without_profiler/RAG_rubik/sample_data/sample_alloc.json', 'r') as f:
    allocs = json.load(f)
inputs = {
    "query":"display my investments.",
    "user_data":data,
    "allocations":allocs,
    "data":"",
    "chat_history": [],
    
}

langchain.debug = True
print(app.invoke(inputs, config={"configurable": {"thread_id": "sample"}}).get('output'))
#print(json_to_table.args_schema.model_json_schema())
'''