import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain.tools import tool
import pandas as pd 
import json 
import re
from copy import deepcopy
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel
from typing import Any, Optional

api_key = os.getenv('PINCEONE_API_KEY')

class JsonToTableInput(BaseModel):
    json_data: Any

class RagToolInput(BaseModel):
    query: str

# Define the tools with proper validation
def json_to_table(input_data: JsonToTableInput):
    """Convert JSON data to a markdown table. Use when user asks to visualise or tabulate structured data."""
    json_data = input_data.json_data
    
    if isinstance(json_data, str):
        try:
            json_data = json.loads(json_data)
        except:
            # If json_data has parsing issues, try to work with it directly
            pass
    
    # Handle a common case in the prompt where 'allocations' might be a nested key
    if isinstance(json_data, dict) and 'allocations' in json_data:
        json_data = json_data['allocations']
    
    # Ensure we have a valid list or dict to convert to DataFrame
    if not json_data:
        json_data = [{"Note": "No allocation data available"}]
    
    df = pd.json_normalize(json_data)
    markdown_table = df.to_markdown(index=False)
    print(f"[DEBUG] json_to_table output:\n{markdown_table}")
    
    return markdown_table

def rag_tool(input_data: RagToolInput):
    """Lets the agent use RAG system as a tool"""
    query = input_data.query
    
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        dimensions=384  
    )
    kb = PineconeVectorStore(
        pinecone_api_key=os.environ.get('PINCEONE_API_KEY'),
        index_name='rag-rubic',
        namespace='vectors_lightmodel'
    )
    retriever = kb.as_retriever(search_kwargs={"k": 10})
    context = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in context])

@tool
def goal_feasibility(goal_amount: float, timeline: float, current_savings: float, income : float) -> dict:
    """Evaluate if a financial goal is feasible based on user income, timeline, and savings. Use when user asks about goal feasibility."""
    # Input checks
    if timeline <= 0:
        return {
            "feasible": False,
            "status": "Invalid",
            "monthly_required": 0,
            "reason": "Timeline must be greater than 0 months."
        }

    # Calculate the remaining amount
    remaining_amount = goal_amount - current_savings
    if remaining_amount <= 0:
        return {
            "feasible": True,
            "status": "Already Achieved",
            "monthly_required": 0,
            "reason": "You have already met or exceeded your savings goal."
        }

    monthly_required = remaining_amount / timeline
    income_ratio = monthly_required / income

    # Feasibility classification
    if income_ratio <= 0.3:
        status = "Feasible"
        feasible = True
        reason = "The required savings per month is manageable for an average income."
    elif income_ratio <= 0.7:
        status = "Difficult"
        feasible = False
        reason = "The required monthly saving is high but may be possible with strict budgeting."
    else:
        status = "Infeasible"
        feasible = False
        reason = "The required monthly saving is unrealistic for an average income."

    return {
        "feasible": feasible,
        "status": status,
        "monthly_required": round(monthly_required, 2),
        "reason": reason
    }


@tool
def save_data(new_user_data:dict, new_alloc_data:dict):
    "Saves the updated user_data and allocations data in a json file."
    path = os.getenv("DATA_PATH", ".")
    save_path = os.path.join(path, "updated_json")
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "updated_user_data.json"), "w") as f:
        json.dump(new_user_data, f, indent=2)

    with open(os.path.join(save_path, "updated_allocations.json"), "w") as f:
        json.dump(new_alloc_data, f, indent=2)

