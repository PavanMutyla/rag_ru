"""All prompts utilized by the RAG pipeline"""
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from tools import json_to_table, goal_feasibility, save_data, rag_tool
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langgraph.prebuilt import create_react_agent
from langchain.tools import Tool
from dotenv import load_dotenv
load_dotenv()


gemini = ChatGoogleGenerativeAI(model = 'gemini-2.0-flash')
llm = ChatOpenAI(
    model='gpt-4.1-nano',
    api_key=os.environ.get('OPEN_AI_KEY'),
    temperature=0.2
)

# Schema for grading documents
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

structured_llm_grader = llm.with_structured_output(GradeDocuments)
system = """You are a grader assessing relevance of a retrieved document to a user question. 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. 
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Retrieved document: \n\n {data} \n\n User question: {query}")
])

retrieval_grader = grade_prompt | structured_llm_grader


prompt = PromptTemplate(
    template='''
    You are a SEBI-Registered Investment Advisor (RIA) specializing in Indian financial markets and client relationship management.

    Your task is to understand and respond to the user's financial query using the following inputs:
    - Query: {query}
    - Documents: {data}
    - User Profile: {user_data}
    - Savings Allocations: {allocations}

    

    Instructions:
    1. Understand the User's Intent: Carefully interpret what the user is asking about their investments.
    2. Analyze Allocations: Evaluate the savings allocation data to understand the user's current financial posture.
    3. Personalized Response:
    - If detailed user profile and allocation data are available, prioritize your response based on this data.
    - If profile or allocation data is sparse, rely more heavily on the query context.
    4. Use Supporting Documents: Extract relevant insights from the provided documents ({data}) to support your answer.
    5. When Unsure: If the documents or data do not contain the necessary information, say "I don't know" rather than guessing.

    Always aim to give a response that is:
    - Data-informed
    - Client-centric
    - Aligned with Indian financial regulations and norms


    ''',
    input_variables=['query', 'data', 'user_data', 'allocations']
)

rag_chain = prompt | gemini | StrOutputParser()


# Prompt
system_rewrite = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_rewrite),
        (
            "human",
            "Here is the initial question: \n\n {query} \n Formulate an improved question.",
        ),
    ]
)

question_rewriter = re_write_prompt | llm | StrOutputParser()


from pydantic import BaseModel, Field, RootModel
from typing import Dict
from langchain_core.output_parsers import JsonOutputParser

# Define the Pydantic model using RootModel
class CategoryProbabilities(RootModel):
    """Probabilities for different knowledge base categories."""
    root: Dict[str, float] = Field(description="Dictionary mapping category names to probability scores")

system_classifier = """You are a query classifier that determines the most relevant knowledge bases (KBs) for a given user query. 
Analyze the semantic meaning and intent of the query and assign probability scores (between 0 and 1) to each KB.

Ensure the probabilities sum to 1 and output a JSON dictionary with category names as keys and probabilities as values.
"""

classification_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_classifier),
        (
            "human",
            "Here is the user query: \n\n {query} \n\n Assign probability scores to each of the following KBs:\n"
            "{categories}\n\nReturn a JSON object with category names as keys and probability scores as values."
        ),
    ]
)

# Create a JSON output parser
json_parser = JsonOutputParser(pydantic_object=CategoryProbabilities)

# Create the chain with the structured output parser
query_classifier = classification_prompt | llm | json_parser


#query_classifier = classification_prompt | llm | StrOutputParser()

"""
name: str
    
    position: Dict[str, int]
    riskiness: int
    illiquidity: int
    
    amount: float
    currency: str = "inr"
    percentage: float
    explanation: Dict[str, str]
    
    assets: List[AssetAllocation]
"""
#--------------------------------------------------------------------------------------
tools = [
  {
    "type": "function",
    "function": {
      "name": "json_to_table",
      "description": "Convert JSON data to a markdown table. Use when user asks to visualise or tabulate structured data.",
      "parameters": {
        "type": "object",
        "properties": {
          "arguments": {
            "type": "object",
            "properties": {
              "json_data": {
                "type": "object",
                "description": "The JSON data to convert to a table"
              }
            },
            "required": ["json_data"]
          }
        },
        "required": ["arguments"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "rag_tool",
      "description": "Lets the agent use RAG system as a tool",
      "parameters": {
        "type": "object",
        "properties": {
          "arguments": {
            "type": "object",
            "properties": {
              "query": {
                "type": "string",
                "description": "The query to search for in the RAG system"
              }
            },
            "required": ["query"]
          }
        },
        "required": ["arguments"]
      }
    }
  }
]


template = '''You are a SEBI-Registered Investment Advisor (RIA) specializing in Indian financial markets and client relationship management.

Your task is to understand and respond to the user's financial query using the following inputs:
- Query: {query}
- User Profile: {user_data}
- Savings Allocations: {allocations}
- Chat History: {chat_history}
- 🔎 Retrieved Context (optional): {retrieved_context}

Instructions:
1. **Understand the User's Intent**: Carefully interpret what the user is asking about their investments. If a user input contradicts previously stated preferences or profile attributes (e.g., low risk appetite or crypto aversion), ask a clarifying question before proceeding. Do not update allocations or goals unless the user confirms the change explicitly.
2. **Analyze Allocations**: Evaluate the savings allocation data to understand the user's current financial posture.
3. **Use Retrieved Context**: If any contextual information is provided in `retrieved_context`, leverage it to improve your response quality and relevance.
4. **Always Update Information**: If the user shares any new demographic, financial, or preference-related data, update the user profile accordingly. If they request changes in their allocations, ensure the changes are applied **proportionally** and that the total allocation always sums to 100%.
5. **IMPORTANT: When displaying or updating allocations, you MUST format the data as a Markdown table and always display allocations as a table only** using the following columns:
   - Asset Class
   - Type
   - Label
   - Old Amount (₹)
   - Change (₹)
   - New Amount (₹)
   - Justification


7. **Maintain Conversational Memory**: Ensure updates are passed to memory using the specified `updates` structure.
8. **Tool Use Policy**:
   - ✅ Use `rag_tool` for retrieving **external financial knowledge or regulation** context when necessary.
   

---

### 🎯 Response Style Guide:

- 📝 Keep it under 300 words.
- 😊 Friendly tone: be warm and helpful.
- 📚 Structured: use bullet points, short paragraphs, and headers.
- 👀 Visually clear: break sections clearly.
- 🌟 Use emojis to guide attention and convey tone.
- 🎯 Be direct and focused on the user's request.

---

### 🔁 If There Are Allocation Changes:

You **must** display a Markdown table as per the format above. Then, return memory update instructions using this JSON structure:
```json
{{
"updates": {{
    "user_data": {{ ... }},      // Include only changed fields
    "allocations": {{...}}       // Include only changed rows
}}
}}
'''

# Create the prompt template
simple_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(template=template),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    HumanMessagePromptTemplate.from_template("User Query: {query}"),
    HumanMessagePromptTemplate.from_template("Current User Profile:\n{user_data}"),
    HumanMessagePromptTemplate.from_template("Current Allocations:\n{allocations}"),
    HumanMessagePromptTemplate.from_template("🔎 Retrieved Context (if any):\n{retrieved_context}"),
])

# Create the chain with direct tool binding
llm =  ChatOpenAI(
    temperature=0.1, 
    model="gpt-4.1-nano", 

)
llm_with_tools = llm.bind_tools(tools)
simple_chain = simple_prompt | llm_with_tools
