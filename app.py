import sys
import os

import pandas as pd
import langchain
os.environ['STREAMLIT_SERVER_ENABLE_STATIC_SERVING'] = 'false'

from simple_rag import app

import streamlit as st
import json
from io import StringIO
import tiktoken
import time
from langchain_community.document_loaders import PyMuPDFLoader
import traceback
import sqlite3  # Import SQLite
from dotenv import load_dotenv
load_dotenv()

import uuid  # Import the UUID library

# Token limits
config={"configurable": {"thread_id": "sample"}}
GPT_LIMIT = 128000
GEMINI_LIMIT = 1000000
config={"configurable": {"thread_id": "sample"}}
# Token counters
def count_tokens_gpt(text):
    enc = tiktoken.encoding_for_model("gpt-4")
    return len(enc.encode(text))

def count_tokens_gemini(text):
    return len(text.split())  # Approximation

# Calculate tokens for the entire context window
def calculate_context_window_usage(json_data=None):
    # Reconstruct the full conversation context
    full_conversation = ""
    for sender, message in st.session_state.chat_history:
        full_conversation += f"{sender}: {message}\n\n"
    
    # Add JSON context if provided
    if json_data:
        full_conversation += json.dumps(json_data)
        
    gpt_tokens = count_tokens_gpt(full_conversation)
    gemini_tokens = count_tokens_gemini(full_conversation)
    
    return gpt_tokens, gemini_tokens




# Page configuration
st.set_page_config(page_title="📊 RAG Chat Assistant", layout="wide")

# --- Database setup ---
# DATABASE_PATH = "Data/chat_history.db"  # Original database path
SESSION_DB_DIR = "Data/sessions"  # Directory to store individual session DBs

def initialize_session_database(session_id):
    """Initializes a new database for a chat session."""
    db_path = os.path.join(SESSION_DB_DIR, f"{session_id}.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender TEXT,
            message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    return db_path

def save_message(db_path, sender, message):
    """Saves a message to the specified session database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chat_history (sender, message) VALUES (?, ?)", (sender, message))
    conn.commit()
    conn.close()

def clear_chat_history(db_path):
    """Clears the chat history in the specified session database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chat_history")
    conn.commit()
    conn.close()

# Initialize session DB directory
if not os.path.exists(SESSION_DB_DIR):
    os.makedirs(SESSION_DB_DIR)

# --- Session state setup ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        ("assistant", "👋 Hello! I'm your RAG assistant. Please upload your JSON files and ask me a question about your portfolio.")
    ]
if "processing" not in st.session_state:
    st.session_state.processing = False
if "total_gpt_tokens" not in st.session_state:
    st.session_state.total_gpt_tokens = 0  # Total accumulated
if "total_gemini_tokens" not in st.session_state:
    st.session_state.total_gemini_tokens = 0  # Total accumulated
if "window_gpt_tokens" not in st.session_state:
    st.session_state.window_gpt_tokens = 0  # Current context window
if "window_gemini_tokens" not in st.session_state:
    st.session_state.window_gemini_tokens = 0  # Current context window

# Generate a unique session ID if one doesn't exist
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.session_db_path = initialize_session_database(st.session_state.session_id)  # Initialize session DB

# --- Load chat history from the session database ---
def load_chat_history(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT sender, message FROM chat_history ORDER BY timestamp")
    history = cursor.fetchall()
    conn.close()
    return history


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Go one level up to reach RAG_rubik/
PROJECT_ROOT = os.path.dirname(BASE_DIR)
print(PROJECT_ROOT, BASE_DIR)
# --- Layout: Chat UI Left | Progress Bars Right ---
col_chat, col_progress = st.columns([3, 1])

# --- LEFT COLUMN: Chat UI ---
with col_chat:
    st.title("💬 RAG Assistant")

    with st.expander("📂 Upload Required JSON Files", expanded=True):
        # user_data_file = st.file_uploader("Upload user_data.json", type="json", key="user_data")
        # allocations_file = st.file_uploader("Upload allocations.json", type="json", key="allocations")
        
        user_data_path = os.getenv('USER_DATA_PATH')
        allocations_path = os.getenv('ALLOCATIONS_PATH')

        try:
            with open(user_data_path, 'r') as f:
                user_data = json.load(f)
        except FileNotFoundError:
            st.error(f"Error: user_data.json not found at {user_data_path}")
            user_data = None
        except json.JSONDecodeError:
            st.error(f"Error: Could not decode user_data.json. Please ensure it is valid JSON.")
            user_data = None

        try:
            with open(allocations_path, 'r') as f:
                allocations = json.load(f)
        except FileNotFoundError:
            st.error(f"Error: allocations.json not found at {allocations_path}")
            allocations = None
        except json.JSONDecodeError:
            st.error(f"Error: Could not decode allocations.json. Please ensure it is valid JSON.")
            allocations = None

        if user_data:
            sematic = user_data.get("sematic", {})
            demographic = sematic.get("demographic", {})
            financial = sematic.get("financial", {})
            episodic = user_data.get("episodic", {}).get("prefrences", [])

            col1, col2, col3 = st.columns(3)

            with col1:
                            st.markdown("### 🧾 **Demographic Info**")
                            for key, value in demographic.items():
                                st.markdown(f"- **{key.replace('_', ' ').title()}**: {value}")

            with col2:
                            st.markdown("### 📊 **Financial Status**")
                            for key, value in financial.items():
                                st.markdown(f"- **{key.replace('_', ' ').title()}**: {value}")

            with col3:
                            st.markdown("### ⚙️ **Preferences & Goals**")
                            st.markdown("**User Preferences:**")
                            for pref in user_data.get("episodic", {}).get("prefrences", []):
                                st.markdown(f"- {pref.capitalize()}")
                            st.markdown("**Goals:**")
                            for goal in user_data.get("episodic", {}).get("goals", []):
                                for k, v in goal.items():
                                    st.markdown(f"- **{k.replace('_', ' ').title()}**: {v}")


       

        if "allocations" not in st.session_state:
            st.session_state.allocations = allocations

        if st.session_state.allocations:
            try:
                # allocations = json.load(StringIO(allocations_file.getvalue().decode("utf-8")))
                st.markdown("### 💼 Investment Allocations")

                # Flatten data for display
                records = []
                for asset_class, entries in st.session_state.allocations.items():
                    for item in entries:
                        records.append({
                            "Asset Class": asset_class.replace("_", " ").title(),
                            "Type": item.get("type", ""),
                            "Label": item.get("label", ""),
                            "Amount (₹)": item.get("amount", 0)
                        })

                df = pd.DataFrame(records)
                st.dataframe(df)

            except Exception as e:
                st.error(f"Failed to parse allocations.json: {e}")


        
        # Clear chat button
        if st.button("Clear Chat"):
            st.session_state.chat_history = [
                ("assistant", "👋 Hello! I'm your RAG assistant. Please upload your JSON files and ask me a question about your portfolio.")
            ]
            st.session_state.total_gpt_tokens = 0
            st.session_state.total_gemini_tokens = 0
            st.session_state.window_gpt_tokens = 0
            st.session_state.window_gemini_tokens = 0
            
            # Clear the chat history in the session database
            clear_chat_history(st.session_state.session_db_path)
       

            st.rerun()

    st.markdown("---")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for sender, message in st.session_state.chat_history:
            if sender == "user":
                st.chat_message("user").write(message)
            else:
                st.chat_message("assistant").write(message)
        
        # Show thinking animation if processing
        if st.session_state.processing:
            thinking_placeholder = st.empty()
            with st.chat_message("assistant"):
                for i in range(3):
                    for dots in [".", "..", "..."]:
                        thinking_placeholder.markdown(f"Thinking{dots}")
                        time.sleep(0.3)

    # Input box at the bottom
    user_input = st.chat_input("Type your question...")

    if user_input and not st.session_state.processing:
        # Set processing flag
        st.session_state.processing = True
        
        # Add user message to history immediately
        st.session_state.chat_history.append(("user", user_input))
        save_message(st.session_state.session_db_path, "user", user_input)  # Save user message to session DB
        
        # Force a rerun to show the message and thinking indicator
        st.rerun()

# This part runs after the rerun if we're processing
if st.session_state.processing:
    if not user_data or not allocations:
        st.session_state.chat_history.append(("assistant", "⚠️ Please upload both JSON files before asking questions."))
        st.session_state.processing = False
        st.rerun()
    else:
        try:
            # Load JSONs
            # user_data = json.load(StringIO(user_data_file.getvalue().decode("utf-8")))
            # allocations = json.load(StringIO(allocations_file.getvalue().decode("utf-8")))
            
            # Combined JSON data (for token calculation)
            combined_json_data = {"user_data": user_data, "allocations": allocations}

            # Get the last user message
            last_user_message = next((msg for sender, msg in reversed(st.session_state.chat_history) if sender == "user"), "")
            
            # Count tokens for this user message
            user_msg_gpt_tokens = count_tokens_gpt(last_user_message)
            user_msg_gemini_tokens = count_tokens_gemini(last_user_message)
            
            # Add to accumulated totals
            st.session_state.total_gpt_tokens += user_msg_gpt_tokens
            st.session_state.total_gemini_tokens += user_msg_gemini_tokens
            
            # Calculate context window usage (conversation + JSON data)
            window_gpt, window_gemini = calculate_context_window_usage(combined_json_data)
            st.session_state.window_gpt_tokens = window_gpt
            st.session_state.window_gemini_tokens = window_gemini

            # Check token limits for context window
            if window_gpt > GPT_LIMIT or window_gemini > GEMINI_LIMIT:
                st.session_state.chat_history.append(("assistant", "⚠️ Your conversation has exceeded token limits. Please clear the chat to continue."))
                st.session_state.processing = False
                st.rerun()
            else:
                # --- Call LangGraph ---
                inputs = {
                    "query": last_user_message,
                    "user_data": user_data,
                    "allocations": allocations,
                    #"data":"",
                    "chat_history": st.session_state.chat_history
                }
                print(st.session_state.chat_history)

                
                
                output = app.invoke(inputs, config = config)
                response = output.get('output')
                print(response)
                

                # Check if the response contains allocation updates
                if "allocations" in output:
                    st.session_state.allocations = output["allocations"]

                # Count tokens for the response
                response_gpt_tokens = count_tokens_gpt(response)
                response_gemini_tokens = count_tokens_gemini(response)
                
                # Add to accumulated totals
                st.session_state.total_gpt_tokens += response_gpt_tokens
                st.session_state.total_gemini_tokens += response_gemini_tokens

                # Add to chat history
                st.session_state.chat_history.append(("assistant", response))
                
                # Update context window calculations after adding response
                window_gpt, window_gemini = calculate_context_window_usage(combined_json_data)
                st.session_state.window_gpt_tokens = window_gpt
                st.session_state.window_gemini_tokens = window_gemini
                
        except Exception as e:
            tb = traceback.extract_stack()
            filename, line_number, function_name, text = tb[-2]
            error_message = f"❌ Error: {str(e)} in {filename} at line {line_number}, function: {function_name}"
            st.session_state.chat_history.append(("assistant", error_message))
        
        # Reset processing flag
        st.session_state.processing = False
        st.rerun()