import autogen
from dotenv import load_dotenv

import os

# --- Load environment variables ---
load_dotenv()

# --- Configuration ---
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file or as an environment variable.")

# Define the config_list directly using the loaded API key
config_list = [
    {
        "model": "gpt-4", 
        "api_key": openai_api_key,
    },
    {
        "model": "gpt-3.5-turbo",
        "api_key": openai_api_key
    }
]


# --- Agent Definitions ---

# 1. User Proxy Agent
# This agent represents the human user.
# It can execute code and provide feedback.
# 'human_input_mode="ALWAYS"' means it will always ask for human approval before executing code.
# 'code_execution_config={"work_dir": "coding"}' specifies a directory for code execution.
user_proxy = autogen.UserProxyAgent(
    name="Admin",
    system_message="A human administrator who will review the code and provide final approval.",
    llm_config={"config_list": config_list},
    human_input_mode="ALWAYS", # Can be "ALWAYS", "NEVER", or "TERMINATE"
    code_execution_config={
        "work_dir": "coding", # Directory where code files will be created and executed
        "use_docker": False,  # Set to True if you want to use Docker for isolated execution
    },
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    # A termination message to signal the end of the conversation.
    # The AssistantAgent will learn to use this when the task is done.
)

# 2. Assistant Agent
# This agent is the AI assistant that will try to solve the problem.
# It's configured to use the LLM (Large Language Model) via config_list.
assistant = autogen.AssistantAgent(
    name="Coder",
    llm_config={"config_list": config_list},
    system_message="""You are a helpful AI assistant specialized in Python programming.
    You can write Python code, explain concepts, and debug issues.
    When you provide code, ensure it is complete and runnable.
    If you need to run code, suggest it and wait for approval.
    Once the task is complete, reply with 'TERMINATE' to end the conversation."""
)

# --- Start the Conversation ---
print("\n--- Starting the AutoGen Conversation ---")
print("User Proxy will initiate the conversation with the Coder agent.")
print("Type 'exit' to terminate the human input at any point.")

user_proxy.initiate_chat(
    assistant,
    message="""
    Write a Python script that finds the first 10 prime numbers.
    The script should print these numbers to the console.
    Please ensure the code is correct and runnable.
    """
)

print("\n--- Conversation Ended ---")
print("Check the 'coding' directory for any generated files.")