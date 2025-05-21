import autogen
import os
from dotenv import load_dotenv
import sys

print("--- Starting test_autogen_execution.py ---")

# --- API Key and Environment Check ---
print("Attempting to load .env file...")
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    print("\nERROR: OPENAI_API_KEY not found.")
    print("Please ensure you have a '.env' file in the same directory as this script.")
    print("Inside '.env', it should contain: OPENAI_API_KEY='your_openai_api_key_here'")
    print("Exiting script.")
    sys.exit(1)
else:
    print("OPENAI_API_KEY successfully loaded (or found in environment variables).")
    print(f"API Key start: {openai_api_key[:5]}...")

# --- AutoGen Configuration ---
config_list = [
    {
        "model": "gpt-4", # Or "gpt-3.5-turbo"
        "api_key": openai_api_key,
    }
]

# --- Custom Message Logger for Console ---
# This function will print messages from agents to the console
def console_message_logger(recipient, messages, sender, config):
    """
    Callback function to log messages from AutoGen agents to the console.
    """
    message = messages[-1] if messages else {} # Get the most recent message
    content = message.get("content", "")

    # Clean up content for display, especially if it's a dictionary or complex object
    if isinstance(content, dict):
        # If it's a tool call, show relevant parts
        if 'tool_code' in content:
            content_display = f"TOOL_CODE (Execution Request):\n{content.get('tool_code', '')}"
        elif 'result' in content:
            content_display = f"TOOL_RESULT (Execution Output):\n{content.get('result', '')}"
        else:
            content_display = str(content) # Fallback for other dicts
    elif not isinstance(content, str):
        content_display = str(content)
    else:
        content_display = content # It's already a string

    print(f"\n--- MESSAGE ---")
    print(f"SENDER: {sender.name}")
    print(f"RECIPIENT: {recipient.name}")
    print(f"CONTENT:\n{content_display}")
    print(f"---------------\n")
    return False, None # Continue normal AutoGen processing

# --- Agent Definitions ---
# Helper function to register the custom logger to an agent
def register_logger_to_agent(agent):
    agent.register_reply(
        [autogen.Agent, None],
        reply_func=console_message_logger,
        config={"callback": None},
    )

coder = autogen.AssistantAgent(
    name="Coder",
    llm_config={"config_list": config_list},
    system_message="You are a helpful assistant that writes Python code. When asked, provide runnable Python code inside a markdown block. Once you have provided the complete and correct code, indicate that you are done by ending your message with 'TERMINATE'.",
)
register_logger_to_agent(coder) # Register logger for Coder

user_proxy = autogen.UserProxyAgent(
    name="Executor",
    human_input_mode="NEVER",
    code_execution_config={
        "work_dir": "coding_test", # Use a unique directory for this test
        "use_docker": False,
    },
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    max_consecutive_auto_reply=10,
)
register_logger_to_agent(user_proxy) # Register logger for Executor


# --- Directory Setup ---
work_dir = "coding_test"
if not os.path.exists(work_dir):
    try:
        os.makedirs(work_dir)
        print(f"Created working directory: {os.path.abspath(work_dir)}")
    except OSError as e:
        print(f"\nERROR: Could not create working directory '{work_dir}': {e}")
        print("Please check file permissions or create the directory manually.")
        print("Exiting script.")
        sys.exit(1)
else:
    print(f"Working directory already exists: {os.path.abspath(work_dir)}")


# --- Initiate Chat ---
print("\n--- Initiating AutoGen Test Chat (This may take a few seconds) ---")
try:
    user_proxy.initiate_chat(
        coder,
        message="""Write a Python script that prints 'Hello from AutoGen file!' to a file named 'hello.txt' in the current working directory, then print the content of that file to the console. Make sure to include the Python code block. End your final message with 'TERMINATE'.""",
    )
    print("\nAutoGen chat initiated successfully.")
except Exception as e:
    print(f"\nERROR: An error occurred during AutoGen chat initiation: {e}")
    print("Exiting script.")
    sys.exit(1)

print("\n--- Test Chat Finished ---")

# --- Verify File Creation ---
file_path = os.path.join(work_dir, "hello.txt")
if os.path.exists(file_path):
    print(f"\nSUCCESS: '{file_path}' found!")
    try:
        with open(file_path, 'r') as f:
            print(f"Content of '{file_path}':")
            print(f.read())
    except Exception as e:
        print(f"Could not read file '{file_path}': {e}")
else:
    print(f"\nFAILURE: '{file_path}' NOT found.")
    print("This means the code was likely not executed or the file was not created by the executed code.")

print("--- Script Finished ---")