import autogen
import os
from dotenv import load_dotenv
import sys

print("--- Starting test_autogen_execution.py ---", flush=True)

# --- API Key and Environment Check ---
print("Attempting to load .env file...", flush=True)
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

api_key = "ollama"

if not openai_api_key:
    print("\nERROR: OPENAI_API_KEY not found.", flush=True)
    print("Please ensure you have a '.env' file in the same directory as this script.", flush=True)
    print("Inside '.env', it should contain: OPENAI_API_KEY='your_openai_api_key_here'", flush=True)
    print("Exiting script.", flush=True)
    sys.exit(1)
else:
    print("OPENAI_API_KEY successfully loaded (or found in environment variables).", flush=True)
    print(f"API Key start: {api_key[:5]}...", flush=True)

# --- AutoGen Configuration ---
openai_config_list = [
    {
        "model": "gpt-4", # Or "gpt-3.5-turbo" if you prefer or have access to only that model
        "api_key": openai_api_key,
    }
]

# --- Ollama AutoGen Configuration ---
ollama_config_list = [
    {
        "model": "llama2:13b",
        "api_key": api_key,
    },
    {
        "model": "llama3.1:latest",
        "api_key": api_key,
    },
    {
        "model": "llama3.2:latest",
        "api_key": api_key,
    }
]


# --- Custom Message Logger for Console ---
def console_message_logger(recipient, messages, sender, config):
    """
    Callback function to log messages from AutoGen agents to the console,
    with flushing to ensure immediate display.
    """
    message = messages[-1] if messages else {}
    content = message.get("content", "")

    if isinstance(content, dict):
        if 'tool_code' in content:
            content_display = f"TOOL_CODE (Execution Request):\n{content.get('tool_code', '')}"
        elif 'result' in content:
            content_display = f"TOOL_RESULT (Execution Output):\n{content.get('result', '')}"
        else:
            content_display = str(content)
    elif not isinstance(content, str):
        content_display = str(content)
    else:
        content_display = content

    print(f"\n--- MESSAGE ---", flush=True)
    print(f"SENDER: {sender.name}", flush=True)
    print(f"RECIPIENT: {recipient.name}", flush=True)
    print(f"CONTENT:\n{content_display}", flush=True)
    print(f"---------------\n", flush=True)
    return False, None

# --- Agent Definitions ---
def register_logger_to_agent(agent):
    agent.register_reply(
        [autogen.Agent, None],
        reply_func=console_message_logger,
        config={"callback": None},
    )

coder = autogen.AssistantAgent(
    name="Coder",
    llm_config={"config_list": ollama_config_list},
    # MODIFIED: Instruct Coder to wait for execution result before terminating
    system_message="""You are a helpful assistant that writes Python code. When asked, provide runnable Python code inside a markdown block.
    After providing the code, **wait for the execution result from the Executor**.
    If the execution is successful and the task is completed, then and only then, end your final message with 'TASK_COMPLETED'.""",
)
register_logger_to_agent(coder)

user_proxy = autogen.UserProxyAgent(
    name="Executor",
    human_input_mode="NEVER",
    code_execution_config={
        "work_dir": "coding_test",
        "use_docker": False,
    },
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TASK_COMPLETED"),
    max_consecutive_auto_reply=20, # Increased turns to allow for more robust interaction
)
register_logger_to_agent(user_proxy)


# --- Directory Setup ---
work_dir = "coding_test"
if not os.path.exists(work_dir):
    try:
        os.makedirs(work_dir)
        print(f"Created working directory: {os.path.abspath(work_dir)}", flush=True)
    except OSError as e:
        print(f"\nERROR: Could not create working directory '{work_dir}': {e}", flush=True)
        print("Please check file permissions or create the directory manually.", flush=True)
        print("Exiting script.", flush=True)
        sys.exit(1)
else:
    print(f"Working directory already exists: {os.path.abspath(work_dir)}", flush=True)


# --- Initiate Chat ---
print("\n--- Initiating AutoGen Test Chat (EXPECT MESSAGES BELOW THIS LINE) ---", flush=True)
print("----------------------------------------------------------------------", flush=True)
try:
    user_proxy.initiate_chat(
        coder,
        # MODIFIED: Emphasize the need for execution and confirmation before termination
        message="""Write a Python script to print first 10 prime numbers, then print the output content to the console.
        Provide the Python code block. After the Executor runs the code and confirms the output, then you can state 'TASK_COMPLETED'.""",
    )
    print("\n----------------------------------------------------------------------", flush=True)
    print("AutoGen chat initiated successfully.", flush=True)
except Exception as e:
    print(f"\nERROR: An error occurred during AutoGen chat initiation: {e}", flush=True)
    print("Exiting script.", flush=True)
    sys.exit(1)

print("\n--- Test Chat Finished ---", flush=True)

# --- Verify File Creation ---
file_path = os.path.join(work_dir, "hello.txt")
if os.path.exists(file_path):
    print(f"\nSUCCESS: '{file_path}' found!", flush=True)
    try:
        with open(file_path, 'r') as f:
            print(f"Content of '{file_path}':", flush=True)
            print(f.read(), flush=True)
    except Exception as e:
        print(f"Could not read file '{file_path}': {e}", flush=True)
else:
    print(f"\nFAILURE: '{file_path}' NOT found.", flush=True)
    print("This means the code was likely not executed or the file was not created by the executed code.", flush=True)

print("--- Script Finished ---", flush=True)