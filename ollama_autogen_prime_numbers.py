import tempfile
import autogen
from autogen_ext.models.ollama import OllamaChatCompletionClient

from dotenv import load_dotenv
import os

# --- Load environment variables ---
load_dotenv()

# --- Configuration ---
# ollama_api_key = os.getenv("OLLAMA_API_KEY")
# if ollama_api_key:
#     print("Note: OLLAMA_API_KEY found. Ensure your Ollama server expects an API key if not local.")

# Ensure the 'coding' directory exists for agent output
if not os.path.exists("coding"):
    os.makedirs("coding")
    print("Created 'coding' directory.")

# --- Ollama Client Configuration (for agents) ---
# Each agent will use this specific Ollama client for its LLM interactions.
# If you want different models for different agents, create separate OllamaChatCompletionClient instances.
ollama_client_for_agents = OllamaChatCompletionClient(
    model="llama2:13b", # Or "llama3:latest", "mistral", etc. - ensure this model is pulled in Ollama
    temperature=0.7,
    seed=42,
    max_tokens=2048,
    # response_format=SomePydanticModel if you want structured output
)

# --- LLM Config for GroupChatManager (for speaker selection) ---
# The GroupChatManager needs an llm_config to select the next speaker.
# We'll use a simplified config list for it, pointing to the local Ollama server.
# This assumes the default Ollama API endpoint.
manager_llm_config = {
    "config_list": [
        {
            "model": "llama2:13b", # Manager also uses llama2:13b for consistency
            "api_type": "ollama",
            "base_url": "http://localhost:11434/api" # Default Ollama API endpoint
        }
    ],
    "temperature": 0.7,
    "timeout": 600 # Increased timeout for local models if needed
}

# --- Start AutoGen Runtime Logging to a .log file ---
# Create a temporary directory for logs if you want them isolated, otherwise use current dir.
# current_log_dir = tempfile.TemporaryDirectory() # Use this if you want logs in a temp dir
logging_session_id = autogen.runtime_logging.start(
    logger_type="file", config={"filename": "conversation_log.log"}) # Logs in current directory
print(f"AutoGen logging started. Session ID: {logging_session_id}")
print("Logs will be saved to 'conversation_log.log'.")


# --- Agent Definitions ---
user_proxy = autogen.UserProxyAgent(
    name="Admin",
    system_message="A human administrator who will review the code and provide final approval for execution. You will also execute tests and report results.",
    # Admin typically doesn't need an LLM for its primary role as human proxy,
    # but can have one if it needs to generate messages itself.
    # If it needs one, use the ollama_client_for_agents directly.
    # model_client=ollama_client_for_agents, # Uncomment if Admin needs to generate LLM responses
    human_input_mode="ALWAYS",
    code_execution_config={
        "work_dir": "coding", # Agents will save/execute code here
        "use_docker": False,
    },
    is_termination_msg=lambda x: x.get(
        "content", "").rstrip().endswith("TERMINATE"),
)

coder = autogen.AssistantAgent( # Renamed 'assistant' to 'coder' for clarity matching system message
    name="Coder",
    llm_config=manager_llm_config,
    system_message="""You are a helpful AI assistant specialized in Python programming.
You can write Python code, explain concepts, and debug issues.
When you provide code, ensure it is complete, runnable, and follows good practices including robust error handling, clear documentation, modularity, and reusability.
Also save your code into a 'coding' sub-folder within the current project directory with appropriate version numbering.
If you need to run code or tests, suggest it and wait for approval.
If the requirements are unclear or ambiguous, ask clarifying questions to ensure a precise understanding.
When providing solutions for complex problems, break them down into smaller, manageable sub-tasks and explain your approach.
Once the task is complete, reply with 'TERMINATE' to end the conversation.""",
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    }
)

reviewer = autogen.AssistantAgent(
    name="Reviewer",
    llm_config=manager_llm_config,
    system_message="""You are a meticulous Code Reviewer.
    Your sole role is to provide feedback, suggestions, and critique on **any Python code presented in the conversation, including application code and test code.**
    **NEVER write or rewrite any code yourself.**
    **NEVER suggest specific code implementations.** Instead, describe *what* needs to be changed or improved conceptually.
    Focus on the following aspects of the code:
    - **Correctness:** Does it solve the problem accurately and without bugs (for application code) or does it accurately test the application code (for test code)? **Does it have any potential security vulnerabilities?**
    - **Efficiency:** Can it be optimized for speed or resource usage?
    - **Readability & Style (PEP 8):** Does it follow Python's PEP 8 style guide for formatting, naming conventions, and overall code structure?
    - **Documentation (PEP 257 for Docstrings):**
    - **Docstrings:** Are all public modules, classes, functions, and methods properly documented with clear, concise docstrings following PEP 257 conventions?
    - **Comments:** Are inline comments used judiciously to explain complex logic or non-obvious parts of the code where necessary?
    - **Edge Cases:** Does the code handle potential edge cases and error conditions gracefully (for application code) or does it include tests for these (for test code)?
    - **Test Coverage (for test code):** Does the test code adequately cover the functionality of the application code? Are there enough test cases?
    Provide constructive feedback and suggest improvements for all the above points.
    If the code is flawless and needs no changes, respond with: 'Looks good! Code review complete.'
    If changes are needed, clearly explain the areas for improvement.
    Once the code is approved or deemed perfect, you are done.
    """
)

test_engineer = autogen.AssistantAgent(
    name="Test_Engineer",
    llm_config=manager_llm_config,
    system_message="""You are a skilled Test Engineer specialized in Python.
Your primary role is to create comprehensive unit tests for the Python code provided by the Coder.
Your tests should:
- Be written using Python's `unittest` framework unless `pytest` is explicitly requested or already in use within the project.
- Cover various scenarios, including normal cases, edge cases, and invalid inputs.
- Assert the correctness of the Coder's functions.
- Be self-contained, runnable, and independent of other tests (each test should be able to run in isolation).
- Have clear and descriptive names that indicate the specific scenario being tested.
- **NEVER modify the application code directly.** Your role is solely to create tests for it.
Also save your code into a 'coding' sub-folder within the current project directory with appropriate version numbering (e.g., `test_module_v1.py`).
Once the test code is complete, **ensure it is runnable and passes locally (if possible) before presenting it to the Reviewer** for their feedback.
After the test code has been reviewed and approved, then you can request the Admin to execute the tests.
If tests look good and no more test cases are needed, signal approval for the main script to proceed.
    """,
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    }
)

# --- Group Chat Setup ---
groupchat = autogen.GroupChat(
    agents=[user_proxy, coder, reviewer, test_engineer], # Used 'coder' instead of 'assistant'
    messages=[],
    max_round=30,  # Increased max_round again to accommodate more review iterations
    speaker_selection_method="auto",
)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=manager_llm_config) # Manager uses its own llm_config

# --- Initiate the chat ---
print("\n--- Starting the AutoGen Conversation ---")
print("Admin will initiate the conversation with the GroupChatManager.")
print("Type 'exit' to terminate the human input at any point.")

user_proxy.initiate_chat(
    manager,
    message="""
    Write a Python class for performing basic arithmetic operations and also implement main program to test these arithmetic operations.
    The script should print these numbers to the console.
    Ensure the code is reviewed for correctness, efficiency, and proper documentation (docstrings and comments).
    **After the application code is reviewed, a Test Engineer should generate unit tests for it.
    The Test Engineer's code should then also be reviewed by the Code Reviewer for quality before I execute those tests to ensure correctness.**
    Once tests pass and the application code is finalized, I will the provide final approval to run the main script.
    """
)

print("\n--- Conversation Ended ---")
print("Check the 'coding' directory for any generated files.")

# --- Stop AutoGen Runtime Logging ---
autogen.runtime_logging.stop()
print("AutoGen logging stopped. You can now review 'conversation_log.log'.")

# Clean up the temporary directory if it was used for logs
# if 'current_log_dir' in locals():
#     current_log_dir.cleanup()
#     print(f"Cleaned up temporary log directory: {current_log_dir.name}")
