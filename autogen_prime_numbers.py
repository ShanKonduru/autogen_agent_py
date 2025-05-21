import autogen
from dotenv import load_dotenv
import os

# --- Load environment variables ---
load_dotenv()

# --- Configuration ---
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file or as an environment variable.")

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

# --- Start AutoGen Runtime Logging to a .log file ---
logging_session_id = autogen.runtime_logging.start(logger_type="file", config={"filename": "conversation_log.log"})
print(f"AutoGen logging started. Session ID: {logging_session_id}")
print("Logs will be saved to 'conversation_log.log'.")

# --- Agent Definitions ---
user_proxy = autogen.UserProxyAgent(
    name="Admin",
    system_message="A human administrator who will review the code and provide final approval for execution. You will also execute tests and report results.",
    llm_config={"config_list": config_list},
    human_input_mode="ALWAYS",
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    },
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
)

assistant = autogen.AssistantAgent(
    name="Coder",
    llm_config={"config_list": config_list},
    system_message="""You are a helpful AI assistant specialized in Python programming.
    You can write Python code, explain concepts, and debug issues.
    When you provide code, ensure it is complete, runnable, and follows good practices including documentation.
    If you need to run code or tests, suggest it and wait for approval.
    Once the task is complete, reply with 'TERMINATE' to end the conversation."""
)

# --- UPDATED Reviewer Agent ---
reviewer = autogen.AssistantAgent(
    name="Reviewer",
    llm_config={"config_list": config_list},
    system_message="""You are a meticulous Code Reviewer.
    Your sole role is to provide feedback, suggestions, and critique on **any Python code presented in the conversation, including application code and test code.**
    **NEVER write or rewrite any code yourself.**
    **NEVER suggest specific code implementations.** Instead, describe *what* needs to be changed or improved conceptually.

    Focus on the following aspects of the code:
    - **Correctness:** Does it solve the problem accurately and without bugs (for application code) or does it accurately test the application code (for test code)?
    - **Efficiency:** Can it be optimized for speed or resource usage?
    - **Readability & Style (PEP 8):** Does it follow Python's PEP 8 style guide for formatting, naming conventions, and overall code structure?
    - **Documentation (PEP 257 for Docstrings):**
        - **Docstrings:** Are all public modules, classes, functions, and methods properly documented with clear, concise docstrings following PEP 257 conventions?
        - **Comments:** Are inline comments used judiciously to explain complex logic or non-obvious parts of the code where necessary?
    - **Edge Cases:** Does the code handle potential edge cases and error conditions gracefully (for application code) or does it include tests for these (for test code)?
    - **Test Coverage (for test code):** Does the test code adequately cover the functionality of the application code? Are there enough test cases?

    Provide constructive feedback and suggest improvements for all the above points.
    If the code is flawless and needs no changes, respond with: 'Looks good!'
    If changes are needed, clearly explain the areas for improvement.
    Once the code is approved or deemed perfect, you are done.
    """
)

# --- UPDATED Test Engineer Agent ---
test_engineer = autogen.AssistantAgent(
    name="Test_Engineer",
    llm_config={"config_list": config_list},
    system_message="""You are a skilled Test Engineer specialized in Python.
    Your primary role is to create comprehensive unit tests for the Python code provided by the Coder.
    Your tests should:
    - Be written using Python's `unittest` or `pytest` framework (prefer `unittest` for simplicity if not specified).
    - Cover various scenarios, including normal cases, edge cases, and invalid inputs.
    - Assert the correctness of the Coder's functions.
    - Be self-contained and runnable.
    Once the test code is complete, **present it to the Reviewer for their feedback before requesting execution by the Admin.**
    After the test code has been reviewed and approved, then you can request the Admin to execute the tests.
    If tests look good and no more test cases are needed, signal approval for the main script to proceed.
    """
)

# --- Group Chat Setup ---
groupchat = autogen.GroupChat(
    agents=[user_proxy, assistant, reviewer, test_engineer],
    messages=[],
    max_round=30, # Increased max_round again to accommodate more review iterations
    speaker_selection_method="auto",
)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list})

# --- Initiate the chat ---
print("\n--- Starting the AutoGen Conversation ---")
print("Admin will initiate the conversation with the GroupChatManager.")
print("Type 'exit' to terminate the human input at any point.")

user_proxy.initiate_chat(
    manager,
    message="""
    Write a Python script that finds the first 10 prime numbers.
    The script should print these numbers to the console.
    Ensure the code is reviewed for correctness, efficiency, and proper documentation (docstrings and comments).
    **After the application code is reviewed, a Test Engineer should generate unit tests for it. The Test Engineer's code should then also be reviewed by the Code Reviewer for quality before I execute those tests to ensure correctness.**
    Once tests pass and the application code is finalized, I will provide final approval to run the main script.
    """
)

print("\n--- Conversation Ended ---")
print("Check the 'coding' directory for any generated files.")

# --- Stop AutoGen Runtime Logging ---
autogen.runtime_logging.stop()
print("AutoGen logging stopped. You can now review 'conversation_log.log'.")