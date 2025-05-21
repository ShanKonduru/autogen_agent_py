import streamlit as st
import autogen
import os
from dotenv import load_dotenv # Used for local development to load .env file

# --- Streamlit Session State Initialization ---
# Initialize chat_history to store messages from agents
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
# Initialize is_chatting to manage the state of the conversation (e.g., disable button)
if "is_chatting" not in st.session_state:
    st.session_state.is_chatting = False

# --- AutoGen Configuration ---
# Load environment variables for local testing.
# For deployment on Streamlit Cloud, use st.secrets["OPENAI_API_KEY"] instead.
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is available
if not openai_api_key:
    st.error("OPENAI_API_KEY not found. Please set it in your Streamlit secrets (for deployment) or in a .env file (for local development).")
    st.stop() # Stop the Streamlit app if the key is missing

# Configuration list for AutoGen agents, specifying models and API key
config_list = [
    {
        "model": "gpt-4",  # You can use "gpt-3.5-turbo" for faster, cheaper interactions
        "api_key": openai_api_key,
    },
    {
        "model": "gpt-3.5-turbo",
        "api_key": openai_api_key
    }
]

# --- Custom Message Logger for Streamlit ---
# This function captures messages exchanged between agents and appends them to Streamlit's session state.
def custom_message_logger(sender, recipient, message, context):
    """
    Callback function to log messages from AutoGen agents to Streamlit's session state.
    This allows the Streamlit UI to display the ongoing conversation.
    """
    # Extract message content. AutoGen messages can have various structures.
    content = message.get("content", "")
    if isinstance(content, dict) and "message" in content:
        content = content["message"] # Handle cases where content might be nested
    elif not isinstance(content, str):
        content = str(content) # Ensure content is a string for display

    # Append the message to the chat history in session state
    st.session_state.chat_history.append({
        "sender": sender.name,
        "recipient": recipient.name, # Useful for debugging who talks to whom
        "message": content
    })
    # Optional: Print to console for real-time debugging in the terminal
    # print(f"Logged: {sender.name} -> {recipient.name}: {content[:100]}...") # Truncate for cleaner console output

# --- Agent Definitions ---
# Helper function to register the custom logger to an agent
def register_logger_to_agent(agent):
    # Register the custom_message_logger to be called whenever this agent sends a reply
    agent.register_reply([autogen.Agent, None], custom_message_logger)

# 1. User Proxy Agent (Admin)
# This agent acts as the human administrator.
# human_input_mode="NEVER" is crucial for Streamlit to prevent blocking the UI
# and waiting for console input during the conversation.
user_proxy = autogen.UserProxyAgent(
    name="Admin",
    system_message="A human administrator who initiates tasks and reviews final outcomes. You will execute tests and report results as requested by other agents. Do not ask for human input during the conversation.",
    llm_config={"config_list": config_list},
    human_input_mode="NEVER",
    code_execution_config={
        "work_dir": "coding", # Directory where code will be executed and saved
        "use_docker": False,  # Set to True if you want to use Docker for isolated execution
    },
    # Defines when the conversation should terminate
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
)
register_logger_to_agent(user_proxy) # Register the logger for the Admin agent

# 2. Assistant Agent (Coder)
# This agent is responsible for writing Python code.
assistant = autogen.AssistantAgent(
    name="Coder",
    llm_config={"config_list": config_list},
    system_message="""You are a helpful AI assistant specialized in Python programming.
    You can write Python code, explain concepts, and debug issues.
    When you provide code, ensure it is complete, runnable, and follows good practices including documentation.
    If you need to run code or tests, suggest it and wait for approval.
    Once the task is complete, reply with 'TERMINATE' to end the conversation."""
)
register_logger_to_agent(assistant) # Register the logger for the Coder agent

# 3. Reviewer Agent
# This agent provides feedback and critiques on code.
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
register_logger_to_agent(reviewer) # Register the logger for the Reviewer agent

# 4. Test Engineer Agent
# This agent is responsible for creating unit tests.
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
register_logger_to_agent(test_engineer) # Register the logger for the Test Engineer agent

# --- Group Chat Setup ---
# Define the group chat with all agents
groupchat = autogen.GroupChat(
    agents=[user_proxy, assistant, reviewer, test_engineer],
    messages=[],
    max_round=30, # Increased max_round to allow for more complex conversations and iterations
    speaker_selection_method="auto", # Auto-selects the next speaker
)

# The manager orchestrates the group chat
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list})


# --- Function to run the AutoGen conversation ---
# This function encapsulates the AutoGen chat initiation logic.
def run_autogen_conversation(prompt):
    """
    Initiates the AutoGen group chat with the given prompt.
    Clears previous chat history and sets the chatting state.
    """
    st.session_state.chat_history = [] # Clear previous conversation history
    st.session_state.is_chatting = True # Set flag to indicate conversation is active

    # Optional: Start AutoGen Runtime Logging to a .log file.
    # This is separate from the Streamlit UI log but can be useful for detailed debugging.
    # logging_session_id = autogen.runtime_logging.start(logger_type="file", config={"filename": "conversation_log.log"})
    # print(f"AutoGen logging started. Session ID: {logging_session_id}")
    # print("Logs will be saved to 'conversation_log.log'.")

    try:
        # Initiate the chat with the manager and the user's prompt
        user_proxy.initiate_chat(
            manager,
            message=prompt
        )
    except Exception as e:
        # Display any errors that occur during the conversation
        st.error(f"An error occurred during the AutoGen conversation: {e}")
    finally:
        st.session_state.is_chatting = False # Reset chatting flag
        # Optional: Stop AutoGen logging when the conversation concludes
        # autogen.runtime_logging.stop()


# --- Streamlit UI Layout ---
st.set_page_config(layout="wide", page_title="AutoGen AI Collaboration")

st.title("AutoGen AI Agent Collaboration for Python Development")
st.markdown("""
This application demonstrates a multi-agent AI system powered by AutoGen, where different AI agents
collaborate to write, review, and test Python code.

Enter your development request below, and watch the agents work together!
""")

# Text area for the user to input their development request
user_question = st.text_area(
    "Your Development Request:",
    # Pre-fill with the example prompt for convenience
    value="""Write a Python script that finds the first 10 prime numbers.
The script should print these numbers to the console.
Ensure the code is reviewed for correctness, efficiency, and proper documentation (docstrings and comments).
**After the application code is reviewed, a Test Engineer should generate unit tests for it. The Test Engineer's code should then also be reviewed by the Code Reviewer for quality before I execute those tests to ensure correctness.**
Once tests pass and the application code is finalized, I will provide final approval to run the main script.
""",
    height=200, # Set height for better user experience
    key="user_request_input" # Unique key for the widget
)

# Button to start the conversation. It's disabled while a conversation is active.
if st.button("Start AI Conversation", disabled=st.session_state.is_chatting):
    if user_question:
        # Show a spinner while agents are working, as it can take time
        with st.spinner("AI Agents are collaborating... This may take a few minutes, please be patient."):
            run_autogen_conversation(user_question)
    else:
        st.warning("Please enter a development request to start the conversation.")

st.markdown("---") # Separator for visual clarity
st.subheader("Conversation Log:")

# Display the conversation history
if st.session_state.chat_history:
    for i, msg in enumerate(st.session_state.chat_history):
        sender = msg["sender"]
        content = msg["message"]

        # Use different Streamlit components or Markdown for different agents
        # to visually distinguish their messages.
        if sender == "Admin":
            st.info(f"**{sender}:**\n\n{content}")
        elif sender == "Coder":
            st.success(f"**{sender}:**\n\n{content}")
        elif sender == "Reviewer":
            st.warning(f"**{sender}:**\n\n{content}")
        elif sender == "Test_Engineer":
            # Using st.error for Test Engineer for a distinct color, can be adjusted
            st.error(f"**{sender}:**\n\n{content}")
        else:
            st.write(f"**{sender}:**\n\n{content}")

        # Add a horizontal rule between messages for better readability
        if i < len(st.session_state.chat_history) - 1:
            st.markdown("---")
else:
    st.info("No conversation started yet. Enter a request and click 'Start AI Conversation'.")

st.markdown("---")
st.markdown("For more details, check the `coding` directory for any generated files (e.g., Python scripts, test files).")

