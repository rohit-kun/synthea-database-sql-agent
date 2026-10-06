import uuid
import streamlit as st

from agent import ask_agent


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Task Manager AI",
    page_icon="✅",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f8fafc;
    }

    .block-container {
        max-width: 1000px;
        padding-top: 2rem;
    }

    .app-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.2rem;
    }

    .app-subtitle {
        color: #6b7280;
        margin-bottom: 2rem;
    }

    .stChatMessage {
        border-radius: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("⚙️ Task Manager")

    st.markdown(
        """
        ### What can I do?

        - ➕ Create tasks
        - 📋 List tasks
        - 🔎 Find tasks
        - ✏️ Update tasks
        - 🗑️ Delete tasks
        - 📊 Check task information

        Your requests are processed using your existing
        PostgreSQL database.
        """
    )

    st.divider()

    st.subheader("Example requests")

    examples = [
        "Show my latest tasks",
        "Create a task to finish the project report",
        "Mark task 123 as completed",
        "Delete task 456",
        "Find tasks related to the project",
    ]

    for example in examples:
        st.caption(f"• {example}")

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())

        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="app-title">✅ Task Manager AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="app-subtitle">'
    'Manage your PostgreSQL tasks using natural language.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# WELCOME MESSAGE
# ============================================================

if not st.session_state.messages:

    with st.chat_message("assistant"):

        st.markdown(
            """
            👋 **Hello!**

            I can help you manage your tasks.

            Try something like:

            > Show my latest tasks

            or

            > Create a task to prepare the monthly report
            """
        )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

query = st.chat_input(
    "Ask me to create, find, update, or delete a task..."
)


if query:

    # --------------------------------------------------------
    # DISPLAY USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):
        st.markdown(query)


    # --------------------------------------------------------
    # CALL AGENT
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Working with your database..."):

            try:

                response = ask_agent(
                    query,
                    thread_id=st.session_state.thread_id
                )

                st.markdown(response)

            except Exception as e:

                response = (
                    "Sorry, I couldn't complete that request. "
                    "Please check the database connection and try again."
                )

                st.error(response)

                # Useful for development.
                # Remove this in production if you don't want
                # technical errors displayed.
                st.exception(e)


    # --------------------------------------------------------
    # SAVE ASSISTANT RESPONSE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )
