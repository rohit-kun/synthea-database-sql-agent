import os

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_agent


# ============================================================
# Load environment variables
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")


# ============================================================
# PostgreSQL connection URL
# ============================================================

DATABASE_URI = (
    f"postgresql+psycopg2://"
    f"{DB_USER}:{DB_PASSWORD}@"
    f"{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

print("Connecting to PostgreSQL...")


# ============================================================
# Create SQLDatabase
# ============================================================

db = SQLDatabase.from_uri(DATABASE_URI)

print("Connected!")
print("Tables:", db.get_usable_table_names())


# ============================================================
# Create Groq LLM
# ============================================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=GROQ_API_KEY,
)


# ============================================================
# Create SQLDatabaseToolkit
# ============================================================

toolkit = SQLDatabaseToolkit(
    db=db,
    llm=llm
)

tools = toolkit.get_tools()


# ============================================================
# System prompt
# ============================================================

SYSTEM_PROMPT = """
You are an expert PostgreSQL database assistant.

You have access to a PostgreSQL database through SQL tools.

Your responsibilities:

1. Answer database questions using the SQL tools.
2. Inspect the database schema before writing SQL when necessary.
3. Never invent table names or column names.
4. Always use PostgreSQL-compatible SQL.
5. For SELECT queries, execute the query and explain the result.
6. INSERT, UPDATE and DELETE are allowed when explicitly requested.
7. Never execute UPDATE without a WHERE clause.
8. Never execute DELETE without a WHERE clause.
9. Never execute DROP DATABASE.
10. Never execute DROP TABLE.
11. Never execute TRUNCATE unless explicitly requested.
12. Never modify the database schema unless explicitly requested.
13. If a request is ambiguous, ask the user for clarification.
14. After an INSERT, UPDATE or DELETE, clearly explain what was changed.
"""


# ============================================================
# Create Agent
# ============================================================

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
)


# ============================================================
# Chat loop
# ============================================================

print("\nPostgreSQL Agent is ready.")
print("Type 'exit' to quit.\n")


while True:

    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    try:

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_input
                    }
                ]
            }
        )

        # Find the final AI response
        for message in reversed(result["messages"]):

            if (
                getattr(message, "type", None) == "ai"
                and message.content
            ):
                print("\nAgent:", message.content)
                break

    except Exception as e:

        print("\nError:", e)
