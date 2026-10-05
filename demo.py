import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent


# ============================================================
# DATABASE CONNECTION
# ============================================================

# PostgreSQL connection
#
# Format:
# postgresql+psycopg://USERNAME:PASSWORD@HOST:PORT/DATABASE
#
# Example:
# postgresql+psycopg://postgres:password@localhost:5432/my_tasks

db = SQLDatabase.from_uri(
    f"postgresql+psycopg://"
    f"{os.getenv('POSTGRES_USER')}:"
    f"{os.getenv('POSTGRES_PASSWORD')}@"
    f"{os.getenv('POSTGRES_HOST')}:"
    f"{os.getenv('POSTGRES_PORT')}/"
    f"{os.getenv('POSTGRES_DB')}"
)

print("PostgreSQL DB Connected Successfully ✅")


# ============================================================
# CREATE TABLE
# ============================================================

db.run("""
CREATE TABLE IF NOT EXISTS tasks (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT CHECK (
        status IN ('pending', 'in_progress', 'completed')
    ) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

print("DB Table Created Successfully ✅")


# ============================================================
# LLM
# ============================================================

llm_model = ChatGroq(
    model="openai/gpt-oss-20b"
)


# ============================================================
# SQL TOOLKIT
# ============================================================

toolkit = SQLDatabaseToolkit(
    db=db,
    llm=llm_model
)

tools = toolkit.get_tools()

# Uncomment to see available tools
# for tool in tools:
#     print(tool.name)


# ============================================================
# MEMORY
# ============================================================

memory = InMemorySaver()


# ============================================================
# SYSTEM PROMPT
# ============================================================

system_prompt = """
You are a task management assistant that interacts with a PostgreSQL
database containing a 'tasks' table.

TASK RULES:

1. Limit SELECT queries to a maximum of 10 results.

2. For task-list queries, always use:
   ORDER BY created_at DESC
   LIMIT 10

3. After every CREATE, UPDATE, or DELETE operation,
   execute a SELECT query to confirm that the operation
   was successful.

4. If the user requests a list of tasks, present the result
   in a structured Markdown table.

5. Never modify the database schema unless explicitly requested.

6. Before executing SQL, make sure the SQL is valid PostgreSQL syntax.

7. Use the following table:

   tasks(
       id,
       title,
       description,
       status,
       created_at
   )

CRUD OPERATIONS:

CREATE:
INSERT INTO tasks(title, description, status)

READ:
SELECT *
FROM tasks
WHERE ...
ORDER BY created_at DESC
LIMIT 10

UPDATE:
UPDATE tasks
SET status = ...
WHERE id = ...
OR title = ...

DELETE:
DELETE FROM tasks
WHERE id = ...
OR title = ...

VALID STATUS VALUES:

- pending
- in_progress
- completed
"""


# ============================================================
# AGENT
# ============================================================

agent = create_agent(
    model=llm_model,
    tools=tools,
    checkpointer=memory,
    system_prompt=system_prompt
)


# ============================================================
# CHAT LOOP
# ============================================================

while True:

    query = input("User: ")

    if query.lower() in ["quit", "exit", "bye"]:
        print("GoodBye 👋")
        break

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ]
        },
        {
            "configurable": {
                "thread_id": "1"
            }
        }
    )

    result = response["messages"][-1].content

    print("AI:", result)




# POSTGRES_USER=postgres
# POSTGRES_PASSWORD=2003
# POSTGRES_HOST=localhost
# POSTGRES_PORT=5432
# POSTGRES_DB=synthea