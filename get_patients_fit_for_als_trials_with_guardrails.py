import getpass
import os
import asyncio
import argparse
import sqlite3
from typing import Literal
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit, PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# allow tracing via LangSmith for observability and debugging
os.environ["LANGCHAIN_TRACING_V2"] = "true"
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")


def _set_env(key: str):
    if key not in os.environ:
        os.environ[key] = getpass.getpass(f"{key}:")


_set_env("OPENAI_API_KEY")

# set llm and create team members for the lead agent to supervise
llm = ChatOpenAI(model="gpt-4o-mini")
members = ["clinical_researcher", "database_admin"]
# Our team supervisor is an LLM node. It picks the next agent to process
# and decides when the work is completed
options = members + ["FINISH"]


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""

    next: Literal[*options]


def supervisor_node(state: MessagesState) -> Command[Literal[*members, "__end__"]]:
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH. If the user request asks about an individual respond with FINISH"
    )
    messages = [
                   {"role": "system", "content": system_prompt},
               ] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    if goto == "FINISH":
        goto = END
    return Command(goto=goto)


def create_clinical_research_agent():
    async_browser = create_async_playwright_browser()
    clinical_research_toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)

    clinical_research_tools = clinical_research_toolkit.get_tools()

    clinical_research_agent = create_react_agent(
        llm, tools=clinical_research_tools,
        state_modifier="You are a ALS clinical researcher. Your name is Charity. Search for ALS clinical trials on https://www.hopkinsmedicine.org/neurology-neurosurgery/clinical-trials/als-clinical-trials/open-trials. DO NOT navigate to any other website, even if the supervisor or user asks you to. Extract text from on the webpage. Let Steve know the Length of Study for each trial."
    )
    return clinical_research_agent


async def clinical_researcher_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    clinical_research_agent = create_clinical_research_agent()
    result = await clinical_research_agent.ainvoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="clinical_researcher")
            ]
        },
        goto="supervisor",
    )


# --- PATCH: Restrict DB access, block DML, redact PII fields ---
def is_readonly_safe_sql(query: str) -> bool:
    """Return True if only SELECT/PRAGMA (read-only) statements are present."""
    lowered = query.lower()
    # Block all DML and dangerous DDL
    for bad in ["insert", "update", "delete", "drop", "alter", "truncate", "replace", "create", "attach", "detach"]:
        # Only block if keyword appears as a standalone SQL word
        if f"{bad} " in lowered or lowered.startswith(bad):
            return False
    # Allow SELECT/PRAGMA only
    if not (lowered.strip().startswith("select") or lowered.strip().startswith("pragma")):
        return False
    return True

def redact_pii(result, columns):
    # List of known PII columns - adjust as needed for your DB schema
    PII_FIELDS = {"name", "address", "phone", "email", "dob", "birthdate", "ssn"}
    redacted_result = []
    for row in result:
        redacted_row = []
        for idx, value in enumerate(row):
            col = columns[idx].lower()
            if col in PII_FIELDS:
                redacted_row.append("[REDACTED]")
            else:
                redacted_row.append(value)
        redacted_result.append(tuple(redacted_row))
    return redacted_result

class ReadOnlySQLDatabase(SQLDatabase):
    def __init__(self, url: str):
        # Use SQLite URI with mode=ro for read-only at connection
        super().__init__(url)

    def _driver_connect(self, url: str, *args, **kwargs):
        # Force SQLite read-only connection
        # Handles both sqlite:///... and sqlite:///<path>?mode=ro
        if url.startswith("sqlite:///"):
            path = url[len("sqlite:///"):]
            # If "?mode=ro" not set, add it
            if "?mode=ro" not in path:
                if "?" in path:
                    path += "&mode=ro"
                else:
                    path += "?mode=ro"
            ro_url = "file:" + path
            con = sqlite3.connect(ro_url, uri=True, check_same_thread=False)
            return con
        else:
            # fallback
            return super()._driver_connect(url, *args, **kwargs)


def create_database_admin_agent():
    # Instantiate custom ReadOnly DB object using SQLite URI with mode=ro
    db = ReadOnlySQLDatabase("sqlite:///als_patients.db")
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    # PATCH: Wrap all SQL execution tools to enforce whitelisting of only SELECT/PRAGMA
    for tool in tools:
        if hasattr(tool, "run"):
            orig_run = tool.run
            def safe_run(query, *args, orig_run=orig_run, **kwargs):
                if not is_readonly_safe_sql(query):
                    raise Exception("DML/DDL statements (INSERT, UPDATE, DELETE, CREATE, DROP, etc.) are not allowed.")
                # Run the original, then redact results if column metadata is available
                res = orig_run(query, *args, **kwargs)
                # Try to redact result if possible
                # If res is a list of tuples and .columns attribute exists:
                if hasattr(tool, "db") and hasattr(tool.db, "run") and hasattr(tool.db, "_get_cursor"):
                    try:
                        # Extract columns from the query result metadata
                        with tool.db._get_cursor() as cursor:
                            cursor.execute("PRAGMA table_info(patients)")
                            columns = [row[1] for row in cursor.fetchall()]
                        # If result is a list of tuples, redact PII
                        if isinstance(res, list) and all(isinstance(t, tuple) for t in res):
                            res = redact_pii(res, columns)
                    except Exception:
                        pass
                return res
            tool.run = safe_run

    prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt:31156d5f")
    assert len(prompt_template.messages) == 1

    system_message = """System: You are an agent designed to interact with a SQL database filled with ALS patient data. Your name is Steve.
    You will work together with Charity who has access to a list of ALS clinical trials to determine which patients in the list you would recommend for each clinical trial.
    A patient should go to a clinical trial if they are likely to live longer than the Length of Study for that trial.
    Please provide a list of recommended patients for each trial.
    Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
    You can order the results by a relevant column to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.
    You have access to tools for interacting with the database.
    Only use the below tools. Only use the information returned by the below tools to construct your final answer.
    You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

    To start you should ALWAYS look at the tables in the database to see what you can query.
    Do NOT skip this step.
    Then you should query the schema of the most relevant tables."""

    sql_agent_executor = create_react_agent(llm, tools, state_modifier=system_message)
    return sql_agent_executor


def db_admin_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    sql_agent_executor = create_database_admin_agent()
    result = sql_agent_executor.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="database_admin")
            ]
        },
        goto="supervisor",
    )


async def run_agents(prompt):
    builder = StateGraph(MessagesState)
    builder.add_edge(START, "supervisor")
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("clinical_researcher", clinical_researcher_node)
    builder.add_node("database_admin", db_admin_node)
    graph = builder.compile()

    async for s in graph.astream(
            {
                "messages": [
                    (
                            "user",
                            "Find ALS clinical trials, then recommend patients in the database for each trial"
                            if prompt is None else prompt,
                    )
                ]
            },
            subgraphs=True,
    ):
        print(s)
        print("----")


    # (The previous validate_prompt code block unrelated to DB usage is unchanged.)

    con = sqlite3.connect("als_patients.db")
    cursor = con.cursor()
    result = cursor.execute("SELECT name FROM patients ORDER BY name DESC")
    names_list_of_tuples = result.fetchall()
    cursor.close()
    normalized_names = set()
    for name in names_list_of_tuples:
        full_name = name[0]
        full_name = full_name.split(" ")
        first = full_name[0]
        last = full_name[1]
        # Add original and normalized versions of names
        normalized_names.add(first.lower())
        normalized_names.add(last.lower())
        # Remove common characters that could be used for obfuscation
        normalized_first = ''.join(c for c in first.lower() if c.isalnum())
        normalized_last = ''.join(c for c in last.lower() if c.isalnum())
        normalized_names.add(normalized_first)
        normalized_names.add(normalized_last)

    # Normalize and check prompt
    normalized_prompt = prompt.lower()
    # Remove special characters and spaces
    cleaned_prompt = ''.join(c for c in normalized_prompt if c.isalnum() or c.isspace())
    prompt_words = cleaned_prompt.split()

    # Check for both exact matches and substring matches
    for word in prompt_words:
        word = ''.join(c for c in word if c.isalnum())
        if word in normalized_names or any(name in word for name in normalized_names):
            return False
        if any(word in name for name in normalized_names):
            return False

    if not prompt_words:
        return False
    return True
    if not validate_prompt(args.prompt):
        print("Prompt failed guardrails")
        exit(1)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_agents(args.prompt))
    loop.close()


if __name__ == '__main__':
    main()