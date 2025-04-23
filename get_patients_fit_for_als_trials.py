import getpass
import os
import asyncio
from typing import Literal, List
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

import re
import inspect

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
        " respond with FINISH."
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
        llm, tools=clinical_research_tools, state_modifier="You are a ALS clinical researcher. Your name is Charity. Search for ALS clinical trials on https://www.hopkinsmedicine.org/neurology-neurosurgery/clinical-trials/als-clinical-trials/open-trials. Extract text from on the webpage. Let Steve know the Length of Study for each trial."
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

############ VULN PATCH START #############
# --- Safe SQL enforcement layer for DB admin agent ---

import functools

def _is_safe_select_query(query: str, allowed_tables: List[str], allowed_columns: dict) -> bool:
    """
    Enforce that query is a single SELECT statement, not containing wildcards or DML/DDL,
    and only allowed columns from allowed_tables.
    """
    # Block newlines/tabs and uppercase/lowercase through normalization
    normalized = query.strip().replace("\n", " ").replace("\r", " ")
    # Only allow SELECTs, block other statements
    if not re.match(r"(?i)^select\s", normalized):
        return False
    # Disallow ; (multiple statements)
    if ";" in normalized:
        return False
    # Disallow suspicious keywords (DML/DDL/PRAGMA/etc.)
    blacklist = [
        r"(?i)\b(insert|update|delete|drop|create|alter|pragma|attach|detach|vacuum|replace)\b"
    ]
    for pattern in blacklist:
        if re.search(pattern, normalized):
            return False
    # Disallow wildcard "*"
    if re.search(r"select\s*\*", normalized, re.IGNORECASE):
        return False
    # Parse out table and columns (simple regex-based check, not SQL parser)
    # Assume query of form: SELECT col1, col2 FROM tablename ... (no joins, no subqueries)
    select_match = re.match(
        r"(?i)select\s+([a-zA-Z0-9_,\s]+)\s+from\s+([a-zA-Z0-9_]+)", normalized
    )
    if not select_match:
        return False
    cols_str, table = select_match.groups()
    table = table.strip().lower()
    if table not in allowed_tables:
        return False
    # Only allowed columns for this table
    allowed = allowed_columns.get(table, [])
    cols = [c.strip().lower() for c in cols_str.split(",")]
    if not all(c in allowed for c in cols):
        return False
    # Rudimentary check against UNION, subselects, etc.
    if " union " in normalized.lower() or " select " in normalized.lower()[8:]:
        return False
    return True

class SafeSQLDatabaseToolkit(SQLDatabaseToolkit):
    """
    Wraps SQLDatabaseToolkit tools so only allow SELECT queries on whitelisted columns/tables.
    """

    SAFE_TABLES = ["patients"]
    # Should be set according to your DB schema!
    SAFE_COLUMNS = {
        "patients": [
            "patient_id", "diagnosis_date", "expected_survival_months", "age", "gender"  # ADJUST to your schema!
        ]
    }

    def get_tools(self):
        # get the real tools, wrap SQL-execution ones with filter
        tools = super().get_tools()
        for idx, tool in enumerate(tools):
            original_run = tool.run
            # Wrap run method
            @functools.wraps(original_run)
            def safe_run(*args, __orig_run=original_run, **kwargs):
                # Figure out the actual SQL, regardless of tool signature (some tools take 'query', some 'input')
                sql = None
                if "query" in kwargs:
                    sql = kwargs["query"]
                elif "input" in kwargs:
                    sql = kwargs["input"]
                elif len(args) > 0:
                    sql = args[0]
                if sql and isinstance(sql, str):
                    if not _is_safe_select_query(sql, self.SAFE_TABLES, self.SAFE_COLUMNS):
                        return "[SECURITY ERROR] Query rejected: Only SELECT queries on allowed columns/tables are permitted."
                # Call the original if safe OR if not actually executing SQL.
                return __orig_run(*args, **kwargs)
            # Patch the tool to use safe_run (for this instance only)
            tool.run = safe_run
            # Also patch ainvoke/invoke if present
            if hasattr(tool, "ainvoke"):
                orig_ainvoke = tool.ainvoke
                async def safe_ainvoke(*args, __orig_run=original_run, **kwargs):
                    sql = None
                    if "query" in kwargs:
                        sql = kwargs["query"]
                    elif "input" in kwargs:
                        sql = kwargs["input"]
                    elif len(args) > 0:
                        sql = args[0]
                    if sql and isinstance(sql, str):
                        if not _is_safe_select_query(sql, self.SAFE_TABLES, self.SAFE_COLUMNS):
                            return "[SECURITY ERROR] Query rejected: Only SELECT queries on allowed columns/tables are permitted."
                    return await orig_ainvoke(*args, **kwargs)
                tool.ainvoke = safe_ainvoke
        return tools
############ VULN PATCH END #############

def create_database_admin_agent():
    db = SQLDatabase.from_uri("sqlite:///als_patients.db")
    # PATCH: use SafeSQLDatabaseToolkit instead of vanilla toolkit
    toolkit = SafeSQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
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


async def run_agents():
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
                    "Find ALS clinical trials, then recommend patients in the database for each trial",
                )
            ]
        },
        subgraphs=True,
    ):
        print(s)
        print("----")


def main():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_agents())
    loop.close()


if __name__ == '__main__':
    main()