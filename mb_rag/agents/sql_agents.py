## give access of db and get the query with tools

from ..prompts_bank import PromptManager
from langchain.agents import create_agent
from langchain.tools import tool
import os
from .tools import list_all_tools,SQLDatabaseTools
from .middleware import LoggingMiddleware, SQLGuardRailsMiddleware

__all__ = ["runtime_sql_agent", "run_sql_agent"]

SYS_PROMPT = PromptManager().get_template("SQL_AGENT_SYS_PROMPT")

class runtime_sql_agent:
    """
    Setting up the DB connection for SQL agent.

    Example:
        sql_db = runtime_sql_agent(db_connection)
    """

    def __init__(self, db_connection):
        self.db_connection = db_connection
    
    def test_basic_mb(self) -> str:
        """
        Test the database connection by executing a simple query.

        Returns:
            str: Result of the test query
        """
        from mb_rag.utils.extra import check_package
        check_package('mb_sql', 'Please install mb_sql package to use test this function: pip install -U mb_sql')
        from mb_sql.utils import list_schemas,list_tables
        try:
            return print(list_schemas(self.db_connection),f'\n\n',list_tables(self.db_connection, schema='public'))
        except Exception as e:
            return f"Database connection failed: {str(e)}"
    
class run_sql_agent:
    """
    Create and return a SQL agent instance.
    
    Args:
        llm: The language model to use.
        db_connection: The database connection object with an `execute_query` method.
        sys_prompt: System prompt for the agent. (Defaults to SYS_PROMPT)
        langsmith_params: If True, enables LangSmith tracing.
        
    Returns:
        Configured SQL agent.
    """
    
    def __init__(self, llm, db_connection, sys_prompt=SYS_PROMPT, langsmith_params=True,use_mb: bool=True):
        self.llm = llm
        self.db_connection = db_connection
        self.sys_prompt = sys_prompt
        self.use_mb = use_mb
        self.langsmith_params = langsmith_params

        if use_mb:
            from mb_rag.utils.extra import check_package
            check_package('mb_sql', 'Please install mb_sql package to use SQL agent with mb_sql: pip install -U mb_sql')
            from mb_sql.sql import read_sql
            from mb_sql.utils import list_schemas
            self.read_sql = read_sql
            self.list_schemas = list_schemas

        if not self.langsmith_params:
            os.environ["LANGCHAIN_TRACING"] = "false"
        else:
            os.environ.setdefault("LANGCHAIN_TRACING", "true")
            self.langsmith_name = os.environ.get("LANGSMITH_PROJECT", "SQL-Agent-Project")
        self.agent = self.create_sql_agent()

    def create_sql_agent(self):
        """
        Create and configure the SQL agent.
        
        Returns:
            Configured SQL agent.
        """
        # from langchain_core.prompts import ChatPromptTemplate

        # from langchain_core.runnables import RunnableSequence

        # prompt = ChatPromptTemplate.from_template("{input}")
        # llm_chain = RunnableSequence(prompt, self.llm)

        if self.langsmith_params:
            from langsmith import traceable

            @traceable(run_type="chain", name=self.langsmith_name)
            def traced_agent():
                return create_agent(
                    system_prompt=self.sys_prompt,
                    tools=[SQLDatabaseTools(self.db_connection).to_tool_table_info(),
                           SQLDatabaseTools(self.db_connection).to_tool_text_to_sql()],
                    model=self.llm,
                    context_schema=self.db_connection,
                    middleware=[
                        LoggingMiddleware(),
                        SQLGuardRailsMiddleware(),
                    ],
                )

            return traced_agent()
        else:
            # No tracing
            return create_agent(
                system_prompt=self.sys_prompt,
                tools=[SQLDatabaseTools(self.db_connection).to_tool_table_info(),
                       SQLDatabaseTools(self.db_connection).to_tool_text_to_sql()],
                model=self.llm,
                context_schema=self.db_connection,
                middleware=[
                    LoggingMiddleware(),
                    SQLGuardRailsMiddleware(),
                ],
            )
        
    def run(self, query: str) -> str:
        """
        Run a SQL query using the configured agent.
        
        Args:
            query: SQL query string.
            
        Returns:
            str: Result of the query execution.
        """
        try:
            result = self.agent.invoke(query) 
            print(f"Result: {result}")
            print(result["messages"][-1].content)
            return result
        except Exception as e:
            print(f"[Agent Error] {e}")
            return str(e)
        
    def _save_to_db():
        """
        Save agent interactions to the given database.
        
        Returns:
            None
        """
        pass  # Implementation depends on specific database schema and requirements