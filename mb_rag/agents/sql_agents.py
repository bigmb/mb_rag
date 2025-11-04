## give access of db and get the query with tools

from ..prompts_bank import PromptManager
from langchain.agents import create_agent
from langchain.tools import tool
import os

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
        sys_prompt: System prompt for the agent.
        langsmith_params: If True, enables LangSmith tracing.
        
    Returns:
        Configured SQL agent.
    """
    
    def __init__(self, llm, db_connection, sys_prompt={'create_text_to_sql_prompt' : SYS_PROMPT}, langsmith_params=True,use_mb: bool=True):
        self.llm = llm
        self.db_connection = db_connection
        self.sys_prompt = sys_prompt
        self.use_mb = use_mb
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

    @tool
    def execute_query(self, query: str,use_mb: bool=True) -> str:
        """
        Execute a SQL query on the database.
        
        Args:
            query: SQL query string.
            use_mb: Whether to use mb_sql for execution.

        Returns:
            str: Result of the query execution.
        """
        try:
            if use_mb:
                results = self.read_sql(query, self.db_connection)
            else:
                results = self.db_connection.execute_query(query)
            return results
        except Exception as e:
            return f"Error executing query: {str(e)}"

    @tool
    def get_table_info(self, table_name: str, schema_name: str) -> str:
        """
        Get information about a specific table in the database.

        Args:
            table_name: Name of the table to retrieve information for.
            schema_name: Name of the schema the table belongs to.

        Returns:
            str: Information about the table.
        """
        query = '''SELECT
                    column_name,
                    data_type,
                    character_maximum_length,
                    is_nullable,
                    column_default
                FROM information_schema.columns
                WHERE table_name = '{table_name}' AND table_schema = '{schema_name}'
                ORDER BY ordinal_position;'''.format(table_name=table_name, schema_name=schema_name)
        return self.read_sql(query, self.db_connection)

    @tool
    def _convert_text_to_sql(self, text: str) -> str:
        """
        Convert natural language text to a SQL query.
        
        Args:
            text: Natural language description of the desired SQL query.
            
        Returns:
            str: Generated SQL query.
        """
        prompt = self.sys_prompt['create_text_to_sql_prompt'].format(text=text)
        sql_query = self.llm.invoke(prompt)
        return sql_query

    def create_sql_agent(self):
        """
        Create and configure the SQL agent.
        
        Returns:
            Configured SQL agent.
        """
        if self.langsmith_params:
            from langsmith import traceable

            @traceable(run_type="agent", name=self.langsmith_name)
            def traced_agent():
                return create_agent(
                    system_prompt=self.sys_prompt,
                    tools=[self.execute_query,self.get_table_info],
                    model_name=self.llm,
                    context_schema=self.db_connection,
                )

            return traced_agent()
        else:
            # No tracing
            return create_agent(
                system_prompt=self.sys_prompt,
                tools=[self.execute_query],
                model_name=self.llm,
                context_schema=self.db_connection,
            )

    def run(self, query: str) -> str:
        """
        Run a SQL query using the configured agent.
        
        Args:
            query: SQL query string.
            
        Returns:
            str: Result of the query execution.
        """
        for step in self.agent.stream(
            {"message": query},
            stream_mode="values",
        ):
            print(step["message"][-1].pretty_print())
