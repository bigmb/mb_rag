'''
File for all tools related functions

State - Mutable data that flows through execution (e.g., messages, counters, custom fields)
Context - Immutable configuration like user IDs, session details, or application-specific configuration
Store - Persistent long-term memory across conversations
Stream Writer - Stream custom updates as tools execute
Config - RunnableConfig for the execution
Tool Call ID - ID of the current tool call
'''

from langchain.tools import tool
from typing import List, Optional
from langchain_community.utilities import SQLDatabase
from mb_rag.prompts_bank import PromptManager
from langchain_core.tools import StructuredTool
from mb_sql.sql import read_sql
from mb_sql.utils import list_schemas

__all__ = ["list_all_tools","SQLDatabaseTools"]

def list_all_tools():
    """
    List all available tools for agents.

    Returns:
        List[str]: List of tool names.
    """
    return [
        "SQLDatabaseTools"
    ]


class SQLDatabaseTools:
    """
    Class to handle SQL Database tools.
    """
    def __init__(self, db_connection):
        self.db_connection = db_connection
        self.read_sql = read_sql
        self.list_schemas = list_schemas

    def _get_table_info(self, table_name: str, schema_name: str) -> str:
        """
        Get information about a specific table in the database.

        Args:
            table_name: Name of the table to retrieve information for.
            schema_name: Name of the schema the table belongs to.
            use_mb: Whether to use mb_sql for execution.

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
        # if use_mb:
        return {"results": self.read_sql(query, self.db_connection)}
        # else:
        #     import pandas as pd
        #     return {"results": pd.read_csv(query, self.db_connection)}
    
    def to_tool_table_info(self):
        return StructuredTool.from_function(
            func=self._get_table_info,
            name="get_table_info",
            description="Get column info for a table",
        )

    def _base_text_to_sql(text: str = None) -> str:
        """
        Convert natural language text to a SQL query.
        
        Args:
            text: Natural language description of the desired SQL query.

        Returns:
            str: Generated SQL query.
        """
        if text:
            prompt = text
        else:
            prompt_manager = PromptManager()
            prompt = prompt_manager.get_template("SQL_AGENT_SYS_PROMPT")
        return prompt
    
    def to_tool_text_to_sql(self):
        return StructuredTool.from_function(
            func=self._base_text_to_sql,
            name="text_to_sql",
            description="Convert natural language text to SQL query",
        )

    def _execute_query_tool(self,
                            query: str, 
                            use_mb: bool) -> str:
        """
        Execute a SQL query on the database.
        Args:
            query: SQL query string.
            use_mb: Whether to use mb_sql for execution.

        Returns:
            str: Result of the query execution.
        """
        try:
            # if use_mb:
            results = self.read_sql(query, self.db_connection)
            results = {'results': results}
            # else:
            #     results = self.db_connection.execute_query(query)
            #     results = {'results': results}
            return results
        except Exception as e:
            return f"Error executing query: {str(e)}"
        
    def to_tool_execute_query(self):
        return StructuredTool.from_function(
            func=self._execute_query_tool,
            name="execute_query",
            description="Execute a SQL query on the database",
        )