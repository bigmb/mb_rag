'''
File for all tools related functions
'''

from langchain.tools import tool
from typing import List, Optional
from langchain_community.utilities import SQLDatabase
from mb_rag.prompts_bank import PromptManager

__all__ = ["list_all_tools", "_execute_query_tool","_get_table_info","_base_text_to_sql"]

def list_all_tools():
    """
    List all available tools for agents.

    Returns:
        List[str]: List of tool names.
    """
    return [
        '_execute_query_tool',
        '_get_table_info',
        '_base_text_to_sql',
    ]

@tool
def _execute_query_tool(query: str, 
                        db_connection : SQLDatabase,
                        use_mb: bool = True, ) -> str:
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
            from mb_sql.sql import read_sql
            results = read_sql(query, db_connection)
        else:
            results = db_connection.execute_query(query)
        return results
    except Exception as e:
        return f"Error executing query: {str(e)}"
    
@tool
def _get_table_info(db_connection: SQLDatabase, table_name: str, schema_name: str, use_mb: bool = True) -> str:
    """
    Get information about a specific table in the database.

    Args:
        db_connection: SQLDatabase connection object.
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
    if use_mb:
        from mb_sql.sql import read_sql
        return read_sql(query, db_connection)
    else:
        import pandas as pd
        return pd.read_csv(query, db_connection)
    
@tool
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