## give access of db and get the query with tools

from ..prompts_bank import PromptManager
from langgraph.prebuilt import create_react_agent
from langchain_core.agents import tool
from ..basic import ModelFactory

__all__ = ["runtime_sql_agent", "run_sql_agent"]

SYS_PROMPT = PromptManager().get_template("SQL_AGENT_SYS_PROMPT")
default_llm = ModelFactory('openai', model_name='gpt-4o')

class runtime_sql_agent:
    """
    A SQL Agent that can generate and execute SQL queries based on user requests.

    Example:
        agent = runtime_sql_agent(db_connection)
        output = agent.run("Get the top 5 users by activity")
        print(output)
    """

    def __init__(self, db_connection):
        self.db_connection = db_connection

@tool 
def execute_query(self, query: str) -> str:
    """
    Execute a SQL query on the database
    
    Args:
        query: SQL query string
        
    Returns:
        str: Result of the query execution
    """
    try:
        cursor = runtime_sql_agent.db_connection.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        cursor.close()
        return results
    except Exception as e:
        return f"Error executing query: {str(e)}"

class run_sql_agent:
    """
    Create and return a SQL agent instance.
    
    Returns:
        Agent: Configured SQL agent
    """
    def __init__(self,llm=default_llm, db_connection=None,sys_prompt=SYS_PROMPT):
            self.llm=llm
            self.db_connection=db_connection
            self.sys_prompt=sys_prompt
            self.agent=self.create_agent()

    def create_agent(self) -> str:
        '''
        Create and configure the SQL agent
        Returns:
            str: Configured SQL agent
        '''
        return create_react_agent(system_prompt=self.sys_prompt, tools=["execute_query"], model_name=self.llm, context_schema=self.db_connection)

    def run(self, query: str) -> str:
        """
        Run a SQL query using the configured agent.

        Args:
            query: SQL query string

        Returns:
            str: Result of the query execution
        """
        for step in self.agent.stream(
            {'message': query},
            stream_mode = "values",
        ):
            print(step['message'][-1].pretty_print())
