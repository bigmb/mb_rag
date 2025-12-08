"""
Base Agent runner implementation
"""

from typing import List,Optional
from mb_rag.utils import viewer
from mb_rag.agents import get_langsmith
from mb_rag.basic import ModelFactory
import os 
from mb_rag.agents.tools import list_all_tools

__all__ = ["AgentRunner"]


class AgentDict:
    """
    Base class to run AI agents 
    
    Attributes:
        llm : Language model instance. (ModelFactory : mb_rag.basic.ModelFactory)
        tools (List[str]) : List of available tools (Check mb_rag.agents.tools for available tools)
        agent_instance (str): The agent instance
        langsmith_params (dict): Langsmith parameters if any. {langsmith_api_key, langsmith_endpoint, langsmith_project, langsmith_tracing}
        middleware (List[str]): Middleware for the agent
        extra_params (dict): Any extra parameters
    """
    def __init__(self,
                llm : ModelFactory,
                agent_instance : str,
                tools: Optional[List[str]] = None,
                langsmith_params: Optional[dict] = None,
                middleware: Optional[List[str]] = None,
                extra_params: Optional[dict] = None,
                logger=None):

        self.llm = llm
        self.tools = tools if tools else []
        self.agent_instance = agent_instance if agent_instance else None
        self.langsmith_params = langsmith_params if langsmith_params else {}
        self.middleware = middleware if middleware else []
        self.extra_params = extra_params if extra_params else {}
        self.logger = logger

    def _supervise_agent(self, input_query: str) -> str:
        """
        Supervise the agent's execution with the given input query.

        Args:
            input_query (str): The input query for the agent.

        Returns:
            str: The agent's response.
        """
        try:
            if self.langsmith_params:
                self._load_langsmith()
            agent = self.create_main_agent()
            response = agent.run(input_query)
            viewer.display_response(response)
            return response
        except Exception as e:
            viewer.display_error(e)
            msg = f"[Agent Error] {e}"
            if self.logger:
                self.logger.error(msg)
            else:
                print(msg)
            return str(e)

    def _support_agent(self, input_query: str) -> str:
        """
        Support the agent's execution with the given input query.

        Args:
            input_query (str): The input query for the agent.

        Returns:
            str: The agent's response.
        """
        try:
            if self.langsmith_params:
                self._load_langsmith()
            agent = self.create_main_agent()
            response = agent.run(input_query)
            viewer.display_response(response)
            return response
        except Exception as e:
            viewer.display_error(e)
            msg = f"[Agent Error] {e}"
            if self.logger:
                self.logger.error(msg)
            else:
                print(msg)
            return str(e)

    def create_main_agent(self):
        """
        Create and configure the main agent.
        
        Returns:
            Configured main agent.
        """
        pass  # To be implemented in subclasses

    def _get_all_tools(self):
        """
        Retrieve all available tools for the agent.
        
        Returns:
            List of tools.
        """
        return self.tools
    
    def _list_all_tools(self):
        """
        List all available tools for the agent.
        
        Returns:
            List of tool names.
        """
        return list_all_tools()

    def _get_all_middlewares(self):
        """
        Retrieve all middlewares for the agent.
        
        Returns:
            List of middlewares.
        """
        return self.middleware
    
    def _get_all_extra_params(self):
        """
        Retrieve all extra parameters for the agent.
        
        Returns:
            Dictionary of extra parameters.
        """
        return self.extra_params
    
    def _get_langsmith_params(self):
        """
        Retrieve Langsmith parameters for the agent.
        
        Returns:
            Dictionary of Langsmith parameters.
        """
        return self.langsmith_params
    
    def _load_langsmith(self):
        """
        Load Langsmith parameters into environment variables.
        
        Returns:
            None
        """
        if os.getenv('LANGSMITH_API_KEY') is not None:
                langsmith_api_key=self.langsmith_params.get('langsmith_api_key', None)
        get_langsmith.set_langsmith_parameters(
            langsmith_api_key=langsmith_api_key,
            langsmith_endpoint=self.langsmith_params.get('langsmith_endpoint', 'https://api.smith.langchain.com'),
            langsmith_project=self.langsmith_params.get('langsmith_project', 'No Project'),
            langsmith_tracing=self.langsmith_params.get('langsmith_tracing', 'false')
        )
        msg = "Langsmith parameters: {} loaded into environment variables.".format(self.langsmith_params)
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)
        return None
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)
        return None
    
