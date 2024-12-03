## file to run the agent

import importlib.util
from langchain.agents import AgentExecutor, create_tool_calling_agent
from mb_rag.chatbot.basic import load_model
from langchain import hub

def check_package(package_name):
    """
    Check if a package is installed
    Args:
        package_name (str): Name of the package
    Returns:
        bool: True if package is installed, False otherwise
    """
    return importlib.util.find_spec(package_name) is not None


__all__ = ['agent_runner_functions']

class agent_runner_functions:
    """
    Class to run the agent
    Args:
        agent_type (str): Type of the agent
        **kwargs: Additional arguments
    """
    def __init__(self, model_name: str = 'openai', model_type: str = 'gpt-4o', **kwargs):
        self.model_type = load_model(model_name=model_name, model_type=model_type)
        if self.model_type is None:
            raise ValueError(f"Failed to initialize model {model_type}. Please ensure required packages are installed.")

    def prompt_template(self, prompt_type: str = 'basic', **kwargs):
        """
        Function to get the prompt template from hub
        Args:
            prompt_type (str): Type of prompt template
            **kwargs: Additional arguments
        Returns:
            Prompt template from hub
        """
        try:
            if prompt_type == 'basic':
                return hub.pull("hwchase17/openai-tools-agent")
            else:
                return hub.pull(prompt_type)
        except Exception as e:
            raise ValueError(f"Error loading prompt template: {str(e)}. Make sure langchain-hub is installed and the prompt type exists.")
        
    def tools(self, tools: list = []):
        """
        Function to get the tools
        Args:
            tools (list): List of tools functions
        Returns:
            list: List of tools functions
        """
        if len(tools) == 0:
            print("No tools provided")
        self.tools = tools
        return self.tools
        
    def create_agent(self, tools: list = [], prompt_type: str = 'basic'):
        """
        Function to create the agent
        Args:
            tools (list): List of tools functions
            prompt_type (str): Type of prompt template
        Returns:
            AgentExecutor: AgentExecutor object
        Raises:
            ValueError: If tools list is empty or prompt template fails to load
        """
        if tools == []:
            tools = self.tools()
        if not tools:
            raise ValueError("No tools provided. Please provide at least one tool.")
            
        try:
            prompt = self.prompt_template(prompt_type)
            return create_tool_calling_agent(self.model_type, tools, prompt)
        except Exception as e:
            raise ValueError(f"Error creating agent: {str(e)}")

    def agent_executor(self, agent: AgentExecutor, tools: list = [], verbose: bool = True, handle_parsing_errors: bool = True, **kwargs):
        """
        Function to run the agent
        Args:
            agent (AgentExecutor): AgentExecutor object
            tools (list): List of tools functions
            verbose (bool): Whether to print verbose output
            handle_parsing_errors (bool): Whether to handle parsing errors
            **kwargs: Additional arguments
        Returns:
            AgentExecutor: AgentExecutor object
        Raises:
            ValueError: If agent is None or tools list is empty
        """
        if agent is None:
            raise ValueError("Agent cannot be None. Please create an agent first using create_agent().")
            
        if tools == []:
            tools = self.tools()
        if not tools:
            raise ValueError("No tools provided. Please provide at least one tool.")
            
        try:
            return AgentExecutor.from_agent_and_tools(
                agent=agent,
                tools=tools,
                verbose=verbose,
                handle_parsing_errors=handle_parsing_errors,
                **kwargs
            )
        except Exception as e:
            raise ValueError(f"Error creating agent executor: {str(e)}")
    
    def agent_invoke(self, agent: AgentExecutor, query: str, **kwargs):
        """
        Function to invoke the agent
        Args:
            agent (AgentExecutor): AgentExecutor object
            query (str): Query to invoke the agent
            **kwargs: Additional arguments
        Returns:
            AgentExecutor: AgentExecutor object
        Raises:
            ValueError: If agent is None or query is empty
        """
        if agent is None:
            raise ValueError("Agent cannot be None. Please create an agent first using create_agent().")
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string.")
            
        try:
            return agent.invoke({"input": query}, **kwargs)
        except Exception as e:
            raise ValueError(f"Error invoking agent: {str(e)}")
