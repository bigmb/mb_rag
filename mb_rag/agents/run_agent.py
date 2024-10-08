## file to run the agent

from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from mb_rag.chatbot.basic import get_chatbot_openai,get_chatbot_ollama,get_chatbot_anthropic,get_chatbot_google_generative_ai

__all__ = ['agent_runner_functions']

class agent_runner_functions:
    """
    Class to run the agent
    Args:
        agent_type (str): Type of the agent
        **kwargs: Additional arguments
    """
    def __init__(self,model_type: str = 'openai',**kwargs):
        if model_type == 'openai':
            self.model_type = get_chatbot_openai(**kwargs)
        elif model_type == 'ollama':
            self.model_type = get_chatbot_ollama(**kwargs)
        elif model_type == 'anthropic':
            self.model_type = get_chatbot_anthropic(**kwargs)
        elif model_type == 'google_generative_ai':
            self.model_type = get_chatbot_google_generative_ai(**kwargs)
        else:
            raise ValueError(f"Invalid agent type: {model_type}")

    def prompt_template(self,prompt_type: str = 'basic',**kwargs):
        if prompt_type == 'basic':
            return hub.pull("hwchase17/openai-tools-agent")
        else:
            return hub.pull(prompt_type)
        
    def tools(self,tools: list = []):
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
        
    def create_agent(self,tools: list = [],prompt_type: str = 'basic'):
        """
        Function to create the agent
        Args:
            tools (list): List of tools functions
        Returns:
            AgentExecutor: AgentExecutor object
        """
        if tools == []:
            tools = self.tools()
        prompt = self.prompt_template(prompt_type)
        return create_tool_calling_agent(self.model_type,tools,prompt)

    def agent_executor(self,agent: AgentExecutor,tools: list = [],verbose: bool = True,handle_parsing_errors: bool = True,**kwargs):
        """
        Function to run the agent
        Args:
            agent (AgentExecutor): AgentExecutor object
        Returns:
            AgentExecutor: AgentExecutor object
        """
        if tools == []:
            tools = self.tools()
        return AgentExecutor.from_agent_and_tools(agent,tools,verbose=verbose,handle_parsing_errors=handle_parsing_errors,**kwargs)
    
    def agent_invoke(self,agent: AgentExecutor,query: str,**kwargs):
        """
        Function to invoke the agent
        Args:
            agent (AgentExecutor): AgentExecutor object
            query (str): Query to invoke the agent
        Returns:
            AgentExecutor: AgentExecutor object
        """
        return agent.invoke({"input": query},**kwargs)





