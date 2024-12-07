"""
Agent runner implementation
"""

from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from langchain.agents import AgentExecutor, create_tool_calling_agent
from mb_rag.chatbot.basic import load_model
from mb_rag.utils.extra import check_package
from langchain import hub

__all__ = ['AgentConfig', 'AgentRunner']

def check_hub_dependencies() -> None:
    """
    Check if required hub packages are installed
    Raises:
        ImportError: If any required package is missing
    """
    if not check_package("langchain_hub"):
        raise ImportError("LangChain Hub package not found. Please install it using: pip install langchain-hub")

# Check dependencies before importing
check_hub_dependencies()

@dataclass
class AgentConfig:
    """Configuration for agent runner"""
    model_name: str = "gpt-4o"
    model_type: str = "openai"
    prompt_type: str = "basic"
    verbose: bool = True
    handle_parsing_errors: bool = True

class AgentRunner:
    """
    Class to run AI agents 
    
    Attributes:
        model_type: The language model instance
        tools: List of available tools
    """

    def __init__(self, config: Optional[AgentConfig] = None, **kwargs):
        """
        Initialize agent runner
        Args:
            config: Agent configuration
            **kwargs: Additional arguments
        """
        self.config = config or AgentConfig(**kwargs)
        self._initialize_model()
        self._tools: List[Any] = []

    @classmethod
    def from_model(cls, model_name: str, model_type: str = "openai", **kwargs) -> 'AgentRunner':
        """
        Create agent runner with specific model configuration
        Args:
            model_name: Name of the model
            model_type: Type of model
            **kwargs: Additional configuration
        Returns:
            AgentRunner: Configured agent runner
        """
        config = AgentConfig(
            model_name=model_name,
            model_type=model_type,
            **kwargs
        )
        return cls(config)

    def _initialize_model(self) -> None:
        """Initialize the language model"""
        self.model_type = load_model(
            model_name=self.config.model_name,
            model_type=self.config.model_type
        )
        if self.model_type is None:
            raise ValueError(f"Failed to initialize model {self.config.model_type}")

    @property
    def tools(self) -> List[Any]:
        """Get available tools"""
        return self._tools

    @tools.setter
    def tools(self, tools: List[Any]) -> None:
        """
        Set available tools
        Args:
            tools: List of tools
        """
        if not isinstance(tools, list):
            raise ValueError("Tools must be provided as a list")
        self._tools = tools

    def get_prompt_template(self, prompt_type: Optional[str] = None) -> Any:
        """
        Get prompt template from hub
        Args:
            prompt_type: Type of prompt template
        Returns:
            Any: Prompt template
        """
        try:
            prompt_type = prompt_type or self.config.prompt_type
            if prompt_type == 'basic':
                return hub.pull("hwchase17/openai-tools-agent")
            return hub.pull(prompt_type)
        except Exception as e:
            raise ValueError(f"Error loading prompt template: {str(e)}")

    def create_agent(self, tools: Optional[List[Any]] = None, prompt_type: Optional[str] = None) -> Any:
        """
        Create an agent
        Args:
            tools: List of tools
            prompt_type: Type of prompt template
        Returns:
            Any: Created agent
        """
        if tools:
            self.tools = tools
        if not self.tools:
            raise ValueError("No tools provided. Please provide tools using the tools property or method parameter.")

        try:
            prompt = self.get_prompt_template(prompt_type)
            return create_tool_calling_agent(
                llm=self.model_type,
                tools=self.tools,
                prompt=prompt
            )
        except Exception as e:
            raise ValueError(f"Error creating agent: {str(e)}")

    def create_executor(self, 
                       agent: Any,
                       tools: Optional[List[Any]] = None,
                       verbose: Optional[bool] = None,
                       handle_parsing_errors: Optional[bool] = None,
                       **kwargs) -> AgentExecutor:
        """
        Create agent executor
        Args:
            agent: The agent to execute
            tools: List of tools
            verbose: Whether to be verbose
            handle_parsing_errors: Whether to handle parsing errors
            **kwargs: Additional arguments
        Returns:
            AgentExecutor: Agent executor
        """
        if agent is None:
            raise ValueError("Agent cannot be None. Create an agent first using create_agent().")

        if tools:
            self.tools = tools
        if not self.tools:
            raise ValueError("No tools provided.")

        try:
            return AgentExecutor.from_agent_and_tools(
                agent=agent,
                tools=self.tools,
                verbose=verbose if verbose is not None else self.config.verbose,
                handle_parsing_errors=handle_parsing_errors if handle_parsing_errors is not None else self.config.handle_parsing_errors,
                **kwargs
            )
        except Exception as e:
            raise ValueError(f"Error creating agent executor: {str(e)}")

    @staticmethod
    def invoke_agent(agent: AgentExecutor, query: str, **kwargs) -> Any:
        """
        Invoke an agent
        Args:
            agent: Agent executor to invoke
            query: Query to process
            **kwargs: Additional arguments
        Returns:
            Any: Agent response
        """
        if agent is None:
            raise ValueError("Agent cannot be None. Create an agent first using create_agent().")
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string.")

        try:
            return agent.invoke({"input": query}, **kwargs)
        except Exception as e:
            raise ValueError(f"Error invoking agent: {str(e)}")

    def run_agent_chain(self, query: str, tools: Optional[List[Any]] = None, **kwargs) -> Any:
        """
        Run a complete agent chain
        Args:
            query: Query to process
            tools: Optional tools to use
            **kwargs: Additional arguments
        Returns:
            Any: Agent response
        """
        if tools:
            self.tools = tools
            
        agent = self.create_agent()
        executor = self.create_executor(agent)
        return self.invoke_agent(executor, query, **kwargs)
