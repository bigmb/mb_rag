## file for chaining functions in chatbot

from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
from langchain.schema.output_parser import StrOutputParser
from mb_rag.chatbot.prompts import invoke_prompt
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from mb_rag.utils.extra import check_package

__all__ = ['Chain', 'ChainConfig']

def check_langchain_dependencies() -> None:
    """
    Check if required LangChain packages are installed
    Raises:
        ImportError: If any required package is missing
    """
    if not check_package("langchain"):
        raise ImportError("LangChain package not found. Please install it using: pip install langchain")
    if not check_package("langchain_core"):
        raise ImportError("LangChain Core package not found. Please install it using: pip install langchain-core")

# Check dependencies before importing
check_langchain_dependencies()

@dataclass
class ChainConfig:
    """Configuration for chain operations"""
    prompt: Optional[str] = None
    prompt_template: Optional[str] = None
    input_dict: Optional[Dict[str, Any]] = None

class Chain:
    """
    Class to chain functions in chatbot with improved OOP design
    """
    def __init__(self, model: Any, config: Optional[ChainConfig] = None, **kwargs):
        """
        Initialize chain
        Args:
            model: The language model to use
            config: Chain configuration
            **kwargs: Additional arguments
        """
        self.model = model
        self._output_parser = StrOutputParser()
        self._initialize_config(config, **kwargs)

    @classmethod
    def from_template(cls, model: Any, template: str, input_dict: Dict[str, Any], **kwargs) -> 'Chain':
        """
        Create chain from template
        Args:
            model: The language model
            template: Prompt template
            input_dict: Input dictionary for template
            **kwargs: Additional arguments
        Returns:
            Chain: New chain instance
        """
        config = ChainConfig(
            prompt_template=template,
            input_dict=input_dict
        )
        return cls(model, config, **kwargs)

    @classmethod
    def from_prompt(cls, model: Any, prompt: str, **kwargs) -> 'Chain':
        """
        Create chain from direct prompt
        Args:
            model: The language model
            prompt: Direct prompt
            **kwargs: Additional arguments
        Returns:
            Chain: New chain instance
        """
        config = ChainConfig(prompt=prompt)
        return cls(model, config, **kwargs)

    def _initialize_config(self, config: Optional[ChainConfig], **kwargs) -> None:
        """Initialize chain configuration"""
        if config:
            self.input_dict = config.input_dict
            if config.prompt_template:
                self.prompt = invoke_prompt(config.prompt_template, self.input_dict)
            else:
                self.prompt = config.prompt
        else:
            self.input_dict = kwargs.get('input_dict')
            if prompt_template := kwargs.get('prompt_template'):
                self.prompt = invoke_prompt(prompt_template, self.input_dict)
            else:
                self.prompt = kwargs.get('prompt')

    @property
    def output_parser(self) -> StrOutputParser:
        """Get the output parser"""
        return self._output_parser

    @staticmethod
    def _validate_chain_components(prompt: Any, middle_chain: Optional[List] = None) -> None:
        """
        Validate chain components
        Args:
            prompt: The prompt to validate
            middle_chain: Optional middle chain to validate
        Raises:
            ValueError: If validation fails
        """
        if prompt is None:
            raise ValueError("Prompt is not provided")
        if middle_chain is not None and not isinstance(middle_chain, list):
            raise ValueError("middle_chain should be a list")

    def invoke(self) -> Any:
        """
        Invoke the chain
        Returns:
            Any: Output from the chain
        Raises:
            Exception: If prompt is not provided
        """
        self._validate_chain_components(self.prompt)
        chain_output = self.prompt | self.model | self.output_parser
        return chain_output

    def chain_sequence_invoke(self, 
                            middle_chain: Optional[List] = None,
                            final_chain: Optional[RunnableLambda] = None) -> Any:
        """
        Chain invoke the sequence
        Args:
            middle_chain: List of functions/Prompts/RunnableLambda to chain
            final_chain: Final chain to run
        Returns:
            Any: Output from the chain
        """
        self._validate_chain_components(self.prompt, middle_chain)
        
        final = final_chain if final_chain is not None else self.output_parser
        
        if middle_chain:
            func_chain = RunnableSequence(self.prompt, middle_chain, final)
            return func_chain.invoke()
        return None

    def chain_parallel_invoke(self, parallel_chain: List) -> Any:
        """
        Chain invoke in parallel
        Args:
            parallel_chain: List of chains to run in parallel
        Returns:
            Any: Output from the parallel chains
        Raises:
            ImportError: If LangChain is not installed
        """
        if not check_package("langchain"):
            raise ImportError("LangChain package not found. Please install it using: pip install langchain")
        return parallel_chain.invoke()

    def chain_branch_invoke(self, branch_chain: Dict) -> Any:
        """
        Chain invoke with branching
        Args:
            branch_chain: Dictionary of branch chains
        Returns:
            Any: Output from the branch chain
        Raises:
            ImportError: If LangChain is not installed
        """
        if not check_package("langchain"):
            raise ImportError("LangChain package not found. Please install it using: pip install langchain")
        return branch_chain.invoke()

    @staticmethod
    def create_parallel_chain(prompt_template: str, model: Any, branches: Dict[str, Any]) -> Any:
        """
        Create a parallel chain
        Args:
            prompt_template: Template for the prompt
            model: The language model
            branches: Dictionary of branch configurations
        Returns:
            Any: Configured parallel chain
        """
        from langchain.schema.runnable import RunnableParallel
        return (
            prompt_template
            | model
            | StrOutputParser()
            | RunnableParallel(branches=branches)
        )

    @staticmethod
    def create_branch_chain(conditions: List[tuple], default_chain: Any) -> Any:
        """
        Create a branch chain
        Args:
            conditions: List of condition-chain tuples
            default_chain: Default chain to use
        Returns:
            Any: Configured branch chain
        """
        from langchain.schema.runnable import RunnableBranch
        return RunnableBranch(*conditions, default_chain)
