"""
Web browsing agent implementation using Google's Gemini model
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from langchain_core.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
# from langchain.tools import WebBrowserTools
from mb_rag.chatbot.basic import ModelFactory
from mb_rag.utils.extra import check_package
# from duckduckgo_search import DDGS
from langchain_core.messages import HumanMessage
from mb_rag.chatbot.basic import ModelFactory
from typing import Any, Dict

__all__ = [
    'WebBrowserConfig',
    'WebBrowserAgent',
    'WebSearchLangGraphAgent'
]

def check_web_dependencies() -> None:
    """
    Check if required web scraping packages are installed
    Raises:
        ImportError: If any required package is missing
    """
    if not check_package("bs4"):
        raise ImportError("BeautifulSoup4 package not found. Please install it using: pip install beautifulsoup4")
    if not check_package("requests"):
        raise ImportError("Requests package not found. Please install it using: pip install requests")

@dataclass
class WebBrowserConfig:
    """Configuration for web browser agent"""
    model_name: str = "gemini-1.5-pro"
    temperature: float = 0.7
    search_type: str = "duckduckgo"

class WebBrowserAgent:
    """
    Agent for web browsing tasks using Google's Gemini model
    
    Attributes:
        model: The Google Generative AI model instance
        search_tool: Search tool for web queries
        browser_tools: Collection of web browsing tools
    """
    
    def __init__(self, config: Optional[WebBrowserConfig] = None, **kwargs):
        """
        Initialize the web browser agent
        
        Args:
            config: Configuration for the agent
            **kwargs: Additional arguments for model initialization
        """
        # Check dependencies before initializing
        check_web_dependencies()
        
        # Initialize configuration
        self.config = config or WebBrowserConfig(**kwargs)
        
        # Initialize model and tools
        self._initialize_model()
        self._initialize_search()
        self._initialize_browser_tools()
        
        # Import required packages after dependency check
        import requests
        from bs4 import BeautifulSoup
        self._requests = requests
        self._BeautifulSoup = BeautifulSoup

    @classmethod
    def from_model(cls, model_name: str, temperature: float = 0.7, **kwargs) -> 'WebBrowserAgent':
        """
        Create agent with specific model configuration
        
        Args:
            model_name: Name of the model to use
            temperature: Temperature for model responses
            **kwargs: Additional configuration
        Returns:
            WebBrowserAgent: Configured agent
        """
        config = WebBrowserConfig(
            model_name=model_name,
            temperature=temperature
        )
        return cls(config, **kwargs)

    def _initialize_model(self) -> None:
        """Initialize the AI model"""
        self.model = ModelFactory.create_google(
            model_name=self.config.model_name,
            temperature=self.config.temperature
        )

    def _initialize_search(self) -> None:
        """Initialize search capabilities"""
        search_wrapper = DuckDuckGoSearchAPIWrapper()
        self.search_tool = DuckDuckGoSearchRun(api_wrapper=search_wrapper)

    def _initialize_browser_tools(self) -> None:
        """Initialize web browsing tools"""
        self.browser_tools = self._get_browser_tools()

    def _get_browser_tools(self) -> List[Tool]:
        """
        Create and return the set of web browsing tools
        
        Returns:
            List[Tool]: List of web browsing tools
        """
        tools = [
            Tool(
                name="web_search",
                description="Search the web for information using DuckDuckGo",
                func=self.search_tool.run
            ),
            Tool(
                name="fetch_webpage",
                description="Fetch and parse content from a webpage",
                func=self._fetch_webpage
            ),
            Tool(
                name="extract_links",
                description="Extract all links from a webpage",
                func=self._extract_links
            )
        ]
        return tools

    def _fetch_webpage(self, url: str) -> str:
        """
        Fetch and parse content from a webpage
        
        Args:
            url: URL of the webpage to fetch
            
        Returns:
            str: Parsed text content from the webpage
        """
        try:
            response = self._requests.get(url)
            response.raise_for_status()
            soup = self._BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
                
            # Get text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            return f"Error fetching webpage: {str(e)}"

    def _extract_links(self, url: str) -> List[str]:
        """
        Extract all links from a webpage
        
        Args:
            url: URL of the webpage to extract links from
            
        Returns:
            List[str]: List of URLs found on the webpage
        """
        try:
            response = self._requests.get(url)
            response.raise_for_status()
            soup = self._BeautifulSoup(response.text, 'html.parser')
            
            links = []
            for link in soup.find_all('a'):
                href = link.get('href')
                if href and href.startswith('http'):
                    links.append(href)
                    
            return links
        except Exception as e:
            return [f"Error extracting links: {str(e)}"]

    def browse(self, query: str) -> str:
        """
        Execute a web browsing task based on the query
        
        Args:
            query: The search query or URL to process
            
        Returns:
            str: Results from the web browsing task
        """
        if self._is_url(query):
            return self._fetch_webpage(query)
        return self.search_tool.run(query)

    @staticmethod
    def _is_url(text: str) -> bool:
        """
        Check if text is a URL
        
        Args:
            text: Text to check
            
        Returns:
            bool: True if text is a URL
        """
        return text.startswith(('http://', 'https://'))

    @property
    def tools(self) -> List[Tool]:
        """Get all available web browsing tools"""
        return self.browser_tools

    def get_tool(self, name: str) -> Optional[Tool]:
        """
        Get a specific tool by name
        
        Args:
            name: Name of the tool
            
        Returns:
            Optional[Tool]: The requested tool or None if not found
        """
        return next((tool for tool in self.tools if tool.name == name), None)


class WebSearchLangGraphAgent:
    """
    A LangGraph Agent for performing web searches and returning a structured JSON summary.

    Example:
        agent = WebSearchLangGraphAgent(model_name="gpt-4o")
        output = agent.search("Latest developments in renewable energy")
        print(output)
    """

    def __init__(self, model_name: str = "gpt-4o", **kwargs):
        # Initialize the underlying model using ModelFactory
        self.llm = ModelFactory(model_name=model_name, **kwargs).model
        
        from langgraph.graph import MessageGraph
        self.graph = MessageGraph()

        # Add nodes
        self.graph.add_node("user", self._user_node)
        self.graph.add_node("search", self._search_node)
        self.graph.add_node("summarize", self._summarize_node)

        # Set the workflow: user -> search -> summarize
        self.graph.add_edge("user", "search")
        self.graph.add_edge("search", "summarize")

        # Set the entry point of the agent to the user node
        self.graph.set_entry_point("user")

        # Compile the graph into an executable agent
        self.compiled_agent = self.graph.compile()

    @staticmethod
    def perform_web_search(query: str) -> str:
        """
        Perform a live web search using the DuckDuckGo API.
        
        Args:
            query (str): The search query.
            
        Returns:
            str: Concatenated text snippets from the search results.
        """
        ## Uses the DuckDuckGoSearchAPIWrapper
        if not check_package("duckduckgo_search"):
            raise ImportError("Package not found. Please install it using: pip install duckduckgo_search")
        
        from duckduckgo_search import DDGS

        results = DDGS().text(query, max_results=5)
        if isinstance(results, list):
            snippets = [str(item) for item in results]
            return "\n".join(snippets)
        return str(results)

    def _user_node(self, state: Dict[str, Any], input_text: str) -> Dict[str, Any]:
        """
        Node to capture the user's search query.
        """
        state["query"] = input_text
        return state

    def _search_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Node to perform the web search.
        """
        query = state.get("query", "")
        raw_results = WebSearchLangGraphAgent.perform_web_search(query)
        state["raw_results"] = raw_results
        return state

    def _summarize_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Node that uses the LLM to generate a structured JSON summary of the web search results.
        """
        raw_results = state.get("raw_results", "")
        prompt = (
            "You are a helpful AI assistant. Please analyze and summarize the web search results below. "
            "Return a structured JSON with two keys: 'summary' for a concise overview and 'details' for more in-depth information.\n\n"
            f"Results:\n{raw_results}"
        )
        message = HumanMessage(content=prompt)
        response = self.llm.invoke([message])
        try:
            state["structured_output"] = response.content
        except Exception:
            state["structured_output"] = response
        return state

    def search(self, query: str) -> str:
        """
        Perform a web search and obtain a structured JSON summary.
        
        Args:
            query (str): The search query.
        
        Returns:
            str: The structured JSON summary generated by the LLM.
        """
        state: Dict[str, Any] = {}
        state = self.compiled_agent(query, state)
        return state.get("structured_output", "No structured output produced")