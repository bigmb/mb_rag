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
