"""
Web browsing agent implementation using Google's Gemini model
"""

from langchain_core.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.tools import WebBrowserTools
from bs4 import BeautifulSoup
import requests
from typing import List, Optional
from mb_rag.chatbot.basic import get_chatbot_google_generative_ai

class WebBrowserAgent:
    """
    Agent for web browsing tasks using Google's Gemini model
    
    Attributes:
        model: The Google Generative AI model instance
        search_tool: DuckDuckGo search tool for web queries
        browser_tools: Collection of web browsing tools
    """
    
    def __init__(
        self,
        model_name: str = "gemini-1.5-pro",
        temperature: float = 0.7,
        **kwargs
    ):
        """
        Initialize the web browser agent
        
        Args:
            model_name: Name of the Google Generative AI model to use
            temperature: Temperature for model responses
            **kwargs: Additional arguments for model initialization
        """
        self.model = get_chatbot_google_generative_ai(
            model_name=model_name,
            temperature=temperature,
            **kwargs
        )
        
        # Initialize search capabilities
        search_wrapper = DuckDuckGoSearchAPIWrapper()
        self.search_tool = DuckDuckGoSearchRun(api_wrapper=search_wrapper)
        
        # Initialize web browsing tools
        self.browser_tools = self._get_browser_tools()
        
    def _get_browser_tools(self) -> List[Tool]:
        """
        Create and return the set of web browsing tools
        
        Returns:
            List of Tool objects for web browsing
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
            Parsed text content from the webpage
        """
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
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
            List of URLs found on the webpage
        """
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
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
            Results from the web browsing task
        """
        # If query looks like a URL, fetch it directly
        if query.startswith(('http://', 'https://')):
            return self._fetch_webpage(query)
        
        # Otherwise, perform a search
        search_results = self.search_tool.run(query)
        return search_results
    
    def get_tools(self) -> List[Tool]:
        """
        Get all available web browsing tools
        
        Returns:
            List of Tool objects
        """
        return self.browser_tools
