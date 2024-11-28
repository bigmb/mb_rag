"""
Example usage of the web browser agent
"""

from mb_rag.agents.web_browser_agent import WebBrowserAgent
from mb_rag.agents.run_agent import agent_runner_functions

def main():
    # Initialize the web browser agent
    web_agent = WebBrowserAgent()
    
    # Create an agent runner with Google's model
    agent_runner = agent_runner_functions(model_type='google_generative_ai')
    
    # Get the web browsing tools
    tools = web_agent.get_tools()
    
    # Set up the tools for the agent runner
    agent_runner.tools(tools)
    
    # Create the agent
    agent = agent_runner.create_agent(tools=tools)
    
    # Create the agent executor
    agent_executor = agent_runner.agent_executor(agent=agent, tools=tools)
    
    # Example queries
    queries = [
        "What are the latest developments in artificial intelligence?",
        "https://google.com",  # Direct URL fetch
        "Find recent news about space exploration"
    ]
    
    # Run queries
    for query in queries:
        print(f"\nQuery: {query}")
        result = agent_executor.agent_invoke(query)
        print(f"Result: {result}")

if __name__ == "__main__":
    main()
