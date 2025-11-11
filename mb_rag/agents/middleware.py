## middleware / hooks for agents
'''
Middleware Classes:

ContextEditingMiddleware:	Edit context in agent workflows
HumanInTheLoopMiddleware:	Enable human intervention in agent processes
LLMToolSelectorMiddleware:	Select tools using LLM-based logic
LLMToolEmulator:	        Emulate tool execution using LLM
ModelCallLimitMiddleware:	Limit the number of model calls
ModelFallbackMiddleware:	Provide fallback model options
PIIMiddleware:	            Handle personally identifiable information
SummarizationMiddleware: 	Summarize content in agent workflows
TodoListMiddleware:      	Manage to-do lists in agent processes
ToolCallLimitMiddleware:    Limit the number of tool calls
AgentMiddleware:	        Base middleware class for agent operations


Components for middleware functionalities in AI agents.

AgentState:             Function to clear tool usage edits
InterruptOnConfig:      Configuration class for interruption handling
ModelRequest:           Represent a request to a model
ModelResponse:          Represent a response from a model
before_model:           Function executed before model calls
after_model:            Function executed after model calls
wrap_model_call:        Function wrapper for model calls
wrap_tool_call:         Function wrapper for tool calls


'''
from typing import List,Optional,Dict,Any
import time
from langchain.agents.middleware import before_model, after_model, wrap_model_call
from langchain.agents.middleware import AgentState, ModelRequest, ModelResponse

def sql_action_removal(input: str, state: Dict) -> str:
    """
    Middleware to prevent destructive SQL actions.
    Args:
        input (str): The input SQL query.
    Returns:
        str: The original input if no destructive actions are found.
    """
    query = input if isinstance(input, str) else str(input)
    forbidden = ["DROP", "DELETE", "TRUNCATE", "ALTER", "UPDATE", "INSERT", "CREATE", "RENAME", "GRANT", "REVOKE", "COMMIT", "ROLLBACK"]
    if any(cmd in query.upper() for cmd in forbidden):
        raise ValueError(f"[SQLMiddleware] Forbidden SQL command detected: {query}")
    return query


def timing_middleware(timer: int, state: Dict) -> str:
    """
    Middleware to log the time taken for agent execution.
    Args:
        timer (int): The maximum time to wait for the agent to respond.
        state (Dict): The state of the agent.
    Returns:
        str: The original input.
    """
    start_time = time.time()
    result = state.get("input", "")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"[TimingMiddleware] Agent execution time: {elapsed_time:.4f} seconds")
    return result