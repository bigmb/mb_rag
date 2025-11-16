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
from langchain.agents.middleware import AgentMiddleware,AgentState,ModelRequest,ModelResponse
from langchain.messages import AIMessage
from langgraph.runtime import Runtime
import json

__all__ = [
    "SQLGuardRailsMiddleware",
    "timing_middleware",
    "LoggingMiddleware",
]
        
def timing_middleware(timer: int, state: AgentState) -> str:
    """
    Middleware to log the time taken for agent execution.
    Args:
        timer (int): The maximum time to wait for the agent to respond.
        state (AgentState): The state of the agent.
    Returns:
        str: The original input.
    """
    start_time = time.time()
    result = state.get("input", "")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"[TimingMiddleware] Agent execution time: {elapsed_time:.4f} seconds")
    return result

class SQLGuardRailsMiddleware(AgentMiddleware):
    """
    Middleware to prevent SQL Modifications.
    """
    def _extract_sql_from_state(self, state: AgentState) -> str:
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                fn_call = msg.additional_kwargs.get("function_call", {})
                if fn_call and fn_call.get("name") == "sql_db_query":
                    try:
                        args = json.loads(fn_call.get("arguments", "{}"))
                        return args.get("query", "")
                    except Exception as e:
                        print(f"[SQLGuardRailsMiddleware] Error parsing SQL args: {e}")
        return ""
    
    def after_model(self, state: AgentState, runtime: Runtime) -> Dict[str, Any]:
        input_query = self._extract_sql_from_state(state)
        query_upper = input_query.upper()
        if any(op in query_upper for op in ["UPDATE", "DELETE", "INSERT", "DROP", "ALTER", "CREATE"]):
            print(f"[SQLGuardRailsMiddleware] SQL Table modification access detected: {input_query}")
            return {
                    "messages": [AIMessage("I cannot respond to that request as it involves modifying SQL tables.")],
                    "jump_to": "end"
            }
        return None
        

class LoggingMiddleware(AgentMiddleware):
    """
    Middleware to log agent interactions.
    """
    def before_model(self,state: AgentState) -> None:
        print(f"[LoggingMiddleware] Before model call with input: {state}")
        return None

    def after_model(self, state: AgentState) -> None:
        print(f"[LoggingMiddleware] After model call with output: {state}")
        return None
    
    def after_agent(self, state: AgentState) -> None:
        print(f"[LoggingMiddleware] After agent execution with final state: {state}")
        return None
    










# class SQLGuardRailsMiddleware(AgentMiddleware):
#     """
#     Middleware to prevent SQL Modifications.
#     """
#     def _extract_sql_from_state(self, state: AgentState) -> str:
#         messages = state.get("messages", [])
#         for msg in reversed(messages):
#             if isinstance(msg, AIMessage):
#                 fn_call = msg.additional_kwargs.get("function_call", {})
#                 if fn_call and fn_call.get("name") == "sql_db_query":
#                     try:
#                         args = json.loads(fn_call.get("arguments", "{}"))
#                         return args.get("query", "")
#                     except Exception as e:
#                         print(f"[SQLGuardRailsMiddleware] Error parsing SQL args: {e}")
#         return ""
    
#     def after_model(self, state: AgentState, runtime: Runtime) -> Dict[str, Any]:
#         input_query = self._extract_sql_from_state(state)
#         query_upper = input_query.upper()
#         if any(op in query_upper for op in ["UPDATE", "DELETE", "INSERT", "DROP", "ALTER", "CREATE"]):
#             print(f"[SQLGuardRailsMiddleware] SQL Table modification access detected: {input_query}")
#             return {
#                     "messages": [AIMessage("I cannot respond to that request as it involves modifying SQL tables.")],
#                     "jump_to": "end"
#             }
#         return None