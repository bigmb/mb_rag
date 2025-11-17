from ..prompts_bank import PromptManager
from langchain.agents import create_agent
from .middleware import LoggingMiddleware
from mb_rag.utils.viewer import display_graph_png


__all__ = ["create_labeling_agent"]

SYS_PROMPT = PromptManager().get_template("LABELING_AGENT_SYS_PROMPT")

class create_labeling_agent:
    """
    Create and return an AutoLabeling agent instance.
    
    Args:
        llm: The language model to use.
        langsmith_params: If True, enables LangSmith tracing.
    """
    
    def __init__(self, 
                 llm,
                langsmith_params=True,
                sys_prompt=SYS_PROMPT,
                recursion_limit: int = 50,
                user_name: str = "default_user",
                logging: bool = False):

        self.llm = llm
        self.langsmith_params = langsmith_params
        self.logging = logging
        self.sys_prompt = sys_prompt
        self.recursion_limit = recursion_limit
        self.user_name = user_name
        self.middleware = []
        if self.langsmith_params:
            self.middleware.append(LoggingMiddleware())

        self.agent = self.create_agent()

    def create_agent(self):
        """
        Create and return the AutoLabeling agent.

        Returns:
            Configured AutoLabeling agent.
        """

        if self.langsmith_params:
            from langsmith import traceable

            @traceable(run_type="chain", name=self.langsmith_name)
            def traced_agent():
                return create_agent(
                    system_prompt=self.sys_prompt,
                tools=[],
                    model=self.llm,
                    middleware=self.middleware,
                ).with_config({"recursion_limit": self.recursion_limit,
                                "tags": ['labeling-agent-trace'],
                                "metadata": {"user_id": self.user_name}
                           })

            return traced_agent()
        else:
            # No tracing
            return create_agent(
                system_prompt=self.sys_prompt,
                tools=[],
                model=self.llm,
                middleware=self.middleware,
            ).with_config({"recursion_limit": self.recursion_limit, 
                           "tags": ['labeling-agent-no-trace'],
                           "metadata": {"user_id": self.user_name}
                           })
    
    def run(self, query: str) -> str:
        """
        Run a labeling task using the configured agent.
        
        Args:
            query: Labeling task description.

        Returns:
            str: The agent's response.
        """
        try:
            for step in self.agent.stream(
                {"messages": [{"role": "user", "content": query}]},
                stream_mode="values",
            ):
                step["messages"][-1].pretty_print()
        except Exception as e:
            print(f"[Labeling Agent Error] {e}")
            return str(e)
        
    def _visualize_agent(self) -> None:
        """
        Visualize a graph from its data representation.

        Args:
            graph_data: The data representation of the graph.
        """
        display_graph_png(self.agent)