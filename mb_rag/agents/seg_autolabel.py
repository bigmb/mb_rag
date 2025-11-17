from ..prompts_bank import PromptManager
from langchain.agents import create_agent
from .middleware import LoggingMiddleware
from mb_rag.utils.viewer import display_graph_png

__all__ = ["create_segmentation_labeling_agent"]

SYS_PROMPT = PromptManager().get_template("SEG_LABELING_AGENT_SYS_PROMPT")

class create_segmentation_labeling_agent:
    """
    Create and return a Segmentation AutoLabeling agent instance.
    
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
        Create and return the Segmentation AutoLabeling agent.

        Returns:
            Configured Segmentation AutoLabeling agent.
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
                                "tags": ['segmentation-labeling-agent-trace'],
                                "metadata": {"user_id": self.user_name},
                               })   
            return traced_agent()
        else:
            return create_agent(
                system_prompt=self.sys_prompt,
                tools=[],
                model=self.llm,
                middleware=self.middleware,
            ).with_config({"recursion_limit": self.recursion_limit,
                            "tags": ['segmentation-labeling-agent-no-trace'],
                            "metadata": {"user_id": self.user_name}
                           })

    def run(self, query: str) -> str:
        """
        Run the Segmentation AutoLabeling agent with the given input query.

        Args:
            query (str): The input query for the agent.

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
            print(f"[Agent Error] {e}")
            return str(e)
        
    def _visualize_agent(self):
        """
        Visualize the agent's structure and components.

        Returns:
            None
        """
        display_graph_png(self.agent)