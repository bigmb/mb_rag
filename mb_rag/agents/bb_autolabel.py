from click import prompt
from ..prompts_bank import PromptManager
from langchain.agents import create_agent
from .middleware import LoggingMiddleware
import base64
import os
from langgraph.graph import Graph, END
from .tools import BBTools

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
                                "tags": ['bb-labeling-agent-trace'],
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
                           "tags": ['bb-labeling-agent-no-trace'],
                           "metadata": {"user_id": self.user_name}
                           })
    
    def run(self, query: str, image: str = None) -> str:
        """
        Run a labeling task using the configured agent.
        
        Args:
            query: Labeling task description.
            image: path of Image file. 

        Returns:
            str: The agent's response.
        """

        try:
            image_base64 = self._image_to_base64(image) if image else self._image_to_base64('./temp_bb_image.jpeg')
            user_content = [
                {"type": "text", "text": query},
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{image_base64}"
                }
            ]

            for step in self.agent.stream(
                {"messages": [{"role": "user", "content": user_content}]},
                stream_mode="values",
            ):
                step["messages"][-1].pretty_print()

        except Exception as e:
            print(f"[Labeling Agent Error] {e}")
            return str(e)

    def _image_to_base64(self,image):
        """
        Convert an image file to a base64-encoded string.
        Args:
            image (str): Path to the image file.
        Returns:
            str: Base64-encoded image string.
        """
        if not os.path.exists(image):
            raise FileNotFoundError(f"Image file not found at path: {image}")

        with open(image, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
        

class LabelingGraph:
    """
    This graph uses:
    - create_labeling_agent
    - BBTools
    and loops until bounding boxes are validated

    Runing :
    agent = create_labeling_agent(llm)
    graph = LabelingGraph(agent, "image.jpg", "Label all objects.")
    result = graph.run()

    print(result)
    """

    def __init__(self, agent: create_labeling_agent, image_path: str, query: str):
        self.agent = agent
        self.image_path = image_path
        self.query = query
        self.workflow = self._build_graph()


    def node_labeler(self, state):
        boxes = self.agent.run(self.query, self.image_path)
        return {"boxes": boxes}

    def node_tool(self, state):
        tool = BBTools(self.image_path)
        tool.apply_bounding_boxes(state["boxes"])
        return {"processed_image": "./temp_bb_image.jpeg"}

    def _llm_validate(self, state):
        """
        Send the processed image (with boxes drawn) back to the LLM
        asking it to validate or correct the bounding boxes.
        """

        validation_prompt = ("""
            "You are checking the visual correctness of bounding boxes.\n"
            "Here is an annotated image.\n"
            "If the bounding boxes are correct, respond with:\n"
            '{"valid": true}\n\n'
            "If any bounding box is incorrect, respond with corrected boxes in the format:\n"
            '{"valid": false, "boxes": { \"label\": [[x0,y0,x1,y1], ...] }}\n"
            "Do not add explanations. JSON only.\n"
                             """
        )

        processed_image_path = "./temp_bb_image.jpeg"

        # Send the image + query into the LLM validator
        result = self.agent.run(validation_prompt, processed_image_path)

        # Expecting JSON result
        return result

    def node_validator(self, state):
        """
        Validation is done by sending the annotated image to the LLM.
        The LLM returns either:
        {"valid": true}
        or
        {"valid": false, "boxes": {...}}
        """

        result = self._llm_validate(state)

        # result might be:
        # {"valid": true}
        # OR
        # {"valid": false, "boxes": {...}}

        # If incorrect → update boxes so the loop can retry
        if not result.get("valid", False):
            return {
                "valid": False,
                "boxes": result["boxes"]   # overwrite incorrect boxes
            }

        return {"valid": True}
    
    def route(self, state):
        return END if state["valid"] else "labeler"

    def _build_graph(self):
        graph = Graph()
        graph.add_node("labeler", self.node_labeler)
        graph.add_node("tool", self.node_tool)
        graph.add_node("validator", self.node_validator)

        graph.add_edge("labeler", "tool")
        graph.add_edge("tool", "validator")

        graph.add_conditional_edges(
            "validator", self.route, {END: END, "labeler": "labeler"}
        )

        return graph.compile()

    def run(self):
        return self.workflow.invoke(
            {
                "agent": self.agent,
                "image_path": self.image_path,
                "query": self.query,
            }
        )