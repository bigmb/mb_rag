from click import prompt
from ..prompts_bank import PromptManager
from langchain.agents import create_agent
from .middleware import LoggingMiddleware
import base64
import os
from langgraph.graph import START, END, StateGraph,MessagesState
from .tools import BBTools
from langsmith import traceable
from typing import TypedDict, Optional, Dict, Any
import json

__all__ = ["create_labeling_agent","LabelingGraph"]

SYS_PROMPT = PromptManager().get_template("BOUNDING_BOX_LABELING_AGENT_SYS_PROMPT")

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
        # if not self.langsmith_params:
        #     os.environ["LANGCHAIN_TRACING"] = "false"
        # else:
        #     os.environ.setdefault("LANGCHAIN_TRACING", "true")
        self.langsmith_name = os.environ.get("LANGSMITH_PROJECT", "BB-Labeling-Agent-Project")
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

        # if self.langsmith_params:
        #     from langsmith import traceable

        @traceable(name=self.langsmith_name)
        def traced_agent():
            return create_agent(
                system_prompt=self.sys_prompt,
            tools=[],
                model=self.llm,
                middleware=self.middleware,
            ).with_config({"recursion_limit": self.recursion_limit,
                            "tags": ['bb-labeling-agent-trace'],
                            "metadata": {"user_id": self.user_name,
                                         "project": self.langsmith_name}
                        })

        return traced_agent()
        # else:
        #     # No tracing
        #     return create_agent(
        #         system_prompt=self.sys_prompt,
        #         tools=[],
        #         model=self.llm,
        #         middleware=self.middleware,
        #     ).with_config({"recursion_limit": self.recursion_limit, 
        #                    "tags": ['bb-labeling-agent-no-trace'],
        #                    "metadata": {"user_id": self.user_name}
        #                    })
    
    def run(self, query: str, image: str = None):
        image_base64 = self._image_to_base64(image) if image else self._image_to_base64('./temp_bb_image.jpeg')

        messages = [
            {"role": "system", "content": self.sys_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_base64}"}
            ]}
        ]

        response = self.llm.invoke(messages,)
        print(f'respone from LLM : {response}')
        raw = response.content.strip()

        if raw.startswith("```"):
            raw = raw.strip("` \n")
            if raw.startswith("json"):
                raw = raw[len("json"):].strip()
        return raw
        # return json.loads(raw)
        
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
        

class LabelingState(TypedDict):
    messages: list
    boxes: Optional[Any]
    processed_image: Optional[str]
    valid: Optional[bool]
    query: str
    image_path: str

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

    def __init__(self, agent: create_labeling_agent): #, image_path: str, query: str):
        self.agent = agent
        # self.image_path = image_path
        # self.query = query
        self.workflow = self._build_graph()


    def node_labeler(self, state):
        boxes = self.agent.run(self.query, self.image_path)
        print(f"BOXES : {boxes}")
        return {**state, "messages": [{"role": "agent", "content": boxes}], "boxes": boxes}


    def node_tool(self, state):
        tool = BBTools(self.image_path)
        # print(state)
        # print(state['boxes'])
        tool._apply_bounding_boxes(state["boxes"],show=True)
        return {**state,"processed_image": "./temp_bb_image.jpeg"}

    def _llm_validate(self, state):
        validation_prompt = """
        You are checking bounding boxes.
        If correct: {"valid": true}
        If incorrect: {"valid": false, "boxes": {...}}
        JSON only.
        """
        processed_image_path = "./temp_bb_image.jpeg"
        return self.agent.run(validation_prompt, processed_image_path)


    def node_validator(self, state):
        result = self._llm_validate(state)

        new_state = dict(state)  # preserve existing state

        if not result.get("valid", False):
            new_state["valid"] = False
            new_state["boxes"] = result["boxes"]
            return new_state

        new_state["valid"] = True
        return new_state

    
    def route(self, state):
        return END if state["valid"] else "labeler"

    def _build_graph(self):
        graph = StateGraph(LabelingState)
        graph.add_node("labeler", self.node_labeler)
        graph.add_node("tool", self.node_tool)
        graph.add_node("validator", self.node_validator)

        graph.add_edge(START, "labeler")
        graph.add_edge("labeler", "tool")
        graph.add_edge("tool", "validator")

        graph.add_conditional_edges(
            "validator", self.route, {END: END, "labeler": "labeler"}
        )

        return graph.compile()

    @traceable
    def run(self, image_path: str, query: str):
        self.image_path = image_path
        self.query = query
        return self.workflow.invoke(
            {
                "agent": self.agent,
                "image_path": self.image_path,
                "query": self.query,
            }
        )