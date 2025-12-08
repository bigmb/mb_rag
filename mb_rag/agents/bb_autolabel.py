from ..prompts_bank import PromptManager
from langchain.agents import create_agent
from .middleware import LoggingMiddleware
import base64
import os
from langgraph.graph import START, END, StateGraph,MessagesState
from .tools import BBTools
from langsmith import traceable
from typing import TypedDict, Optional, Dict, Any,List
import json
from langchain.agents.middleware import ModelCallLimitMiddleware,ToolCallLimitMiddleware


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
                logging: bool = False,
                logger=None):

        self.llm = llm
        self.langsmith_params = langsmith_params
        self.langsmith_name = os.environ.get("LANGSMITH_PROJECT", "BB-Labeling-Agent-Project")
        self.logging = logging
        self.logger = logger
        self.sys_prompt = sys_prompt
        self.recursion_limit = recursion_limit
        self.user_name = user_name
        self.middleware = [ModelCallLimitMiddleware(
                            run_limit=3,
                            exit_behavior="end"),
                            ToolCallLimitMiddleware(
                            tool_name="Bounding Box Visualization Tool",
                            run_limit=3,
                            exit_behavior="end")]
        if self.langsmith_params:
            self.middleware.append(LoggingMiddleware())

        self.agent = self.create_agent()

    def create_agent(self):
        """
        Create and return the AutoLabeling agent.

        Returns:
            Configured AutoLabeling agent.
        """

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

    @traceable(run_type="chain", name="Agent Run")
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
        if self.logger:
            self.logger.debug(f'respone from LLM : {response}')
        if type(response.content)==list:
            response.content= response.content[0]['text'] ## for gemini 3 pro preview model
        raw = response.content.strip()

        if raw.startswith("```"):
            raw = raw.strip("` \n")
            if raw.startswith("json"):
                raw = raw[len("json"):].strip()
        return raw
            
    @traceable(run_type="tool", name="Image to Base64")
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
    messages: List[Dict[str, Any]]
    boxes_json: Optional[str]
    labeled_objects: Optional[List[Dict[str, Any]]] 
    temp_bb_img_path : Optional[str]
    valid: Optional[bool]
    query: str
    image_path: str
    failed_labels: Optional[List[str]]

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

    def __init__(self, agent: create_labeling_agent, logger=None):
        self.agent = agent
        self.logger = logger
        self.workflow = self._build_graph()

    @traceable(run_type="chain", name="Labeler Node")
    def node_labeler(self, state: LabelingState):
            """
            Generates/Corrects the bounding box JSON based on the initial query 
            and any feedback from failed_items.
            """
            
            current_query = state["query"]
            if state.get("failed_items"):
                failed_list = ", ".join(state["failed_items"])
                correction_prompt = (
                    f"{current_query}\n\n"
                    f"ATTENTION: The previously generated bounding boxes for the following items were marked as incorrect or missing: **{failed_list}**. "
                    f"Please review the provided image (which shows the last attempt) and **regenerate the complete and correct list of bounding boxes and labels** for all requested items, focusing on correcting the issues for {failed_list}."
                )
            else:
                correction_prompt = current_query
                
            boxes_json = self.agent.run(correction_prompt, state["image_path"])
            
            try:
                parsed_data = json.loads(boxes_json)
                labeled_objects = parsed_data.get("labeled_objects", [])
                if not isinstance(labeled_objects, list):
                    raise TypeError("Expected 'labeled_objects' to be a list.")
            except (json.JSONDecodeError, TypeError) as e:
                msg = f"Warning: LLM returned invalid JSON format: {e}. Forcing re-run."
                if self.logger:
                    self.logger.warning(msg)
                else:
                    print(msg)
                return {
                    **state, 
                    "valid": False,
                    "failed_labels": ["All objects (JSON format error)"]
                }

            return {
                **state, 
                "messages": [{"role": "agent", "content": boxes_json}], 
                "boxes_json": boxes_json,
                "labeled_objects": labeled_objects, # Storing the iterable list
                "failed_labels": None 
            }
    
    @traceable(run_type="tool", name="Bounding Box Visualization Tool")
    def node_tool(self, state: LabelingState):
            """Draws the bounding boxes on the image."""
            tool = BBTools(state['image_path'])
            tool._apply_bounding_boxes(state["boxes_json"], show=True, save_location=state['temp_bb_img_path'])
            return state

    @traceable(run_type="llm", name="Validator Single item LLM Call")
    def _llm_validate_single_item(self, state: LabelingState, item_data: Dict[str, Any]) -> bool:
            """
            Helper to call the LLM to validate a single item.
            The tool must draw ONLY this item's box on a temporary image.
            This is a complex step, as it requires dynamic image generation per loop iteration.
            """
            validation_prompt = f"""
            You are a Bounding Box Validator. Review the following object data and the full image with all boxes drawn on it.
            
            - **Label to Check**: {item_data['label']}
            - **Box Coordinates**: {item_data['box']}
            
            Based on the visual evidence, is the box accurate and tight?
            Your response must be a single JSON object: {{"valid": true}} or {{"valid": false, "reason": "..."}}.
            """
            validation_result_json = self.agent.run(validation_prompt, state['temp_bb_img_path'])
            
            try:
                result = json.loads(validation_result_json)
                return result.get("valid", False)
            except json.JSONDecodeError:
                return False

    @traceable(run_type="chain", name="Validator Node")
    def node_validator(self, state: LabelingState):
        """Iterates over each item and checks its validity."""
        
        updated_objects = []
        all_valid = True
        failed_labels = []

        for item in state.get("labeled_objects", []):
            current_label = item['label']
            is_valid = item.get("valid", False)
            
            if not is_valid:
                break 
        
        validation_result_json = self._llm_validate_full_list(state)
        
        try:
            result = json.loads(validation_result_json)
            all_valid = result.get("valid", False)
            failed_labels = result.get("failed_labels", []) 
        except json.JSONDecodeError:
            all_valid = False
            failed_labels = ["All labels (Validator JSON error)"]
            
        if all_valid:
            msg = "Validation successful."
            if self.logger:
                self.logger.info(msg)
            else:
                print(msg)
            return {**state, "valid": True}
        else:
            msg = f"Validation failed. Items to correct: {failed_labels}"
            if self.logger:
                self.logger.warning(msg)
            else:
                print(msg)
            return {
                **state, 
                "valid": False,
                "failed_labels": failed_labels 
            }
    
    @traceable(run_type="llm", name="Validator LLM Call")
    def _llm_validate_full_list(self, state: LabelingState) -> str:
        """
        Helper to call the LLM for one validation pass on the entire processed image.
        This prompt forces the LLM to return the list of items that failed.
        """
        validation_prompt = f"""
        You are a Bounding Box Validator. Review the image which contains all drawn bounding boxes.
        The objects requested were: {state['query']}
        
        You must evaluate **every single drawn bounding box** and label.
        
        Your response must be a JSON object:
        1. **"valid"**: A boolean (true if ALL boxes/labels are correct, false otherwise).
        2. **"failed_labels"**: A list of strings. If "valid" is true, this list is empty: []. If "valid" is false, list the **names/labels** of all items that are missing, have incorrect bounding boxes, or have incorrect labels (e.g., ["blue chair", "coffee mug (missing)"]).

        Return JSON only.
        """
        return self.agent.run(validation_prompt, state['temp_bb_img_path'])

    
    def route(self, state: LabelingState):
        """Conditional edge: END if valid, otherwise loop back to labeler."""
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
    def run(self, image_path: str, query: str, temp_image :str = './data/temp_bb_image.jpg'):
        self.image_path = image_path
        self.query = query
        self.temp_image = temp_image
        return self.workflow.invoke(
            {
                "agent": self.agent,
                "image_path": self.image_path,
                "query": self.query,
                "temp_bb_img_path": self.temp_image,
                "valid": 'false'
            }
        )