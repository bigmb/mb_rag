from ..prompts_bank import PromptManager
from langchain.agents import create_agent
from .middleware import LoggingMiddleware
import base64
import os
from langgraph.graph import START, END, StateGraph,MessagesState
from .tools import SEGTOOLS,BBTools
from langsmith import traceable
from typing import TypedDict, Optional, Dict, Any,List
import json
from langchain.agents.middleware import ModelCallLimitMiddleware,ToolCallLimitMiddleware

__all__ = ["SegmentationGraph","create_bb_agent"]

SYS_PROMPT = PromptManager().get_template("BOUNDING_BOX_LABELING_AGENT_SYS_PROMPT")

class create_bb_agent:
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
        self.langsmith_name = os.environ.get("LANGSMITH_PROJECT", "Seg-Labeling-Agent-Project")
        self.logging = logging
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


    @traceable(run_type="chain", name="BB Agent Run")
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
        

class SegmentationState(TypedDict):
    messages: List[Dict[str, Any]]
    labeled_objects: Optional[List[Dict[str, Any]]] 
    temp_bb_img_path : Optional[str]
    temp_seg_img_path : Optional[str]
    temp_segm_mask_path : Optional[str]
    positve_points : Optional[List[int]]
    negative_points : Optional[List[int]]
    bbox_json: Optional[str]
    bb_valid: Optional[bool]
    seg_valid: Optional[bool]
    query: str
    image_path: str
    failed_labels: Optional[List[str]]
    failed_segmentation : Optional[List[str]]
    sam_model_path : str

class SegmentationGraph:
    """
    This graph uses:
    - create_segmentations_agents
    - SEGTools
    and loops until Segmentation mask is validated

    Runing :
    bb_agent = create_bb_agent(llm)
    graph = SegmentationGraph(bb_agent, "image.jpg", "Prompt")
    result = graph.run()

    print(result)
    """

    def __init__(self, agent: create_bb_agent):
        self.bb_agent = agent
        self.workflow = self._build_graph()

    @traceable(run_type="chain", name="Labeler Node")
    def node_bb_labeler(self, state: SegmentationState):
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
                
            boxes_json = self.bb_agent.run(correction_prompt, state["image_path"])
            
            try:
                parsed_data = json.loads(boxes_json)
                labeled_objects = parsed_data.get("labeled_objects", [])
                if not isinstance(labeled_objects, list):
                    raise TypeError("Expected 'labeled_objects' to be a list.")
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Warning: LLM returned invalid JSON format: {e}. Forcing re-run.")
                return {
                    **state, 
                    "bb_valid": False,
                    "failed_labels": ["All objects (JSON format error)"]
                }

            print(f"STATE in bb labeler : {state}")
            return {
                **state, 
                "messages": [{"role": "agent", "content": boxes_json}], 
                "bbox_json": boxes_json,
                "labeled_objects": labeled_objects, # Storing the iterable list
                "failed_labels": None
            }
    
    @traceable(run_type="tool", name="Bounding Box Visualization Tool")
    def node_bb_tool(self, state: SegmentationState):
            """Draws the bounding boxes on the image."""
            tool = BBTools(state['image_path'])
            print(f"STATE  : {state}")
            tool._apply_bounding_boxes(state["bbox_json"], show=True, save_location=state['temp_bb_img_path'])
            return state

    @traceable(run_type="llm", name="Validator Single item LLM Call")
    def _llm_validate_single_item(self, state: SegmentationState, item_data: Dict[str, Any]) -> bool:
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
            Your response must be a single JSON object: {{"bb_valid": true}} or {{"bb_valid": false, "reason": "..."}}.
            """
            validation_result_json = self.bb_agent.run(validation_prompt, state['temp_bb_img_path'])
            
            try:
                result = json.loads(validation_result_json)
                return result.get("bb_valid", False)
            except json.JSONDecodeError:
                return False
            
    @traceable(run_type="llm", name="BB Validator LLM Call")
    def _llm_validate_full_list(self, state: SegmentationState) -> str:
        """
        Helper to call the LLM for one validation pass on the entire processed image.
        This prompt forces the LLM to return the list of items that failed.
        """
        validation_prompt = f"""
        You are a Bounding Box Validator. Review the image which contains all drawn bounding boxes.
        The objects requested were: {state['query']}
        
        You must evaluate **every single drawn bounding box** and label.
        
        Your response must be a JSON object:
        1. **"bb_valid"**: A boolean (true if ALL boxes/labels are correct, false otherwise).
        2. **"failed_labels"**: A list of strings. If "bb_valid" is true, this list is empty: []. If "bb_valid" is false, list the **names/labels** of all items that are missing, have incorrect bounding boxes, or have incorrect labels (e.g., ["blue chair", "coffee mug (missing)"]).

        Return JSON only.
        """
        return self.bb_agent.run(validation_prompt, state['temp_bb_img_path'])
    
    @traceable(run_type="chain", name="Validator Node")
    def node_bb_validator(self, state: SegmentationState):
        """
        Iterates over each item and checks its validity.
        """
        
        all_valid = True
        failed_labels = []

        for item in state.get("labeled_objects", []):
            is_valid = item.get("bb_valid", False)            
            if not is_valid:
                break 
        
        validation_result_json = self._llm_validate_full_list(state)
        
        try:
            result = json.loads(validation_result_json)
            all_valid = result.get("bb_valid", False)
            failed_labels = result.get("failed_labels", []) 
        except json.JSONDecodeError:
            all_valid = False
            failed_labels = ["All labels (Validator JSON error)"]
            
        if all_valid:
            print("Validation successful.")
            return {**state, "bb_valid": True}
        else:
            print(f"Validation failed. Items to correct: {failed_labels}")
            return {
                **state, 
                "bb_valid": False,
                "failed_labels": failed_labels 
            }

    @traceable
    def bb_route(self, state: SegmentationState):
        """
        Conditional edge: END if valid, otherwise loop back to labeler.
        """
        return "seg_tool_bb" if state["bb_valid"] else "bb_labeler"

    @traceable(run_type="tool", name="Segmentation Visualization Tool")
    def node_seg_tool_bb(self,state: SegmentationState):
        """
        Tool for get segmentation mask using SAM3 and bounding box. 
        If bounding box doesnt work it will add points to make it better
        """
        tool = SEGTOOLS(state["image_path"])
        tool._apply_segmentation_mask_using_bb(state["bbox_json"])

    @traceable
    def node_seg_validator_bb(self,state: SegmentationState):
        """
        Check the segmentation masks
        """
        validation_prompt = """
        You are a Segmentation Validator. Review the following mask data and the full image with all boxes drawn on it.
        
        - **Label to Check**: {item_data['label']}

        Based on the visual evidence, is the Mask accurate and tight?
        Your response must be a single JSON object: {{"seg_valid": true}} or {{"seg_valid": false, "reason": "..."}}.
        If invalid, suggest points to add more points to improve the mask. Start by adding 1 point on either side depending upon the mask.
        {{positive_points: [[x1,y1],[x2,y2]]}} and
        {{negative_points: [[x1,y1],[x2,y2]]}}.
        """
        validation_result_json = self.bb_agent.run(validation_prompt, state['temp_bb_img_path'],state['temp_seg_mask_path'])
        
        try:
            result = json.loads(validation_result_json)
            return result.get("seg_valid", False)
        except json.JSONDecodeError:
            return False

    @traceable(run_type="tool", name="Segmentation Visualization Tool")
    def node_seg_tool_points(self,state: SegmentationState):
        """
        Tool for get segmentation mask using SAM3 and bounding box. 
        If bounding box doesnt work it will add points to make it better
        """
        tool = SEGTOOLS(state["image_path"],state['sam_model_path'])
        tool._apply_segmentation_mask_using_bb(state["bbox_json"])

    @traceable
    def seg_route1(self,state: SegmentationState):
        return END if state["seg_valid"] else "seg_validation_points"

    @traceable
    def node_seg_validation_points(self,state: SegmentationState):
        """
        Check the segmentation masks after adding points
        """
        validation_prompt = """
        You are a Segmentation Validator. Review the following mask data and the full image with all boxes drawn on it.
        
        - **Label to Check**: {item_data['label']}

        Based on the visual evidence, is the Mask accurate and tight?
        Your response must be a single JSON object: {{"seg_valid": true}} or {{"seg_valid": false, "reason": "..."}}.
        If invalid, suggest points to add more points to improve the mask. Start by adding 1 point on either side depending upon the mask.
        {{positive_points: [[x1,y1],[x2,y2]]}} and
        {{negative_points: [[x1,y1],[x2,y2]]}}.
        """
        validation_result_json = self.bb_agent.run(validation_prompt, state['temp_bb_img_path'],state['temp_seg_mask_path'])
        
        try:
            result = json.loads(validation_result_json)
            return result.get("seg_valid", False)
        except json.JSONDecodeError:
            return False
        
    @traceable
    def seg_route2(self,state: SegmentationState):
        return END if state["seg_valid"] else "seg_tool_validation"


    def _build_graph(self):
        graph = StateGraph(SegmentationState)
        graph.add_node("bb_labeler", self.node_bb_labeler)
        graph.add_node("bb_tool", self.node_bb_tool)
        graph.add_node("bb_validator", self.node_bb_validator)

        graph.add_node("seg_tool_bb", self.node_seg_tool_bb)
        graph.add_node("seg_tool_bb_validation",self.node_seg_validator_bb)

        graph.add_node("seg_tool_points",self.node_seg_tool_points)
        graph.add_node("seg_tool_points_validation",self.node_seg_validation_points)


        graph.add_edge(START, "bb_labeler")
        graph.add_edge("bb_labeler", "bb_tool")
        graph.add_edge("bb_tool", "bb_validator")

        graph.add_conditional_edges(
            "bb_validator", 
            self.bb_route, 
            {
                "seg_tool_bb": "seg_tool_bb",
                "bb_labeler": "bb_labeler"
            }
        )

        graph.add_edge("seg_tool_bb", "seg_tool_bb_validation")

        graph.add_conditional_edges(
            "seg_tool_bb_validation",
            self.seg_route1,
            {
                END: END,
                "seg_tool_points": "seg_tool_points"
            }
        )

        graph.add_edge("seg_tool_points", "seg_tool_points_validation")

        graph.add_conditional_edges(
            "seg_tool_points_validation",
            self.seg_route2,
            {
                END: END,
                "seg_tool_points": "seg_tool_points"
            }
        )
        return graph.compile()

    @traceable
    def run(self, image_path: str, 
            query: str, 
            temp_image :str = './data/temp_bb_image.jpg', 
            temp_seg_mask_path: str = './data/temp_seg_mask.png',
            sam_model_path: str = './models/sam_model.pth'):
        self.image_path = image_path
        self.query = query
        self.temp_image = temp_image
        self.temp_seg_mask_path = temp_seg_mask_path
        self.sam_model_path = sam_model_path
        return self.workflow.invoke(
            {
                "agent": self.bb_agent,
                "image_path": self.image_path,
                "query": self.query,
                "temp_bb_img_path": self.temp_image,
                "temp_segm_mask_path": self.temp_seg_mask_path,
                "bb_valid": 'false',
                "seg_valid": 'false',
                "sam_model_path": self.sam_model_path
            }
        )