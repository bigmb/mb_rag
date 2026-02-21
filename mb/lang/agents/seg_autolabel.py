import re
import base64
import os
import json
from typing import TypedDict, Optional, Dict, Any, List

from ..prompts_bank import PromptManager
from langchain.agents import create_agent
from .middleware import LoggingMiddleware
from langgraph.graph import START, END, StateGraph
from .tools import SEGTOOLS, BBTools
from langsmith import traceable
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from mb.utils.logging import logg

__all__ = ["SegmentationGraph", "CreateBBAgent", "create_bb_agent"]

SYS_PROMPT = PromptManager().get_template("BOUNDING_BOX_LABELING_AGENT_SYS_PROMPT")

class CreateBBAgent:
    """
    Create and return an AutoLabeling agent instance.

    Args:
        llm: The language model to use.
        langsmith_params: If True, enables LangSmith tracing.
        sys_prompt: System prompt for the agent.
        recursion_limit: Maximum recursion depth for the agent.
        user_name: User identifier for tracing metadata.
        logging: Whether to enable logging.
        logger: Logger instance.
    """
    
    def __init__(self, 
                 llm,
                langsmith_params=True,
                sys_prompt=SYS_PROMPT,
                recursion_limit: int = 3,
                user_name: str = "default_user",
                logging: bool = False,
                logger=None):

        self.llm = llm
        self.langsmith_params = langsmith_params
        self.langsmith_name = os.environ.get("LANGSMITH_PROJECT", "Seg-Labeling-Agent-Project")
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

        self.agent = self._create_agent()

    def _create_agent(self):
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

    def _parse_llm_response(self, response) -> str:
        """
        Parse and clean the LLM response content, stripping markdown
        code fences and extracting raw JSON.
        """
        if isinstance(response.content, list):
            response.content = response.content[0]["text"]
        raw = response.content.strip()

        # Strip markdown code fences (```json ... ``` or ``` ... ```)
        match = re.match(r"^```(?:json)?\s*\n?(.*?)\n?\s*```$", raw, re.DOTALL)
        if match:
            raw = match.group(1).strip()

        return raw

    def _build_messages(self, query: str, *image_paths: str) -> List[Dict[str, Any]]:
        """
        Build a multimodal message list from a query and one or more image paths.

        Raises:
            ValueError: If no image paths are provided.
        """
        if not image_paths:
            raise ValueError("At least one image path must be provided.")
        content: List[Dict[str, Any]] = [{"type": "text", "text": query}]
        for path in image_paths:
            image_base64 = self._image_to_base64(path)
            content.append(
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_base64}"}
            )
        return [
            {"role": "system", "content": self.sys_prompt},
            {"role": "user", "content": content},
        ]

    @traceable(run_type="chain", name="BB Agent Run")
    def run(self, query: str, image: str = None):
        if not image:
            raise ValueError("An image path must be provided.")
        messages = self._build_messages(query, image)
        response = self.llm.invoke(messages)
        return self._parse_llm_response(response)

    @traceable(run_type="chain", name="Validation Segmentation Run")
    def run_seg(self, query: str, image: str = None, image_seg_bb: str = None):
        if not image or not image_seg_bb:
            raise ValueError("Both image and image_seg_bb paths must be provided.")
        messages = self._build_messages(query, image, image_seg_bb)
        response = self.llm.invoke(messages)
        return self._parse_llm_response(response)

    @traceable(run_type="chain", name="Validation Segmentation with Points Run")
    def run_seg_with_points(self, query: str, image: str = None, image_seg_bb: str = None, image_seg_points: str = None):
        if not image or not image_seg_bb or not image_seg_points:
            raise ValueError("All three image paths must be provided.")
        messages = self._build_messages(query, image, image_seg_bb, image_seg_points)
        response = self.llm.invoke(messages)
        return self._parse_llm_response(response)

    @traceable(run_type="tool", name="Image to Base64")
    def _image_to_base64(self, image: str) -> str:
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
            return base64.b64encode(f.read()).decode("utf-8")


# Backward-compatible alias
create_bb_agent = CreateBBAgent


class SegmentationState(TypedDict):
    messages: List[Dict[str, Any]]
    labeled_objects: Optional[List[Dict[str, Any]]] 
    temp_bb_img_path : Optional[str]
    temp_seg_img_path : Optional[str]
    temp_segm_mask_path : Optional[str]
    temp_segm_mask_points_path : Optional[str]
    positive_points : Optional[List[List[int]]]
    negative_points : Optional[List[List[int]]]
    bbox_json_reason : Optional[List[str]]
    bbox_json: Optional[str]
    bb_valid: Optional[bool]
    seg_validation_reason : Optional[List[str]]
    seg_valid: Optional[bool]
    query: str
    image_path: str
    failed_labels: Optional[List[str]]
    failed_segmentation : Optional[List[str]]
    sam_model_path : str
    sam_model_file_path : str

class SegmentationGraph:
    """
    Orchestrates a bounding-box -> segmentation pipeline as a LangGraph state graph.

    The graph:
      1. Generates bounding-box labels via an LLM agent.
      2. Draws the boxes on the image.
      3. Validates boxes via LLM; loops back to step 1 on failure.
      4. Runs SAM segmentation using the bounding boxes.
      5. Validates the segmentation mask; optionally refines with points.

    Usage::

        bb_agent = CreateBBAgent(llm)
        graph = SegmentationGraph(bb_agent)
        result = graph.run("image.jpg", "Detect all chairs")
        print(result)
    """

    def __init__(self, agent: CreateBBAgent, logger=None, show_images=False, sam_predictor=None):
        self.bb_agent = agent
        self.logger = logger
        self.show_images = show_images
        self.sam_predictor = sam_predictor
        self.workflow = self._build_graph()

    @traceable(run_type="chain", name="Labeler Node")
    def node_bb_labeler(self, state: SegmentationState):
        """
        Generates or corrects the bounding box JSON based on the initial query
        and any feedback from failed_labels.
        """
        current_query = state["query"]
        if state.get("failed_labels"):
            failed_list = ", ".join(state["failed_labels"])
            correction_prompt = (
                f"{current_query}\n\n"
                f"ATTENTION: The previously generated bounding boxes for the following items were marked as incorrect or missing: **{failed_list}**. "
                f"Please review the provided image (which shows the last attempt) and regenerate."
            )
        else:
            correction_prompt = current_query + "\n\nReturn JSON only."
            
        boxes_json = self.bb_agent.run(correction_prompt, state["image_path"])
        
        try:
            parsed_data = json.loads(boxes_json)
            labeled_objects = parsed_data.get("labeled_objects", [])
            if not isinstance(labeled_objects, list):
                raise TypeError("Expected 'labeled_objects' to be a list.")
        except (json.JSONDecodeError, TypeError) as e:
            msg = f"Warning: LLM returned invalid JSON format: {e}. Forcing re-run."
            logg.warning(msg, logger=self.logger)
            return {
                **state, 
                "bb_valid": False,
                "failed_labels": ["All objects (JSON format error)"]
            }

        return {
            **state, 
            "messages": [{"role": "agent", "content": boxes_json}], 
            "bbox_json": boxes_json,
            "labeled_objects": labeled_objects, 
            "failed_labels": None
        }
    
    @traceable(run_type="tool", name="Bounding Box Visualization Tool")
    def node_bb_tool(self, state: SegmentationState):
        """Draws the bounding boxes on the image."""
        tool = BBTools(state['image_path'], logger=self.logger)
        tool._apply_bounding_boxes(
            state["bbox_json"], show=self.show_images, save_location=state['temp_bb_img_path']
        )
        return state

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
        Validates all bounding boxes via a single LLM call on the annotated image.
        Updates the per-item 'valid' field on each labeled_object.
        """
        validation_result_json = self._llm_validate_full_list(state)

        try:
            result = json.loads(validation_result_json)
            all_valid = result.get("bb_valid", False)
            failed_labels = result.get("failed_labels", [])
        except json.JSONDecodeError:
            all_valid = False
            failed_labels = ["All labels (Validator JSON error)"]

        # Update per-item 'valid' field based on validation result
        labeled_objects = state.get("labeled_objects", [])
        failed_set = {label.lower() for label in failed_labels}
        updated_objects = []
        for item in labeled_objects:
            item_copy = dict(item)
            if all_valid or item_copy.get("label", "").lower() not in failed_set:
                item_copy["valid"] = True
            else:
                item_copy["valid"] = False
            updated_objects.append(item_copy)

        # Also update bbox_json to reflect the new valid flags
        bbox_data = json.loads(state.get("bbox_json", "{}")) if isinstance(state.get("bbox_json"), str) else state.get("bbox_json", {})
        bbox_data["labeled_objects"] = updated_objects
        updated_bbox_json = json.dumps(bbox_data)

        if all_valid:
            return {**state, "bb_valid": True, "labeled_objects": updated_objects, "bbox_json": updated_bbox_json}
        else:
            msg = f"Validation failed. Items to correct: {failed_labels}"
            logg.warning(msg, logger=self.logger)
            return {
                **state, 
                "bb_valid": False,
                "failed_labels": failed_labels,
                "labeled_objects": updated_objects,
                "bbox_json": updated_bbox_json,
            }

    @traceable
    def bb_route(self, state: SegmentationState):
        """
        Conditional edge: END if valid, otherwise loop back to labeler.
        """
        return "seg_tool_bb" if state["bb_valid"] else "bb_labeler"

    @traceable(run_type="tool", name="Segmentation Visualization Tool")
    def node_seg_tool_bb(self, state: SegmentationState):
        """
        Generate segmentation mask using SAM and bounding box coordinates.
        """
        tool = SEGTOOLS(
            state["image_path"], state['sam_model_path'], state['sam_model_file_path'],
            logger=self.logger, predictor=self.sam_predictor
        )
        tool._apply_segmentation_mask_using_bb(
            state["bbox_json"], show=self.show_images, save_location=state['temp_segm_mask_path']
        )
        return state

    @traceable
    def node_seg_validator_bb(self, state: SegmentationState):
        """
        Validate the segmentation mask produced from bounding boxes.
        """
        validation_prompt = f"""
            You are a Segmentation Validator. Review the segmentation mask and the original image.

            - **Label to Check**: {state['labeled_objects']}
            - The segmentation mask should not include the object's background (inside or outside) or any other objects.

            Your response must be a single JSON object:
            If the mask is correct:
            {{"seg_valid": true}}

            If the mask is incorrect, you MUST suggest positive points (on the object that should be included) and negative points (on background that should be excluded). Use pixel coordinates [x, y]:
            {{"seg_valid": false, "reason": "...", "positive_points": [[x1,y1]], "negative_points": [[x1,y1]]}}

            Return JSON only.
            """

        validation_result_json = self.bb_agent.run_seg(
            validation_prompt,
            state["temp_bb_img_path"],
            state["temp_segm_mask_path"]
        )
        try:
            result = json.loads(validation_result_json)
            return {**state, 
                    "seg_valid": result.get("seg_valid", False),
                    "seg_validation_reason": result.get("reason", ""),
                    "positive_points": result.get("positive_points", []), 
                    "negative_points": result.get("negative_points", [])}
        except json.JSONDecodeError:
            msg = f"Warning: LLM returned invalid JSON format: {validation_result_json}. Forcing re-run."
            logg.warning(msg, logger=self.logger)
            return {**state, 
                    "seg_valid": False,
                    "seg_validation_reason": "Invalid JSON format",
                    "positive_points": [], 
                    "negative_points": []}

    @traceable(run_type="tool", name="Segmentation Points Refinement Tool")
    def node_seg_tool_points(self, state: SegmentationState):
        """
        Refine segmentation mask using SAM with bounding box and positive/negative points.
        """
        tool = SEGTOOLS(
            state["image_path"], state['sam_model_path'], state['sam_model_file_path'],
            logger=self.logger, predictor=self.sam_predictor
        )
        tool._apply_segmentation_mask_using_points(
            bbox_data=state["bbox_json"],
            pos_points=state.get("positive_points", []),
            neg_points=state.get("negative_points", []),
            show=self.show_images,
            save_location=state['temp_segm_mask_points_path']
        )
        return state

    @traceable
    def seg_route1(self,state: SegmentationState):
        return END if state["seg_valid"] else "seg_tool_points"

    @traceable
    def node_seg_validation_points(self, state: SegmentationState):
        """
        Validate the segmentation mask after point-based refinement.
        """
        validation_prompt = f"""
        You are a Segmentation Validator. Review the following mask data and the full image with all boxes drawn on it.
        
        - **Label to Check**: {state['labeled_objects']}

        Based on the visual evidence, is the Mask accurate and tight?
        Your response must be a single JSON object.

        If the mask is correct:
        {{"seg_valid": true}}

        If the mask is incorrect, include positive points (on the object) and negative points (on background to exclude). Use pixel coordinates [x, y]. Start by adding 1 point on either side:
        {{"seg_valid": false, "reason": "...", "positive_points": [[x1,y1],[x2,y2]], "negative_points": [[x1,y1],[x2,y2]]}}

        Return JSON only.
        """
        validation_result_json = self.bb_agent.run_seg_with_points(
            validation_prompt,
            state['temp_bb_img_path'],
            state['temp_segm_mask_path'],
            state['temp_segm_mask_points_path']
        )

        try:
            result = json.loads(validation_result_json)
            return {**state,
                    "seg_valid": result.get("seg_valid", False),
                    "seg_validation_reason": result.get("reason", ""),
                    "positive_points": result.get("positive_points", []), 
                    "negative_points": result.get("negative_points", [])}
        except json.JSONDecodeError:
            msg = f"Warning: LLM returned invalid JSON format in points validation: {validation_result_json}. Forcing re-run."
            logg.warning(msg, logger=self.logger)
            return {**state, 
                    "seg_valid": False,
                    "seg_validation_reason": "Invalid JSON format",
                    "positive_points": [], 
                    "negative_points": []}
        
    @traceable
    def seg_route2(self,state: SegmentationState):
        return END if state["seg_valid"] else "seg_tool_points"


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
            temp_image: str = './data/temp_bb_image.jpg',
            temp_segm_mask_path: str = './data/temp_seg_image_bb.jpg',
            temp_segm_mask_points_path: str = './data/temp_seg_image_points.jpg',
            sam_model_path: str = './models/sam2_hiera_small.pt',
            sam_model_file_path: str = './models/sam2.1_hiera_small.yaml',
            recursion_limit: int = 25):
        """
        Execute the full bounding-box -> segmentation pipeline.

        Args:
            image_path: Path to the source image.
            query: The labeling/segmentation prompt.
            temp_image: Path for temporary bounding-box annotated image.
            temp_segm_mask_path: Path for temporary segmentation mask image.
            temp_segm_mask_points_path: Path for temporary segmentation points image.
            sam_model_path: Path to the SAM2 model weights.
            recursion_limit: Maximum graph recursion depth.

        Returns:
            Final SegmentationState dict after pipeline completion.
        """
        return self.workflow.invoke(
            {
                "image_path": image_path,
                "query": query,
                "temp_bb_img_path": temp_image,
                "temp_segm_mask_path": temp_segm_mask_path,
                "temp_segm_mask_points_path": temp_segm_mask_points_path,
                "bb_valid": False,
                "bbox_json_reason": [],
                "seg_valid": False,
                "sam_model_path": sam_model_path,
                "sam_model_file_path": sam_model_file_path,
                "seg_validation_reason": [],
                "failed_labels": [],
                "failed_segmentation": [],
                "positive_points": [],
                "negative_points": [],
                "messages": [],
                "labeled_objects": [],
            },
            config={"recursion_limit": recursion_limit}
        )
    