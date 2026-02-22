import re
import base64
import os
import json
from typing import TypedDict, Optional, Dict, Any, List
from ..prompts_bank import PromptManager
from langchain.agents import create_agent
from .middleware import LoggingMiddleware
from langgraph.graph import START, END, StateGraph
from .tools import BBTools
from langsmith import traceable
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from mb.utils.logging import logg

__all__ = ["CreateLabelingAgent", "create_labeling_agent", "LabelingGraph"]

SYS_PROMPT = PromptManager().get_template("BOUNDING_BOX_LABELING_AGENT_SYS_PROMPT")

class CreateLabelingAgent:
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
create_labeling_agent = CreateLabelingAgent


class LabelingState(TypedDict):
    messages: List[Dict[str, Any]]
    bbox_json: Optional[str]
    labeled_objects: Optional[List[Dict[str, Any]]]
    temp_bb_img_path: Optional[str]
    bb_valid: Optional[bool]
    bbox_json_reason: Optional[List[str]]
    query: str
    image_path: str
    failed_labels: Optional[List[str]]

class LabelingGraph:
    """
    Orchestrates a bounding-box labeling pipeline as a LangGraph state graph.

    The graph:
      1. Generates bounding-box labels via an LLM agent.
      2. Draws the boxes on the image.
      3. Validates boxes via LLM; loops back to step 1 on failure.

    Usage::

        labeler = CreateLabelingAgent(llm_labeler)
        validator = CreateLabelingAgent(llm_validator)
        graph = LabelingGraph(agent=labeler, validator_agent=validator)
        result = graph.run("image.jpg", "Label all objects.")
        print(result)

    If ``validator_agent`` is omitted, the labeler agent is reused for
    validation as well.
    """

    def __init__(
        self,
        agent: CreateLabelingAgent,
        validator_agent: CreateLabelingAgent = None,
        logger=None,
        show_images=False,
    ):
        self.agent = agent
        self.validator_agent = validator_agent or agent
        self.logger = logger
        self.show_images = show_images
        self.workflow = self._build_graph()

    @traceable(run_type="chain", name="Labeler Node")
    def node_labeler(self, state: LabelingState):
        """
        Generates or corrects the bounding box JSON based on the initial query
        and any feedback from failed_labels.
        """
        current_query = state["query"]
        if state.get("failed_labels"):
            failed_list = ", ".join(state["failed_labels"])
            correction_prompt = (
                f"{current_query}\n\n"
                f"ATTENTION: The previously generated bounding boxes for the following items "
                f"were marked as incorrect or missing: **{failed_list}**. "
                f"Please review the provided image (which shows the last attempt) and regenerate."
            )
        else:
            correction_prompt = current_query + "\n\nReturn JSON only."

        boxes_json = self.agent.run(correction_prompt, state["image_path"])

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
                "failed_labels": ["All objects (JSON format error)"],
            }

        return {
            **state,
            "messages": [{"role": "agent", "content": boxes_json}],
            "bbox_json": boxes_json,
            "labeled_objects": labeled_objects,
            "failed_labels": None,
        }
    
    @traceable(run_type="tool", name="Bounding Box Visualization Tool")
    def node_tool(self, state: LabelingState):
        """Draws the bounding boxes on the image."""
        tool = BBTools(state['image_path'], logger=self.logger)
        tool._apply_bounding_boxes(
            state["bbox_json"], show=self.show_images, save_location=state['temp_bb_img_path']
        )
        return state

    @traceable(run_type="llm", name="BB Validator LLM Call")
    def _llm_validate_full_list(self, state: LabelingState) -> str:
        """
        Call the LLM for one validation pass on the entire processed image.
        The prompt forces the LLM to return the list of items that failed.
        """
        validation_prompt = f"""
        You are a Bounding Box Validator. Review the image which contains all drawn bounding boxes.
        The objects requested were: {state['query']}

        You must evaluate **every single drawn bounding box** and label.

        Your response must be a JSON object:
        1. **"bb_valid"**: A boolean (true if ALL boxes/labels are correct, false otherwise).
        2. **"failed_labels"**: A list of strings. If "bb_valid" is true, this list is empty: []. \
If "bb_valid" is false, list the **names/labels** of all items that are missing, have incorrect \
bounding boxes, or have incorrect labels (e.g., ["blue chair", "coffee mug (missing)"]).

        Return JSON only.
        """
        return self.validator_agent.run(validation_prompt, state['temp_bb_img_path'])

    @traceable(run_type="chain", name="Validator Node")
    def node_validator(self, state: LabelingState):
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
            return {
                **state,
                "bb_valid": True,
                "labeled_objects": updated_objects,
                "bbox_json": updated_bbox_json,
            }
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
    def route(self, state: LabelingState):
        """Conditional edge: END if valid, otherwise loop back to labeler."""
        return END if state["bb_valid"] else "labeler"

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
    def run(self, image_path: str, query: str,
            temp_image: str = '../data/temp_bb_image.jpg',
            recursion_limit: int = 25):
        """
        Execute the bounding-box labeling pipeline.

        Args:
            image_path: Path to the source image.
            query: The labeling prompt.
            temp_image: Path for temporary bounding-box annotated image.
            recursion_limit: Maximum graph recursion depth.

        Returns:
            Final LabelingState dict after pipeline completion.
        """
        return self.workflow.invoke(
            {
                "image_path": image_path,
                "query": query,
                "temp_bb_img_path": temp_image,
                "bb_valid": False,
                "bbox_json_reason": [],
                "failed_labels": [],
                "messages": [],
                "labeled_objects": [],
            },
            config={"recursion_limit": recursion_limit},
        )