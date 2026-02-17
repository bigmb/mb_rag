'''
File for all tools related functions

State - Mutable data that flows through execution (e.g., messages, counters, custom fields)
Context - Immutable configuration like user IDs, session details, or application-specific configuration
Store - Persistent long-term memory across conversations
Stream Writer - Stream custom updates as tools execute
Config - RunnableConfig for the execution
Tool Call ID - ID of the current tool call
'''

from langchain.tools import tool
from typing import List, Optional, Any
from langchain_community.utilities import SQLDatabase
from mb_rag.prompts_bank import PromptManager
from langchain_core.tools import StructuredTool
from mb_rag.utils.extra import ImagePredictor
from mb_sql.sql import read_sql
from mb_sql.utils import list_schemas
from PIL import Image,ImageDraw,ImageFont
import os
import warnings
warnings.filterwarnings('ignore', message='Unable to import Axes3D')
import matplotlib.pyplot as plt
import json
from langsmith import traceable
import numpy as np

__all__ = ["list_all_tools","SQLDatabaseTools"]

def list_all_tools():
    """
    List all available tools for agents.

    Returns:
        List[str]: List of tool names.
    """
    return [
        "SQLDatabaseTools"
    ]


class SQLDatabaseTools:
    """
    Class to handle SQL Database tools.
    """
    def __init__(self, db_connection, logger=None):
        self.db_connection = db_connection
        self.logger = logger
        self.read_sql = read_sql
        self.list_schemas = list_schemas

    def _get_database_schemas(self) -> List[str]:
        """
        Get the list of schemas in the database.

        Returns:
            List[str]: List of schema names.
        """
        return self.list_schemas(self.db_connection)
    
    def to_tool_database_schemas(self):
        return StructuredTool.from_function(
            func=self._get_database_schemas,
            name="get_database_schemas",
            description="Get list of schemas in the database",
        )

    def _get_table_info(self, table_name: str, schema_name: str) -> str:
        """
        Get information about a specific table in the database.

        Args:
            table_name: Name of the table to retrieve information for.
            schema_name: Name of the schema the table belongs to.
            use_mb: Whether to use mb_sql for execution.

        Returns:
            str: Information about the table.
        """
        query = '''SELECT
                    column_name,
                    data_type,
                    character_maximum_length,
                    is_nullable,
                    column_default
                FROM information_schema.columns
                WHERE table_name = '{table_name}' AND table_schema = '{schema_name}'
                ORDER BY ordinal_position;'''.format(table_name=table_name, schema_name=schema_name)
        # if use_mb:
        return {"results": self.read_sql(query, self.db_connection)}
        # else:
        #     import pandas as pd
        #     return {"results": pd.read_csv(query, self.db_connection)}
    
    def to_tool_table_info(self):
        return StructuredTool.from_function(
            func=self._get_table_info,
            name="get_table_info",
            description="Get column info for a table",
        )

    def _get_list_tables(self, schema_name: str) -> List[str]:
        """
        Get the list of tables in a specific schema.

        Args:
            schema_name: Name of the schema to list tables from.
        Returns:
            List[str]: List of table names.
        """
        query = '''SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = '{schema_name}';'''.format(schema_name=schema_name)
        results = self.read_sql(query, self.db_connection)
        return {"results": results}
    
    def to_tool_list_tables(self):
        return StructuredTool.from_function(
            func=self._get_list_tables,
            name="list_tables",
            description="List all tables in a schema",
        )

    def _base_text_to_sql(text: str = None) -> str:
        """
        Convert natural language text to a SQL query.
        
        Args:
            text: Natural language description of the desired SQL query.

        Returns:
            str: Generated SQL query.
        """
        if text:
            prompt = text
        else:
            prompt_manager = PromptManager()
            prompt = prompt_manager.get_template("SQL_AGENT_SYS_PROMPT")
        return prompt
    
    def to_tool_text_to_sql(self):
        return StructuredTool.from_function(
            func=self._base_text_to_sql,
            name="text_to_sql",
            description="Convert natural language text to SQL query",
        )

    def _execute_query_tool(self,
                            query: str, 
                            ) -> str:
        """
        Execute a SQL query on the database.
        Args:
            query: SQL query string.
            use_mb: Whether to use mb_sql for execution.

        Returns:
            str: Result of the query execution.
        """
        try:
            # if use_mb:
            results = self.read_sql(query, self.db_connection)
            results = {'results': results}
            # else:
            #     results = self.db_connection.execute_query(query)
            #     results = {'results': results}
            return results
        except Exception as e:
            return f"Error executing query: {str(e)}"
        
    def to_tool_execute_query(self):
        return StructuredTool.from_function(
            func=self._execute_query_tool,
            name="execute_query",
            description="Execute a SQL query on the database",
        )
    
class BBTools:
    """
    Class to handle Bounding Box tools.
    """
    def __init__(self, image_path: str, logger=None):
        self.image_path = image_path
        self.logger = logger

        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image file not found at path: {self.image_path}")

        self.image = self._load_image()

    def _load_image(self) -> Image.Image:
        """
        Load an image from the specified path.
        """
        return Image.open(self.image_path)        

    @traceable(run_type='tool',name='bounding_box_visualizer')
    def _apply_bounding_boxes(self, boxes, show: bool = False, save_location: str = './data/temp_bb_image.jpg') -> Image.Image:
        """
        Draw labeled bounding boxes on the image.
        
        Args:
            boxes: dict or JSON string:
                Expected structure: 
                {"labeled_objects": [{"label": "...", "box": [...]}, ...]}
            show: display the image using matplotlib

        Returns:
            Image.Image: image with bounding boxes and labels
        """
        self.img_bb = self.image.copy()
        draw = ImageDraw.Draw(self.img_bb)

        if isinstance(boxes, str):
            boxes_data = json.loads(boxes)
        elif isinstance(boxes, dict):
            boxes_data = boxes
        else:
            msg = "Error: Invalid boxes format received."
            if self.logger:
                self.logger.error(msg)
            else:
                print(msg)
            return self.img_bb # Return original image on error

        labeled_objects = boxes_data.get("labeled_objects", [])
        
        if not labeled_objects and isinstance(boxes_data, list):
            labeled_objects = boxes_data

        W, H = self.img_bb.size
        
        try:
            font = ImageFont.truetype("./data/arial.ttf",80)
        except IOError:
            msg = "Loading default font"
            if self.logger:
                self.logger.info(msg)
            else:
                print(msg)
            font = ImageFont.load_default()
            
        text_fill_color = "green" 
        text_color = "white"

        for obj in labeled_objects:
            if not obj.get("box") or not isinstance(obj["box"], list) or len(obj["box"]) != 4:
                continue

            label = obj["label"]
            box = obj["box"] 
            
            if 0.0 <= box[0] <= 1.0 and 0.0 <= box[2] <= 1.0:
                x0, y0, x1, y1 = (
                    round(int(box[0] * W),2),
                    round(int(box[1] * H),2),
                    round(int(box[2] * W),2),
                    round(int(box[3] * H),2)
                )
            else:
                # Assume coordinates are absolute if not normalized
                x0, y0, x1, y1 = map(int, box)
            
            x0, x1 = min(x0, x1), max(x0, x1)
            y0, y1 = min(y0, y1), max(y0, y1)

            draw.rectangle([x0, y0, x1, y1], outline="green", width=5)
            
            # try:
            bbox_text = draw.textbbox((0, 0), label, font=font)
            text_w = bbox_text[2] - bbox_text[0]
            text_h = bbox_text[3] - bbox_text[1]
            # except AttributeError:
            #     # Fallback for older PIL versions
            #     text_w, text_h = draw.textsize(label, font=font)

            text_x = x0
            text_y = max(0, y0 - text_h) 

            draw.rectangle(
                [text_x, text_y, text_x + text_w, text_y + text_h],
                fill=text_fill_color
            )
            
            draw.text((text_x, text_y), label, fill=text_color, font=font)

        if show:
            plt.imshow(self.img_bb)
            plt.axis("off")
            plt.show()

        self.img_bb.save(save_location)
        return self.img_bb
    
    def to_tool_bounding_boxes(self):
        return StructuredTool.from_function(
            func=self._apply_bounding_boxes,
            name="apply_bounding_boxes",
            description="Apply bounding boxes on image",
        )


class SEGTOOLS:
    def __init__(self, image_path: str, model_path: str, logger=None, predictor=None):
        
        self.image_path = image_path
        self.logger = logger

        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image file not found at path: {self.image_path}")

        self.image = self._load_image()

        # Use provided predictor or create new one
        if predictor is not None:
            self.predictor = predictor
            self.predictor.set_image(self.image_path)
        else:
            self.predictor = ImagePredictor('./sam2_hiera_s.yaml', model_path)
            self.predictor.set_image(self.image_path)

    def _load_image(self) -> Image.Image:
        """
        Load an image from the specified path.
        """
        return Image.open(self.image_path) 
    
    @traceable(run_type='tool',name='Segmentation BB Visualizer')
    def _apply_segmentation_mask_using_bb(self,
                                          bbox_data: Optional[str], 
                                          show: bool = False, 
                                          save_location: Optional[str] = './data/temp_seg_image_bb.jpg'):
        if len(self.predictor.image.shape) == 2:
            H, W = self.predictor.image.shape
        else:
            H, W, _ = self.predictor.image.shape

        # print(f'Original bbox_data : {bbox_data}')
        bbox_data_loaded = json.loads(bbox_data) if isinstance(bbox_data, str) else bbox_data

        bbox_data = [obj["box"] for obj in bbox_data_loaded.get("labeled_objects", [])]
        bbox_labels = [obj["label"] for obj in bbox_data_loaded.get("labeled_objects", [])]
        
        # print(f'Modified bbox_data : {bbox_data}')

        bbox_data = [[bbox[1]*H, bbox[0]*W, bbox[3]*H, bbox[2]*W] for bbox in bbox_data]
        # print(f'bbox_data (pixel) : {bbox_data}')

        mask, _, _ = self.predictor.predict_item(
            bbox=bbox_data,
            labels_names=bbox_labels,
            gemini_bbox=True
        )
        # print(f'Raw mask shape: {mask.shape}')
        if len(mask.shape) == 4:
            mask = np.squeeze(mask,axis=1)
        # print(f'Squeezed mask shape: {mask.shape}')
        if mask.ndim == 4:                     # (N, C, H, W)
            mask_new = np.transpose(mask, (0, 2, 3, 1))   # (N, H, W, C)
        elif mask.ndim == 3:                   # (C, H, W)
            mask_new = np.transpose(mask, (1, 2, 0))      # (H, W, C)
        else:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")

        if mask_new.shape[-1] == 1:
            mask_new = mask_new[..., 0]      

        if mask_new.ndim == 2:
            masks = [mask_new]
        elif mask_new.ndim == 3:
            if mask_new.shape[-1] > 1:
                masks = [mask_new[..., i] for i in range(mask_new.shape[-1])]
            else:
                masks = [mask_new[i] for i in range(mask_new.shape[0])]
        else:
            raise ValueError(f"Unexpected mask_new shape: {mask_new.shape}")
        
        composite = np.max(np.stack(masks, axis=0), axis=0)   # (H, W)
        if show:
            plt.imshow(composite, cmap='gray')
            plt.axis("off")
            plt.show()

        # composite = np.max(np.stack(masks, axis=0), axis=0)   # (H, W)
        img = Image.fromarray((composite * 255).astype(np.uint8))
        img.save(save_location)

    @traceable(run_type='tool',name='Segmentation Points Visualizer')
    def _apply_segmentation_mask_using_points(self,bbox_data: Optional[str],
                                              pos_points: Optional[list],
                                              neg_points: Optional[list],
                                              show: bool = False, 
                                              save_location: Optional[str] = './data/temp_seg_image_points.jpg'):
        if len(self.predictor.image.shape) == 2:
            H, W = self.predictor.image.shape
        else:
            H, W, _ = self.predictor.image.shape

        # Parse bbox_data in the same way as _apply_segmentation_mask_using_bb
        bbox_data_loaded = json.loads(bbox_data) if isinstance(bbox_data, str) else bbox_data
        bbox_list = [obj["box"] for obj in bbox_data_loaded.get("labeled_objects", [])]
        bbox_labels = [obj["label"] for obj in bbox_data_loaded.get("labeled_objects", [])]
        
        # Convert normalized coordinates to pixel coordinates
        bbox_pixel = [[bbox[1]*H, bbox[0]*W, bbox[3]*H, bbox[2]*W] for bbox in bbox_list]
        
        # Convert points to pixel coordinates (handle empty lists)
        pos_points = pos_points or []
        neg_points = neg_points or []
        pos_points_pixel = [[int(pt[0]*H), int(pt[1]*W)] for pt in pos_points] if pos_points else []
        neg_points_pixel = [[int(pt[0]*H), int(pt[1]*W)] for pt in neg_points] if neg_points else []
        
        all_points = pos_points_pixel + neg_points_pixel
        all_labels = [1]*len(pos_points_pixel) + [0]*len(neg_points_pixel)

        # Predict with bbox, points, and labels
        mask, _, _ = self.predictor.predict_item(
            bbox=bbox_pixel,
            points=all_points if all_points else None,
            point_labels=all_labels if all_labels else None,
            labels_names=bbox_labels,
            gemini_bbox=True
        )
        
        # Process mask using the same robust pattern as bb method
        if len(mask.shape) == 4:
            mask = np.squeeze(mask, axis=1)
            
        if mask.ndim == 4:                     # (N, C, H, W)
            mask_new = np.transpose(mask, (0, 2, 3, 1))   # (N, H, W, C)
        elif mask.ndim == 3:                   # (C, H, W)
            mask_new = np.transpose(mask, (1, 2, 0))      # (H, W, C)
        else:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")

        if mask_new.shape[-1] == 1:
            mask_new = mask_new[..., 0]      

        if mask_new.ndim == 2:
            masks = [mask_new]
        elif mask_new.ndim == 3:
            if mask_new.shape[-1] > 1:
                masks = [mask_new[..., i] for i in range(mask_new.shape[-1])]
            else:
                masks = [mask_new[i] for i in range(mask_new.shape[0])]
        else:
            raise ValueError(f"Unexpected mask_new shape: {mask_new.shape}")
        
        composite = np.max(np.stack(masks, axis=0), axis=0)   # (H, W)
        if show:
            plt.imshow(composite, cmap='gray')
            plt.axis("off")
            plt.show()

        img = Image.fromarray((composite * 255).astype(np.uint8))
        img.save(save_location)