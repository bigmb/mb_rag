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
from mb_sql.sql import read_sql
from mb_sql.utils import list_schemas
from PIL import Image,ImageDraw,ImageFont
import os
import matplotlib.pyplot as plt
import json

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
    def __init__(self, db_connection):
        self.db_connection = db_connection
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
    def __init__(self, image_path: str):
        self.image_path = image_path

        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image file not found at path: {self.image_path}")

        self.image = self._load_image()

    def _load_image(self) -> Image.Image:
        """
        Load an image from the specified path.
        """
        return Image.open(self.image_path)        

    def _apply_bounding_boxes(self, boxes, show: bool = False,save_location: str = './temp_bb_image.jpeg') -> Image.Image:
        """
        Draw labeled bounding boxes on the image.

        Args:
            boxes: dict or JSON string:
                { "label": [[x_min, y_min, x_max, y_max]], ... }
            show: display the image using matplotlib

        Returns:
            Image.Image: image with bounding boxes and labels
        """
        self.img_bb = self.image.copy()
        draw = ImageDraw.Draw(self.img_bb)

        boxes = json.loads(boxes) if isinstance(boxes, str) else boxes

        W, H = self.img_bb.size

        font = ImageFont.load_default()

        for label, box_list in boxes.items():
            for box in box_list:

                if 0 <= box[0] <= 1 and 0 <= box[2] <= 1:
                    x0, y0, x1, y1 = (
                        int(box[0] * W),
                        int(box[1] * H),
                        int(box[2] * W),
                        int(box[3] * H)
                    )
                else:
                    x0, y0, x1, y1 = map(int, box)

                draw.rectangle([x0, y0, x1, y1], outline="green", width=3)
                # # text_w, text_h = draw.textsize(label, font=font)
                # draw.rectangle([x0, y0 - text_h, x0 + text_w, y0], fill="green")
                # draw.text((x0, y0 - text_h), label, fill="white", font=font)

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
