from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing import Optional, List, Any, Union

__all__ = [
    'ConversationModel'
]

class ConversationModel:
    """
    A class to handle conversation with AI models
    
    Attributes:
        chatbot: The AI model for conversation
        message_list (List): List of conversation messages
        file_path (str): Path to save/load conversations. Can be local or S3
    """
    
    def __init__(self, 
                llm: Any,
                message_list: Optional[List[Any]] = None,
                file_path: Optional[str] = None,
                **kwargs) -> None:
        """Initialize conversation model"""
        self.chatbot = llm
        if message_list:
            self.message_list = message_list
        else:
            self.message_list = []
        if file_path:
            self.file_path = file_path
        else:
            self.file_path = None

    def initialize_conversation(self,context_message: str = "") -> None:
        """Initialize conversation state.
            Getting the content from file_path if provided"""
        if self.file_path:
            self.load_conversation()

        if context_message:
            self.message_list.append(SystemMessage(content=context_message))
        else:
            self.message_list.append(SystemMessage(content="""This is conversation model. 
                            Look into the conversation history and answer the question if provided.
                            Give a brief introduction of the conversation history."""))
        message_list_content = "".join(self.all_messages_content)
        return self.add_message(message_list_content,get_content_only=True)    

    def _ask_question(self,query: str,images: list = None, 
                     get_content_only: bool = True) -> str:
        """
        Ask a question and get response
        Args:
            query: Question to ask
            get_content_only: Whether to return only content
        Returns:
            str: Response from the model
        """
        if images:
            res = self.chatbot.invoke_query(query,images=images,get_content_only=get_content_only)
        else:
            res = self.chatbot.invoke_query(query,get_content_only=get_content_only)
        return res

    def add_message(self, query: str,images: list = None,get_content_only: bool = True) -> str:
        """
        Add a message to the conversation
        Args:
            query (str): Question to ask
            images (list): List of images to send to the model
            get_content_only (bool): Whether to return only content
        Returns:
            str: Response from the chatbot
        """
        self.message_list.append(HumanMessage(content=query))
        res = self._ask_question(query,images=images,get_content_only=get_content_only)
        self.message_list.append(AIMessage(content=res))
        return res

    @property
    def all_messages(self) -> List[Union[SystemMessage, HumanMessage, AIMessage]]:
        """Get all messages"""
        return self.message_list

    @property
    def last_message(self) -> str:
        """Get the last message"""
        return self.message_list[-1].content

    @property
    def all_messages_content(self) -> List[str]:
        """Get content of all messages"""
        return [message.content for message in self.message_list]

    def _is_s3_path(self, path: str) -> bool:
        """
        Check if path is an S3 path
        Args:
            path (str): Path to check
        Returns:
            bool: True if S3 path
        """
        return path.startswith("s3://")

    def save_conversation(self, file_path: Optional[str] = None, **kwargs) -> bool:
        """
        Save the conversation
        Args:
            file_path: Path to save the conversation
            **kwargs: Additional arguments for S3
        Returns:
            bool: Success status
        """
        if self._is_s3_path(file_path or self.file_path):
            print("Saving conversation to S3.")
            self.save_file_path = file_path
            return self._save_to_s3(self.file_path,**kwargs)
        return self._save_to_file(file_path or self.file_path)

    def _save_to_s3(self,**kwargs) -> bool:
        """Save conversation to S3"""
        try:
            client = kwargs.get('client', self.client)
            bucket = kwargs.get('bucket', self.bucket)
            client.put_object(
                Body=str(self.message_list),
                Bucket=bucket,
                Key=self.save_file_path
            )
            print(f"Conversation saved to s3_path: {self.s3_path}")
            return True
        except Exception as e:
            raise ValueError(f"Error saving conversation to s3: {e}")

    def _save_to_file(self, file_path: str) -> bool:
        """Save conversation to file"""
        try:
            with open(file_path, 'w') as f:
                for message in self.message_list:
                    f.write(f"{message.content}\n")
            print(f"Conversation saved to file: {file_path}")
            return True
        except Exception as e:
            raise ValueError(f"Error saving conversation to file: {e}")

    def load_conversation(self, file_path: Optional[str] = None, **kwargs) -> List[Any]:
        """
        Load a conversation
        Args:
            file_path: Path to load from
            **kwargs: Additional arguments for S3
        Returns:
            List: Loaded messages
        """
        self.message_list = []
        if self._is_s3_path(file_path or self.file_path):
            print("Loading conversation from S3.")
            self.file_path = file_path
            return self._load_from_s3(**kwargs)
        return self._load_from_file(file_path or self.file_path)

    def _load_from_s3(self, **kwargs) -> List[Any]:
        """Load conversation from S3"""
        try:
            client = kwargs.get('client', self.client)
            bucket = kwargs.get('bucket', self.bucket)
            res = client.get_response(client, bucket, self.s3_path)
            res_str = eval(res['Body'].read().decode('utf-8'))
            self.message_list = [SystemMessage(content=res_str)]
            print(f"Conversation loaded from s3_path: {self.file_path}")
            return self.message_list
        except Exception as e:
            raise ValueError(f"Error loading conversation from s3: {e}")

    def _load_from_file(self, file_path: str) -> List[Any]:
        """Load conversation from file"""            
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                self.message_list.append(SystemMessage(content=line))
            print(f"Conversation loaded from file: {file_path}")
            return self.message_list
        except Exception as e:
            raise ValueError(f"Error loading conversation from file: {e}")
