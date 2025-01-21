import os
from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from IPython.display import display, HTML
from typing import Optional, List, Dict, Any, Union
from mb_rag.utils.extra import check_package
import base64

__all__ = [
    'ChatbotBase',
    'ModelFactory',
    'ConversationModel',
    'IPythonStreamHandler'
]

class ChatbotBase:
    """Base class for chatbot functionality"""
    
    @staticmethod
    def load_env(file_path: str) -> None:
        """
        Load environment variables from a file
        Args:
            file_path (str): Path to the environment file
        """
        load_dotenv(file_path)
    
    @staticmethod
    def add_os_key(name: str, key: str) -> None:
        """
        Add an API key to the environment
        Args:
            name (str): Name of the API key
            key (str): API key
        """
        os.environ[name] = key

    @staticmethod
    def get_client():
        """
        Returns a boto3 client for S3
        """
        if not check_package("boto3"):
            raise ImportError("Boto3 package not found. Please install it using: pip install boto3")
        
        import boto3
        return boto3.client('s3')

class ModelFactory:
    """Factory class for creating different types of chatbot models"""
    
    def __init__(self, model_type: str = 'openai', model_name: str = "gpt-4o", **kwargs) -> Any:
        """
        Factory method to create any type of model
        Args:
            model_type (str): Type of model to create. Default is OpenAI. Options are openai, anthropic, google, ollama , groq
            model_name (str): Name of the model
            **kwargs: Additional arguments
        Returns:
            Any: Chatbot model
        """
        creators = {
            'openai': self.create_openai,
            'anthropic': self.create_anthropic,
            'google': self.create_google,
            'ollama': self.create_ollama,
            'groq': self.create_groq
        }
        
        model_data = creators.get(model_type)
        if not model_data:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        try:
            self.model = model_data(model_name, **kwargs)
        except Exception as e:
            raise ValueError(f"Error creating {model_type} model: {str(e)}")
        
    @classmethod
    def create_openai(cls, model_name: str = "gpt-4o", **kwargs) -> Any:
        """
        Create OpenAI chatbot model
        Args:
            model_name (str): Name of the model
            **kwargs: Additional arguments
        Returns:
            ChatOpenAI: Chatbot model
        """
        if not check_package("openai"):
            raise ImportError("OpenAI package not found. Please install it using: pip install openai langchain-openai")
        
        from langchain_openai import ChatOpenAI
        kwargs["model_name"] = model_name
        return ChatOpenAI(**kwargs)

    @classmethod
    def create_anthropic(cls, model_name: str = "claude-3-opus-20240229", **kwargs) -> Any:
        """
        Create Anthropic chatbot model
        Args:
            model_name (str): Name of the model
            **kwargs: Additional arguments
        Returns:
            ChatAnthropic: Chatbot model
        """
        if not check_package("anthropic"):
            raise ImportError("Anthropic package not found. Please install it using: pip install anthropic langchain-anthropic")
        
        from langchain_anthropic import ChatAnthropic
        kwargs["model_name"] = model_name
        return ChatAnthropic(**kwargs)

    @classmethod
    def create_google(cls, model_name: str = "gemini-1.5-flash", **kwargs) -> Any:
        """
        Create Google chatbot model
        Args:
            model_name (str): Name of the model
            **kwargs: Additional arguments
        Returns:
            ChatGoogleGenerativeAI: Chatbot model
        """
        if not check_package("langchain_google_genai"):
            raise ImportError("langchain_google_genai package not found. Please install it using: pip install google-generativeai langchain-google-genai")
        
        from langchain_google_genai import ChatGoogleGenerativeAI
        kwargs["model"] = model_name
        return ChatGoogleGenerativeAI(**kwargs)

    @classmethod
    def create_ollama(cls, model_name: str = "llama3", **kwargs) -> Any:
        """
        Create Ollama chatbot model
        Args:
            model_name (str): Name of the model
            **kwargs: Additional arguments
        Returns:
            Ollama: Chatbot model
        """
        if not check_package("langchain_community"):
            raise ImportError("Langchain Community package not found. Please install it using: pip install langchain-community")
        
        from langchain_community.llms import Ollama
        kwargs["model"] = model_name
        return Ollama(**kwargs)

    @classmethod
    def create_groq(cls, model_name: str = "llama-3.3-70b-versatile", **kwargs) -> Any:
        """
        Create Groq chatbot model
        Args:
            model_name (str): Name of the model
            **kwargs: Additional arguments. Options are: temperature, groq_api_key, model_name
        Returns:
            ChatGroq: Chatbot model
        """
        if not check_package("langchain-groq"):
            raise ImportError("Langchain Groq package not found. Please install it using: pip install langchain-groq")

        from langchain_groq import ChatGroq
        kwargs["model"] = model_name
        return ChatGroq(**kwargs)

    def invoke_query(self,query: str,get_content_only: bool = True,images: list = None,pydantic_model = None) -> str:
        """
        Invoke the model
        Args:
            query (str): Query to send to the model
            get_content_only (bool): Whether to return only content
            images (list): List of images to send to the model
            pydantic_model: Pydantic model for structured output
        Returns:
            str: Response from the model
        """
        
        if pydantic_model is not None:
            try:
                self.model = self.model.with_structured_output(pydantic_model)
            except Exception as e:
                raise ValueError(f"Error with pydantic_model: {e}")
        if images:
            res = self._model_invoke_images(images=images,prompt=query,pydantic_model=pydantic_model)
        else:
            res = self.model.invoke(query)
            if get_content_only:
                try:
                    return res.content
                except Exception:
                    return res
        return res

    def _image_to_base64(self,image):
        with open(image, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def _model_invoke_images(self,images: list, prompt: str,pydantic_model = None):
        """
        Function to invoke the model with images
        Args:
            model (ChatOpenAI): Chatbot model
            images (list): List of images
            prompt (str): Prompt
            pydantic_model (PydanticModel): Pydantic model
    Returns:
        str: Output from the model
    """
        base64_images = [self._image_to_base64(image) for image in images]
        image_prompt_create = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_images[i]}"}} for i in range(len(images))]
        prompt_new = [{"type": "text", "text": prompt},
                      *image_prompt_create,]
        if pydantic_model is not None:
            try:
                self.model = self.model.with_structured_output(pydantic_model)
            except Exception as e:
                print(f"Error with pydantic_model: {e}")
                print("Continuing without structured output")
        message= HumanMessage(content=prompt_new,)
        response = self.model.invoke([message])
        return response.content


class ConversationModel:
    """
    A class to handle conversation with AI models
    
    Attributes:
        chatbot: The AI model for conversation
        message_list (List): List of conversation messages
        file_path (str): Path to save/load conversations. Can be local or S3
    """
    
    def __init__(self, 
                 model_name: str = "gpt-4o",
                 model_type: str = 'openai',
                 **kwargs) -> None:
        """Initialize conversation model"""
        self.chatbot = ModelFactory(model_type, model_name, **kwargs)

    def initialize_conversation(self,
                                question: Optional[str],
                                context: Optional[str] = None,
                                file_path: Optional[str]=None) -> None:
        """Initialize conversation state"""
        if file_path:
            self.file_path = file_path  
            self.load_conversation(file_path)
                
        else:
            if not question:
                raise ValueError("Question is required.")
            
            if context:
                self.context = context
            else:
                self.context = "Answer question to the point and don't hallucinate."
            self.message_list = [
                SystemMessage(content=context),
                HumanMessage(content=question)
            ]
            
            res = self._ask_question(self.message_list)
            print(res)
            self.message_list.append(AIMessage(content=res))

    def _ask_question(self,messages: List[Union[SystemMessage, HumanMessage, AIMessage]], 
                     get_content_only: bool = True) -> str:
        """
        Ask a question and get response
        Args:
            messages: List of messages
            get_content_only: Whether to return only content
        Returns:
            str: Response from the model
        """
        res = self.chatbot.invoke_query(messages)
        if get_content_only:
            try:
                return res.content
            except Exception:
                return res
        return res

    def add_message(self, message: str) -> str:
        """
        Add a message to the conversation
        Args:
            message (str): Message to add
        Returns:
            str: Response from the chatbot
        """
        self.message_list.append(HumanMessage(content=message))
        res = self._ask_question(self.message_list)
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

class IPythonStreamHandler(StreamingStdOutCallbackHandler):
    """Handler for IPython display"""
    
    def __init__(self):
        self.output = ""
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Handle new token"""
        self.output += token
        display(HTML(self.output), clear=True)
