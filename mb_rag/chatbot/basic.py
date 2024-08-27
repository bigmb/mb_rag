from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import os
import boto3
from langchain_community.llms import Ollama


__all__ = ["load_env", "add_os_key", "get_chatbot_openai", "ask_question", "conversation_model", "get_chatbot_anthropic", "get_chatbot_google_generative_ai", "get_client", "get_chatbot_ollama"]

def load_env(file_path: str):
    """
    Load environment variables from a file
    Args:
        file_path (str): Path to the environment file
    Returns:
        None
    """
    load_dotenv(file_path)
    
def add_os_key(name: str ,key: str):
    """
    Add an API key to the environment
    Args:
        name (str): Name of the API key
        key (str): API key
    Returns:
        None
    """
    os.environ[name] = key

def get_chatbot_openai(model_name: str = "gpt-4o",**kwargs):
    """
    Load the chatbot model from OpenAI
    Args:
        model_name (str): Name of the model
        **kwargs: Additional arguments (temperature, max_tokens, timeout, max_retries, api_key etc.)
    Returns:
        ChatOpenAI: Chatbot model
    """
    kwargs["model_name"] = model_name
    return ChatOpenAI(**kwargs)

def get_chatbot_anthropic(model_name: str = "claude-3-opus-20240229",**kwargs):
    """
    Load the chatbot model from Anthropic
    Args:
        model_name (str): Name of the model
        **kwargs: Additional arguments (temperature, max_tokens, timeout, max_retries, api_key etc.)
    Returns:
        ChatAnthropic: Chatbot model
    """
    kwargs["model_name"] = model_name
    return ChatAnthropic(**kwargs)

def get_chatbot_google_generative_ai(model_name: str = "gemini-1.5-flash",**kwargs):
    """
    Load the chatbot model from Google Generative AI
    Args:
        model_name (str): Name of the model
        **kwargs: Additional arguments (temperature, max_tokens, timeout, max_retries, api_key etc.)
    Returns:
        ChatGoogleGenerativeAI: Chatbot model
    """
    kwargs["model_name"] = model_name
    return ChatGoogleGenerativeAI(**kwargs)

def get_chatbot_ollama(model_name: str = "llama3",**kwargs):
    """
    Load the chatbot model from Ollama
    Args:
        model_name (str): Name of the model
        **kwargs: Additional arguments (temperature, max_tokens, timeout, max_retries, api_key etc.)
    Returns:
        ChatOllama: Chatbot model
    """
    kwargs["model"] = model_name
    return Ollama(**kwargs)

def ask_question(chatbot, question: str, get_content_only: bool = True):
    """
    Ask a question to the chatbot
    Args:
        chatbot (ChatOpenAI): Chatbot model
        question (str): Question to ask
    Returns:
        str: Answer from the chatbot
    """
    res =chatbot.invoke(question)
    if get_content_only:
        return res.content
    return res

def get_client():
    """
    Returns a boto3 client for S3
    """
    return boto3.client('s3')

class conversation_model:
    """
    A class to represent a conversation model
    Attributes:
        chatbot (ChatOpenAI): Chatbot model
        context (str): Context of the conversation
        question (str): Question to ask
        message_list (list): List of messages in the conversation
        file_path (str): Path to the conversation file (if s3_path then add s3_path='loc' and client and bucket)
    """
    def __init__(self, model_name: str = "gpt-4o",model_type: str = 'openai' ,file_path : str = None,context: str = None, question :str = None,**kwargs):
        if model_type == 'openai':
            self.chatbot = get_chatbot_openai(model_name, **kwargs)
        elif model_type == 'anthropic':
            self.chatbot = get_chatbot_anthropic(model_name, **kwargs)
        elif model_type == 'google':
            self.chatbot = get_chatbot_google_generative_ai(model_name, **kwargs)
        elif model_type == 'ollama':
            self.chatbot = get_chatbot_ollama(model_name, **kwargs)
        else:
            raise ValueError(f"Model type {model_type} is not supported")
                    
        try:
            self.s3_path = kwargs['s3_path']
            print(self.s3_path)
            if self.s3_path is not None:
                self.client = kwargs['client']
                self.bucket = kwargs['bucket']
        except Exception:
            self.s3_path = None

        if file_path is not None:
            self.file_path = file_path
            self.load_conversation(file_path)
        else:
            if context is not None:
                self.context = context
            else:
                raise ValueError("Context/Title is required. Please provide context or previous conversation file_path")
        
            if question is not None:
                self.question = question
            else:
                raise ValueError("Question is required.")

            self.message_list = [SystemMessage(content=self.context), HumanMessage(content=self.question)]
        res = ask_question(self.chatbot, self.message_list , get_content_only=True)
        print(res)
        self.message_list.append(AIMessage(content=res))

    def add_message(self, message: str):
        """
        Add a message to the conversation
        Args:
            message (str): Message to add
        Returns:
            str: Answer from the chatbot. Adds the message to the conversation also.
        """
        self.message_list.append(HumanMessage(content=message))
        res = ask_question(self.chatbot, self.message_list , get_content_only=True)
        # print(res)
        self.message_list.append(AIMessage(content=res))
        return res
    
    def get_all_messages(self):
        """
        Get all messages from the conversation
        Returns:
            list: List of messages
        """
        return self.message_list
    
    def get_last_message(self):
        """
        Get the last message from the conversation
        Returns:
            str: Last message
        """
        return self.message_list[-1].content
    
    def get_all_messages_content(self):
        """
        Get all messages from the conversation
        Returns:
            list: List of messages
        """
        return [message.content for message in self.message_list]

    def save_conversation(self, file_path: str = None,**kwargs):
        """
        Save the conversation to a file or s3
        Args:
            file_path (str): Path to the file. if none then it will save to the file_path given in the constructor
            **kwargs: Additional arguments (client, bucket)
        Returns:
            bool: True if saved successfully
        """
        print(f"s3 path given : {self.s3_path}")
        if self.s3_path is not None:
            try:
                client = kwargs['client']
                bucket = kwargs['bucket']
                client.put_object(Body=str(self.message_list),Bucket=bucket,Key=self.s3_path)
            except Exception as e:
                print("Check the s3_path, client and bucket")
                raise ValueError(f"Error saving conversation to s3: {e}")
            print(f"Conversation saved to s3_path: {self.s3_path}")
        else:    
            try:
                if file_path is None:
                    file_path = self.file_path
                with open(file_path, 'w') as f:
                    for message in self.message_list:
                        f.write(f"{message.content}\n")
                print(f"Conversation saved to file as s3_path not given: {file_path}")
            except Exception as e:
                raise ValueError(f"Error saving conversation to file: {e}")
        return True
    
    def load_conversation(self, file_path: str = None, **kwargs):
        """
        Load the conversation from a file or s3
        Args:
            file_path (str): Path to the file
            **kwargs: Additional arguments (client, bucket)
        Returns:
            list: List of messages
        """
        self.message_list = []
        if self.s3_path is not None:
            client = kwargs['client']
            bucket = kwargs['bucket']
            res = client.get_response(client,bucket,self.s3_path)
            res_str = eval(res['Body'].read().decode('utf-8'))
            self.message_list = [SystemMessage(content=res_str)]
            print(f"Conversation loaded from s3_path: {self.s3_path}")
        else:
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    self.message_list.append(SystemMessage(content=line))
                print(f"Conversation loaded from file: {file_path}")
            except Exception as e:
                raise ValueError(f"Error loading conversation from file: {e}")
        return self.message_list
