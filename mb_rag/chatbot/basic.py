from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import os

__all__ = ["load_env", "add_os_key", "get_chatbot_openai", "ask_question", "conversation_model"]

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

class conversation_model:
    """
    A class to represent a conversation model
    Attributes:
        chatbot (ChatOpenAI): Chatbot model
        context (str): Context of the conversation
        question (str): Question to ask
        message_list (list): List of messages in the conversation
    """
    def __init__(self, model_name: str = "gpt-4o",model_type: str = 'openai' ,context: str = None, question :str = None,**kwargs):
        if model_type == 'openai':
            self.chatbot = get_chatbot_openai(model_name, **kwargs)
        elif model_type == 'anthropic':
            self.chatbot = get_chatbot_anthropic(model_name, **kwargs)
        elif model_type == 'google':
            self.chatbot = get_chatbot_google_generative_ai(model_name, **kwargs)
        else:
            raise ValueError(f"Model type {model_type} is not supported")
        
        if context is not None:
            self.context = context
        else:
            raise ValueError("Context is required")
        
        if question is not None:
            self.question = question
        else:
            raise ValueError("Question is required")
        
        self.message_list = [SystemMessage(content=self.context), HumanMessage(content=self.question)]
        res = ask_question(self.chatbot, self.message_list , get_content_only=True)
        print(res)
        self.message_list.append(AIMessage(content=res))

    def add_message(self, message: str):
        self.message_list.append(HumanMessage(content=message))
        res = ask_question(self.chatbot, self.message_list , get_content_only=True)
        print(res)
        self.message_list.append(AIMessage(content=res))
        return res
    
    def get_all_messages(self):
        return self.message_list
    
    def get_last_message(self):
        return self.message_list[-1].content
    
    def get_all_messages_content(self):
        return [message.content for message in self.message_list]

    def save_conversation(self, file_path: str):
        with open(file_path, 'w') as f:
            for message in self.message_list:
                f.write(f"{message.content}\n")
        return True
    
    def load_conversation(self, file_path: str):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        self.message_list = []
        for line in lines:
            self.message_list.append(SystemMessage(content=line))
        return True
