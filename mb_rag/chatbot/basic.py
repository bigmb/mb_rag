from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

__all__ = ["load_env", "add_os_key", "get_chatbot_openai", "ask_question"]

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
        **kwargs: Additional arguments (temperature, max_tokens, timeout, max_retries, apt_key etc.)
    Returns:
        ChatOpenAI: Chatbot model
    """
    kwargs["model_name"] = model_name
    return ChatOpenAI(**kwargs)

def ask_question(chatbot, question: str):
    """
    Ask a question to the chatbot
    Args:
        chatbot (ChatOpenAI): Chatbot model
        question (str): Question to ask
    Returns:
        str: Answer from the chatbot
    """
    return chatbot.ask(question)

