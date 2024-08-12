## file for chaining functions in chatbot

from langchain.schema.output_parser import StrOutputParser
from mb_rag.chatbot.prompts import invoke_prompt

class chain:
    """
    Class to chain functions in chatbot
    """
    def __init__(self,prompt_template: str = None, input_dict: dict = None, **kwargs):
        self.output_parser = StrOutputParser()
        if prompt_template is not None:
            self.prompt = invoke_prompt(prompt_template, input_dict)