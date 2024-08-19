## file for storing basic prompts template
from langchain.prompts import ChatPromptTemplate

__all__ = ["prompts", "invoke_prompt"]

class prompts:
    """
    Class to get different prompts example for chatbot and templates
    """

    def get_code_prompts(self):
        """
        Get code prompts
        Returns:
            str: Code prompt
        """
        list_code_prompts = {'coding_python ': """You are a Python developer.
                                                Human: {}"""}

    def get_text_prompts(self):
        """
        Get text prompts
        Returns:
            str: Text prompt
        """
        list_text_prompts = {
            'multiple_placeholders': """You are a helpful assistant.
                                        Human: Tell me a more about {adjective1} and its relation to {adjective2}.
                                        Assistant:"""
        }

    def get_image_prompts(self):
        """
        Get image prompts
        Returns:
            str: Image prompt
        """
        list_image_prompts = {'map_function': "*map(lambda x: image_url, baseframes_list)"} # for passing multiple images from a video or a list of images

    def get_assistant_prompts(self):
        """
        Get assistant prompts
        Returns:
            str: Assistant prompt
        """
        list_assistant_prompts = {}

def invoke_prompt(template: str, input_dict : dict = None):
    """
    Invoke a prompt
    Args:
        template (str): Template for the prompt
        input_dict (dict): Input dictionary for the prompt
    Returns:
        str: Prompt
    """
    prompt_multiple = ChatPromptTemplate.from_template(template)
    prompt = prompt_multiple.invoke(input_dict)
    return prompt