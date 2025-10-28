## simple llm wrapper to replace invoke with invoke_query/own model query

__all__ = ["LLMWrapper"]

class LLMWrapper:
    """A simple wrapper for the language model to standardize the invoke method.
    """

    def __init__(self, llm):
        self.llm = llm

    def invoke_query(self, prompt: str) -> str:
        """
        Invoke the language model with the given prompt.
        
        Args:
            prompt (str): The prompt to send to the language model.
        
        Returns:
            str: The generated response.
        """
        return self.llm.invoke(prompt)