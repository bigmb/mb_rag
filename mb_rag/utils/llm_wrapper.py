## simple llm wrapper to replace invoke with invoke_query/own model query

__all__ = ["LLMWrapper"]

class LLMWrapper:
    """A simple wrapper for the language model to standardize the invoke method.
    """

    def __init__(self, llm):
        self.llm = llm

    def __getattr__(self, name):
        """Get all attributes from llm module. (invoke_query, invoke_query_threads, etc.)"""
        return getattr(self.llm, name)

    def invoke(self, use_threads=False,**kwargs) -> str:
        """
        Invoke the language model with a list of messages.
        Using invoke_query method of the underlying model.
        Check ModelFactory for more details.
        
        Args:
            use_threads (bool): Whether to use threading for invocation. Defaults to False.
            **kwargs: Keyword arguments for the model invocation.

        Returns:
            str: The generated response.
        """
        if use_threads:
            return self.llm.invoke_query_threads(**kwargs)
        return self.llm.invoke_query(**kwargs)