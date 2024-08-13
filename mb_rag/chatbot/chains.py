## file for chaining functions in chatbot

from langchain.schema.output_parser import StrOutputParser
from mb_rag.chatbot.prompts import invoke_prompt
from langchain.schema.runnable import RunnableLambda, RunnableSequence

class chain:
    """
    Class to chain functions in chatbot
    """
    def __init__(self,model,prompt: str = None,prompt_template: str = None, input_dict: dict = None, **kwargs):
        self.model = model
        self.output_parser = StrOutputParser() ## self.output_parser = RunnableLambda(lambda x: x.content) - can use this also
        if input_dict is not None:
            self.input_dict = input_dict
        if prompt_template is not None: 
            self.prompt = invoke_prompt(prompt_template, self.input_dict)
        else:
            self.prompt = prompt


    def invoke(self):
        """
        Invoke the chain
        Returns:
            str: Output from the chain
        """
        if self.prompt is not None:
            chain_output = self.prompt | self.model | self.output_parser
            return chain_output
        else:
            return Exception("Prompt is not provided")            
        
    def chain_seqeunce_invoke(self,middle_chain: list, final_chain: RunnableLambda= None):
        """
        Chain invoke the chain
        Args:
            middle_chain (list): List of functions/Prompts/RunnableLambda to chain
            final_chain (RunnableLambda): Final chain to run. Default is self.output_parser
        Returns:
            str: Output from the chain
        """
        if final_chain is not None:
            self.final_chain = final_chain
        else:
            self.final_chain = self.output_parser
        if self.prompt is not None:
            if middle_chain is not None:
                assert isinstance(middle_chain, list), "middle_chain should be a list"
                func_chain = RunnableSequence(self.prompt, middle_chain, self.final_chain)  
                return func_chain.invoke()
        else:
            return Exception("Prompt is not provided")
    
    