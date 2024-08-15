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
    
    def chain_parrellel_invoke(self,parrellel_chain: list):
        """
        Chain invoke the chain #### better to use RunnableParallel outside the class
        Args:
            parrellel_chain (list): List of functions/Prompts/RunnableLambda to chain 
        Returns:
            str: Output from the chain
        """
        return parrellel_chain.invoke()
    
    def chain_branch_invoke(self,branch_chain: dict):
        """
        Chain invoke the chain #### better to use RunnableBranch outside the class
        Args:
            branch_chain (dict): Dictionary of functions/Prompts/RunnableLambda to chain 
        Returns:
            str: Output from the chain
        """
        return branch_chain.invoke()
    
## Example of parrellel chain and branch chain

# from langchain.schema.runnable import RunnableParallel, RunnableBranch
# from langchain.schema.output_parser import StrOutputParser
# from mb_rag.chatbot.prompts import invoke_prompt
# from langchain.schema.runnable import RunnableLambda, RunnableSequence


### Example of parrellel chain
# # Define prompt template
# prompt_template = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are an expert product reviewer."),
#         ("human", "List the main features of the product {product_name}."),
#     ]
# )


# # Define pros analysis step
# def analyze_pros(features):
#     pros_template = ChatPromptTemplate.from_messages(
#         [
#             ("system", "You are an expert product reviewer."),
#             (
#                 "human",
#                 "Given these features: {features}, list the pros of these features.",
#             ),
#         ]
#     )
#     return pros_template.format_prompt(features=features)


# # Define cons analysis step
# def analyze_cons(features):
#     cons_template = ChatPromptTemplate.from_messages(
#         [
#             ("system", "You are an expert product reviewer."),
#             (
#                 "human",
#                 "Given these features: {features}, list the cons of these features.",
#             ),
#         ]
#     )
#     return cons_template.format_prompt(features=features)


# # Combine pros and cons into a final review
# def combine_pros_cons(pros, cons):
#     return f"Pros:\n{pros}\n\nCons:\n{cons}"


# # Simplify branches with LCEL
# pros_branch_chain = (
#     RunnableLambda(lambda x: analyze_pros(x)) | model | StrOutputParser()
# )

# cons_branch_chain = (
#     RunnableLambda(lambda x: analyze_cons(x)) | model | StrOutputParser()
# )

# # Create the combined chain using LangChain Expression Language (LCEL)
# chain = (
#     prompt_template
#     | model
#     | StrOutputParser()
#     | RunnableParallel(branches={"pros": pros_branch_chain, "cons": cons_branch_chain})
#     | RunnableLambda(lambda x: combine_pros_cons(x["branches"]["pros"], x["branches"]["cons"]))
# )

# # Run the chain
# result = chain.invoke({"product_name": "MacBook Pro"})

# # Output
# print(result)


### Example of branch chain
# positive_feedback_template = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a helpful assistant."),
#         ("human",
#          "Generate a thank you note for this positive feedback: {feedback}."),
#     ]
# )

# negative_feedback_template = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a helpful assistant."),
#         ("human",
#          "Generate a response addressing this negative feedback: {feedback}."),
#     ]
# )

# neutral_feedback_template = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a helpful assistant."),
#         (
#             "human",
#             "Generate a request for more details for this neutral feedback: {feedback}.",
#         ),
#     ]
# )

# escalate_feedback_template = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a helpful assistant."),
#         (
#             "human",
#             "Generate a message to escalate this feedback to a human agent: {feedback}.",
#         ),
#     ]
# )

# # Define the feedback classification template
# classification_template = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a helpful assistant."),
#         ("human",
#          "Classify the sentiment of this feedback as positive, negative, neutral, or escalate: {feedback}."),
#     ]
# )

# # Define the runnable branches for handling feedback
# branches = RunnableBranch(
#     (
#         lambda x: "positive" in x,
#         positive_feedback_template | model | StrOutputParser()  # Positive feedback chain
#     ),
#     (
#         lambda x: "negative" in x,
#         negative_feedback_template | model | StrOutputParser()  # Negative feedback chain
#     ),
#     (
#         lambda x: "neutral" in x,
#         neutral_feedback_template | model | StrOutputParser()  # Neutral feedback chain
#     ),
#     escalate_feedback_template | model | StrOutputParser()
# )

# # Create the classification chain
# classification_chain = classification_template | model | StrOutputParser()

# # Combine classification and response generation into one chain
# chain = classification_chain | branches

# # Run the chain with an example review
# # Good review - "The product is excellent. I really enjoyed using it and found it very helpful."
# # Bad review - "The product is terrible. It broke after just one use and the quality is very poor."
# # Neutral review - "The product is okay. It works as expected but nothing exceptional."
# # Default - "I'm not sure about the product yet. Can you tell me more about its features and benefits?"

# review = "The product is terrible. It broke after just one use and the quality is very poor."
# result = chain.invoke({"feedback": review})

# # Output the result
# print(result)