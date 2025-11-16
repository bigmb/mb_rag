from langchain_core.prompts.chat import ChatPromptTemplate

__all__ = ["PromptManager"]

class PromptManager:
    """
    Central class for storing and invoking prompt templates.

    Example:
        pm = PromptManager()
        prompt_text = pm.render_prompt("greeting")
        print(prompt_text)

        pm = PromptManager()
        prompt_text = pm.render_prompt("todo_task", {"task": "Plan a deep learning project for image recognition"})
        print(prompt_text)
    """

    def __init__(self):
        self.templates = {
            "coding_python": """You are a Python developer.
Human: {question}
Assistant:""",

            "greeting": """You are a friendly assistant.
Human: Hello!
Assistant: Hi there! How can I assist you today?""",

            "goodbye": """You are a friendly assistant.
Human: Goodbye!
Assistant: Goodbye! Have a great day!""",

            "todo_task": """You are a helpful assistant.
Human: Please create a to-do list for the following task: {task}
Assistant:""",

            "map_function": "*map(lambda x: image_url, baseframes_list)",
            
            "SQL_AGENT_SYS_PROMPT": """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct postgresql query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most 5 results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

RULES:
- THINK step by step before answering.
- Use the provided database schema to inform your queries.
- When you need to retrieve data, generate a SQL query and execute it using the provided tools.
- Read-only mode: Do not attempt to modify the database.
- NO INSERT/UPDATE/DELETE/ALTER/DROP/CREATE/REPLACE/TRUNCATE statements allowed.
- LIMIT your results to 10 rows. Unless specified otherwise.
- If you encounter an error while executing a query, analyze the error message and adjust your query accordingly.
- Prefer using explicit column names instead of SELECT * for better performance.    
- Always ensure your SQL syntax is correct. """
        }

    def get_template(self, name: str) -> str:
        """
        Get a prompt template by name.
        Args:
            name (str): The key name of the prompt.
        Returns:
            str: The prompt template string.
        """
        template = self.templates.get(name)
        if not template:
            raise ValueError(f"Prompt '{name}' not found. Available prompts: {list(self.templates.keys())}")
        return template

    def render_prompt(self, name: str, context: dict = None) -> str:
        """
        Fill and return a rendered prompt string.
        Args:
            name (str): The key name of the prompt.
            context (dict): Variables to fill into the template.
        Returns:
            str: The final rendered prompt text.
        """
        template = self.get_template(name)
        chat_prompt = ChatPromptTemplate.from_template(template)
        rendered = chat_prompt.invoke(context or {})
        return rendered.to_string()
