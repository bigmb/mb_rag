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

            "map_function": "*map(lambda x: image_url, baseframes_list)"
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
