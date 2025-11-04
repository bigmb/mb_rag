## Langsmith agent integration

import os

__all__ = ["set_langsmith_parameters"]

def set_langsmith_parameters(langsmith_api_key: str=None,
                            langsmith_endpoint: str='https://api.smith.langchain.com',
                            langsmith_project: str='No Project',
                            langsmith_tracing: str='true') -> None:
    """
    Set up environment variables for Langsmith agent integration.

    Args:
        langsmith_api_key: API key for Langsmith
        langsmith_endpoint: Endpoint URL for Langsmith
        langsmith_project: Project name in Langsmith
        langsmith_tracing: Enable or disable tracing

    Returns:
        None
    """
    if langsmith_api_key:
        os.environ['LANGSMITH_API_KEY'] = langsmith_api_key
    if langsmith_endpoint:
        os.environ['LANGSMITH_ENDPOINT'] = langsmith_endpoint
    if langsmith_project:
        os.environ['LANGSMITH_PROJECT'] = langsmith_project
    if langsmith_tracing:
        os.environ['LANGSMITH_TRACING'] = str(langsmith_tracing)