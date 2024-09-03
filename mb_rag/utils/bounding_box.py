from PIL import Image
import google.generativeai as genai
import os

__all__ = ["google_model","generate_bounding_box"]

# genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
# model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")

def google_model(model="gemini-1.5-pro-latest",api_key=None,**kwargs):
    """
    Function to get google genai model 
    Args:
        model (str): Model name
        api_key (str): API key. If None, it will be fetched from the environment variable GOOGLE_API_KEY
        **kwargs: Additional arguments
    Returns:
        model (GenerativeModel): GenerativeModel object
    """
    if api_key == None:
        api_key = os.environ.get("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name=model,**kwargs)

def generate_bounding_box(model,image_path: str,prompt: str= 'Return bounding boxes of container, for each one return [ymin, xmin, ymax, xmax]'):
    """
    Function to generate bounding boxes
    Args:
        model (GenerativeModel): GenerativeModel object
        image_path (str): Image path
        prompt (str): Prompt
    Returns:
        res (str): Result
    """
    image = Image.open(image_path)
    res = model.generate_content([image,prompt])
    print(res.text)
    return res

