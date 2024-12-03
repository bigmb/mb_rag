import os
import importlib.util
from mb_rag.chatbot.basic import get_chatbot_google_generative_ai

__all__ = [ "generate_bounding_box", "add_bounding_box"]


def check_package(package_name):
    """
    Check if a package is installed
    Args:
        package_name (str): Name of the package
    Returns:
        bool: True if package is installed, False otherwise
    """
    return importlib.util.find_spec(package_name) is not None


def generate_bounding_box(image_path: str,model_name: str = "gemini-1.5-flash", prompt: str = 'Return bounding boxes of container, for each only one return [ymin, xmin, ymax, xmax]'):
    """
    Function to generate bounding boxes
    Args:
        image_path (str): Image path
        model_name (GenerativeModel): GenerativeModel object - google model (Default: )
        prompt (str): Prompt
    Returns:
        res (str): Result
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If model is None or invalid
    """
    try:
        model = get_chatbot_google_generative_ai(model_name)
    except Exception as e:
        raise ValueError(f"Error initializing model: {str(e)}")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        if not check_package('PIL'):
            raise ImportError("PIL package not found. Please install it using: pip install pillow")    
        from PIL import Image     

        image = Image.open(image_path)
        res = model.generate_content([image, prompt])
        return res
    except Exception as e:
        raise ValueError(f"Error generating bounding boxes: {str(e)}")

def add_bounding_box(image_path: str, bounding_box: list, show: bool = False):
    """
    Function to add bounding box to image
    Args:
        image_path (str): Image path
        bounding_box (list): Bounding boxes
        show (bool): Whether to display the image
    Returns:
        image (Image): Image with bounding box
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If bounding box format is invalid
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    if not isinstance(bounding_box, list):
        raise ValueError("bounding_box must be a list of bounding box coordinates with keys as labels and values as [ymin, xmin, ymax, xmax]")
    
    try:
        if not check_package('cv2'):
            raise ImportError("cv2 package not found. Please install it using: pip install opencv-python")    
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        

        for box in bounding_box:
            if not isinstance(box, dict):
                raise ValueError("bounding_box must be a list of bounding box coordinates with keys as labels and values as [ymin, xmin, ymax, xmax]")
            for key, value in box.items():
                if not isinstance(value, list) or len(value) != 4:
                    raise ValueError(f"Invalid bounding box format for key {key}. Expected [ymin, xmin, ymax, xmax]")
            
                cv2.rectangle(img, (value[1], value[0]), (value[3], value[2]), (0, 0, 255), 4)
                cv2.putText(img, key, (value[1], value[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if show:
            cv2.imshow("Image", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return img
    except Exception as e:
        raise ValueError(f"Error adding bounding box to image: {str(e)}")
