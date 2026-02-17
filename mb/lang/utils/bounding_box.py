"""
Bounding box utilities
"""

import os
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from mb_rag.utils.extra import check_package

__all__ = ['BoundingBoxConfig', 'BoundingBoxProcessor']

def check_image_dependencies() -> None:
    """
    Check if required image processing packages are installed
    Raises:
        ImportError: If any required package is missing
    """
    if not check_package("PIL"):
        raise ImportError("Pillow package not found. Please install it using: pip install Pillow")
    if not check_package("cv2"):
        raise ImportError("OpenCV package not found. Please install it using: pip install opencv-python")
    if not check_package("google.generativeai"):
        raise ImportError("Google Generative AI package not found. Please install it using: pip install google-generativeai")

@dataclass
class BoundingBoxConfig:
    """Configuration for bounding box operations"""
    model_name: str = "gemini-1.5-pro-latest"
    api_key: Optional[str] = None
    default_prompt: str = 'Return bounding boxes of container, for each only one return [ymin, xmin, ymax, xmax]'

class BoundingBoxProcessor:
    """
    Class for processing images and generating bounding boxes
    
    Attributes:
        model: The Google Generative AI model instance
        config: Configuration for bounding box operations
    """

    def __init__(self, config: Optional[BoundingBoxConfig] = None, **kwargs):
        """
        Initialize bounding box processor
        Args:
            config: Configuration for the processor
            **kwargs: Additional arguments
        """
        check_image_dependencies()
        self.config = config or BoundingBoxConfig(**kwargs)
        self._initialize_model()
        self._initialize_image_libs()

    @classmethod
    def from_model(cls, model_name: str, api_key: Optional[str] = None, **kwargs) -> 'BoundingBoxProcessor':
        """
        Create processor with specific model configuration
        Args:
            model_name: Name of the model
            api_key: Optional API key
            **kwargs: Additional configuration
        Returns:
            BoundingBoxProcessor: Configured processor
        """
        config = BoundingBoxConfig(
            model_name=model_name,
            api_key=api_key
        )
        return cls(config, **kwargs)

    def _initialize_model(self) -> None:
        """Initialize the AI model"""
        import google.generativeai as genai
        
        api_key = self.config.api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key not found. Please provide api_key parameter or set GOOGLE_API_KEY environment variable.")
        
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name=self.config.model_name)
        except Exception as e:
            raise ValueError(f"Error initializing Google Generative AI model: {str(e)}")

    def _initialize_image_libs(self) -> None:
        """Initialize image processing libraries"""
        from PIL import Image
        import cv2
        self._Image = Image
        self._cv2 = cv2

    @staticmethod
    def _validate_image_path(image_path: str) -> None:
        """
        Validate image path
        Args:
            image_path: Path to image
        Raises:
            FileNotFoundError: If image doesn't exist
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

    def generate_bounding_boxes(self, 
                              image_path: str, 
                              prompt: Optional[str] = None
                              ) -> Any:
        """
        Generate bounding boxes for an image
        Args:
            image_path: Path to image
            prompt: Custom prompt for the model
        Returns:
            Any: Model response with bounding boxes
        """
        self._validate_image_path(image_path)
        
        try:
            image = self._Image.open(image_path)
            prompt = prompt or self.config.default_prompt
            return self.model.generate_content([image, prompt])
        except Exception as e:
            raise ValueError(f"Error generating bounding boxes: {str(e)}")

    def add_bounding_boxes(self, 
                          image_path: str,
                          bounding_boxes: Dict[str, List[int]],
                          color: Tuple[int, int, int] = (0, 0, 255),
                          thickness: int = 4,
                          font_scale: float = 1.0,
                          show: bool = False,
                          google_bb= False
                          ) -> Any:
        """
        Add bounding boxes to an image
        Args:
            image_path: Path to image
            bounding_boxes: Dictionary of bounding boxes
            color: BGR color tuple
            thickness: Line thickness
            font_scale: Font scale for labels
            show: Whether to display the image
        Returns:
            Any: Image with bounding boxes
        """
        self._validate_image_path(image_path)
        
        if not isinstance(bounding_boxes, dict):
            raise ValueError("bounding_boxes must be a dictionary")
        
        try:
            img = self._cv2.imread(image_path)                
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            for key, value in bounding_boxes.items():
                if not isinstance(value, list) or len(value) != 4:
                    raise ValueError(f"Invalid bounding box format for key {key}. Expected [ymin, xmin, ymax, xmax]")
                
                if google_bb:
                    value = [int(value[0] * img.shape[0] * 0.001), int(value[1] * img.shape[1]* 0.001),
                              int(value[2] * img.shape[0] * 0.001), int(value[3] * img.shape[1] * 0.001)]
                    print("Orignal Bounding Box from GOOGLE BBOX: ", value)

                self._cv2.rectangle(
                    img=img,
                    pt1=(value[1], value[0]),  # xmin, ymin
                    pt2=(value[3], value[2]),  # xmax, ymax
                    color=color,
                    thickness=thickness
                )
                self._cv2.putText(
                    img=img,
                    text=key,
                    org=(value[1], value[0]),
                    fontFace=self._cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_scale,
                    color=color,
                    thickness=thickness//2
                )
            
            if show:
                self._display_image(img)
            
            return img
        except Exception as e:
            raise ValueError(f"Error adding bounding boxes to image: {str(e)}")

    def _display_image(self, img: Any) -> None:
        """
        Display an image
        Args:
            img: Image to display
        """
        self._cv2.imshow("Image", img)
        self._cv2.waitKey(0)
        self._cv2.destroyAllWindows()

    def save_image(self, img: Any, output_path: str) -> None:
        """
        Save an image
        Args:
            img: Image to save
            output_path: Path to save the image
        """
        try:
            self._cv2.imwrite(output_path, img)
        except Exception as e:
            raise ValueError(f"Error saving image: {str(e)}")

    def process_image(self, 
                     image_path: str,
                     output_path: Optional[str] = None,
                     show: bool = False,
                     **kwargs) -> Any:
        """
        Complete image processing pipeline
        Args:
            image_path: Path to input image
            output_path: Optional path to save output
            show: Whether to display the result
            **kwargs: Additional arguments for bounding box generation
        Returns:
            Any: Processed image
        """
        boxes = self.generate_bounding_boxes(image_path, **kwargs)
        img = self.add_bounding_boxes(image_path, boxes, show=show)
        
        if output_path:
            self.save_image(img, output_path)
        
        return img
