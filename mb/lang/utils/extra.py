## extra functions for RAG's

import os
from dotenv import load_dotenv
import importlib.util
from typing import Optional,List,Dict,Union,Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt


__all__ = ["load_env_file", "check_package", "pdf_to_text", "convert_pdfs_in_folder","SAM2Processor","ImagePredictor"]

def load_env_file(file_path='.env'):
    """
    Load environment variables from a .env file.

    Args:
        file_path (str): Path to the .env file. Defaults to '.env'.

    Returns:
        None
    """
    load_dotenv(file_path)

    # Get the loaded environment variables
    env_vars = os.environ

    return env_vars

def check_package(package_name, error_message=None):
    """
    Check if a package is installed
    Args:
        package_name (str): Name of the package
        error_message (str, optional): Custom error message to display if the package is not found
    Returns:
        bool: True if package is installed, False otherwise
    """

    if importlib.util.find_spec(package_name) is not None:
        return True
    else:
        if error_message:
            print(error_message)
        else:
            print(f"Package '{package_name}' not found.")
    return False

def pdf_to_text(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    try:
        if not check_package("PyPDF2"):
            raise ImportError("PyPDF2 package not found. Please install it using: pip install pypdf2")
        import PyPDF2
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except PyPDF2.errors.PdfReadError as e:
        print(f"Error reading {pdf_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred with {pdf_path}: {e}")
    return text

def convert_pdfs_in_folder(folder_path):
    """
    Convert all PDF files in the given folder to text files.
    Args:
        folder_path (str): Path to the folder containing the PDF files.
    Returns:
        None
    Example : convert_pdfs_in_folder('/folder_path') # folder_path is the path to the folder containing the PDF files.
    The converted PDF files and text files will be created in the same folder
    """
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            text = pdf_to_text(pdf_path)
            if text:  # Only write to file if text is not empty
                text_filename = os.path.splitext(filename)[0] + '.txt'
                text_path = os.path.join(folder_path, text_filename)
                with open(text_path, 'w', encoding='utf-8') as text_file:
                    text_file.write(text)
                print(f"Converted: {filename} to {text_filename}")
            else:
                print(f"No text extracted from {filename}")


class SAM2Processor:
    """
    Main class for SAM2 model operations including mask generation and visualization.
    """
    
    def __init__(self, sam2_checkpoint: str = '../checkpoints/sam2_hiera_large.pt',
                 model_cfg: str = 'sam2_hiera_l.yaml', device: str = 'cpu'):
        """
        Initialize SAM2Processor.

        Args:
            sam2_checkpoint (str): Path to model checkpoint
            model_cfg (str): Path to model configuration
            device (str): Device to run on
        """
        self.device = device
        self.sam2_checkpoint = sam2_checkpoint
        self.model_cfg = model_cfg
        self.mask_generator = self._initialize_mask_generator()


    def _initialize_mask_generator(self):
        """Initialize the mask generator."""
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        sam2 = build_sam2(self.model_cfg, self.sam2_checkpoint, 
                         device=self.device, apply_postprocessing=False)
        return SAM2AutomaticMaskGenerator(sam2)

    def show_anns(self, anns: List[Dict], borders: bool = True, show: bool = True) -> Optional[np.ndarray]:
        """Display annotations on an image."""
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)
     
        img = np.ones((sorted_anns[0]['segmentation'].shape[0], 
                      sorted_anns[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.5]])
            img[m] = color_mask 
            if borders:
                contours, _ = cv2.findContours(m.astype(np.uint8),
                                             cv2.RETR_EXTERNAL, 
                                             cv2.CHAIN_APPROX_NONE)
                contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) 
                          for contour in contours]
                cv2.drawContours(img, contours, -1, (0,0,1,0.4), thickness=1)
        if show:    
            ax.imshow(img)
        return img

    def get_similarity_value(self, box1: List[float], box2: List[float]) -> float:
        """Calculate similarity between two bounding boxes."""
        val1 = abs(box1[0]-box2[0])
        val2 = abs(box1[1]-box2[1])
        val3 = abs(box1[2]-box2[2])
        val4 = abs(box1[3]-box2[3])
        return val1 + val2 + val3 + val4

    def get_final_similar_box(self, box1: List[float], box2: List[List[float]]) -> Tuple[List[float], int]:
        """Find the most similar box from a list of boxes."""
        best_box = None
        best_val = None
        index = None
        for i in box2:
            val = self.get_similarity_value(box1, i)
            if best_box is None or val < best_val:
                best_box = i
                best_val = val
                index = box2.index(i)
        return best_box, index

    def get_mask_for_bbox(self, image_path: str, bbox_value: List[float],
                         show_full: bool = False, show_final: bool = False) -> Tuple[np.ndarray, List[float], List[List[float]]]:
        """Get mask for a specific bounding box."""
        print('Getting mask')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_full = self.mask_generator.generate(image)
        if show_full:
            self.show_anns(mask_full)
        print('Getting final mask')

        main_bbox = []
        for i in mask_full:
            mask_val = [i['bbox'][1], i['bbox'][0],
                       (i['bbox'][3]+i['bbox'][1]), (i['bbox'][2]+i['bbox'][0])]
            main_bbox.append(mask_val)

        value_list, index = self.get_final_similar_box(bbox_value, main_bbox)
        final_mask = mask_full[index]
        final_bbox = [final_mask['bbox'][1], final_mask['bbox'][0],
                     (final_mask['bbox'][3]+final_mask['bbox'][1]),
                     (final_mask['bbox'][2]+final_mask['bbox'][0])]
        if show_final:
            self.show_anns([final_mask])
        return final_mask['segmentation'], final_bbox, main_bbox

    def get_all_masks(self, image_path: str) -> List[Dict]:
        """Get all masks for an image."""
        print('Getting all masks')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_full = self.mask_generator.generate(image)
        print('Getting final mask')
        return mask_full

    @staticmethod
    def show_mask(mask: np.ndarray, ax: plt.Axes, obj_id: Optional[int] = None,
                 random_color: bool = False) -> None:
        """Display a mask on a given axis."""
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    @staticmethod
    def show_masks_image(
        image: np.ndarray,
        masks: List[np.ndarray],
        scores: List[float],
        point_coords: Optional[np.ndarray] = None,
        box_coords: Optional[Union[List, np.ndarray]] = None,
        point_labels: Optional[np.ndarray] = None,
        labels_names: Optional[List[str]] = None,
        borders: bool = True
    ) -> None:
        """Display multiple masks on an image."""

        if box_coords is not None:
            box_coords = np.array(box_coords, dtype=np.float32)

            if box_coords.ndim == 1:
                box_coords = box_coords.reshape(1, 4)

            if box_coords.shape[0] == 1 and len(masks) > 1:
                box_coords = np.repeat(box_coords, len(masks), axis=0)

            per_mask_boxes = (box_coords.shape[0] == len(masks))
        else:
            per_mask_boxes = False

        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        ax.imshow(image)

        masks = np.stack(masks)
        for i, mask in enumerate(masks):
            SAM2Processor.show_mask(mask, ax, obj_id=i)

        if point_coords is not None:
            assert point_labels is not None
            SAM2Processor.show_points(point_coords, point_labels, ax)

        if box_coords is not None:
            if per_mask_boxes:
                SAM2Processor.show_box(box_coords, ax, labels=labels_names)
            else:
                SAM2Processor.show_box(box_coords, ax, labels=labels_names)

        # if len(scores) > 1:
            # ax.set_title("Masks: " + ", ".join(str(i+1) for i in range(len(masks))), fontsize=18)

        ax.axis("off")
        plt.show()

    @staticmethod
    def show_points(coords: np.ndarray,
                    labels: np.ndarray,
                    ax: plt.Axes,
                    marker_size: int = 200) -> None:
        """
        Draw 2D points on matplotlib axis.
        
        Supports:
            coords: (P, 2) or (B, P, 2)
            labels: (P,) or (B, P)
        
        Positive label (1) = green star
        Negative label (0) = red star
        """

        coords = np.asarray(coords)
        labels = np.asarray(labels)

        if coords.ndim == 2:
            pos = coords[labels == 1]
            neg = coords[labels == 0]

            if len(pos) > 0:
                ax.scatter(pos[:, 0], pos[:, 1],
                        color='green', marker='*',
                        s=marker_size, edgecolor='white', linewidth=1.2)

            if len(neg) > 0:
                ax.scatter(neg[:, 0], neg[:, 1],
                        color='red', marker='*',
                        s=marker_size, edgecolor='white', linewidth=1.2)
            return

        if coords.ndim == 3:
            B = coords.shape[0]

            for i in range(B):
                pts = coords[i]
                lbl = labels[i]

                pos = pts[lbl == 1]
                neg = pts[lbl == 0]

                if len(pos) > 0:
                    ax.scatter(pos[:, 0], pos[:, 1],
                            color='green', marker='*',
                            s=marker_size, edgecolor='white', linewidth=1.2)

                if len(neg) > 0:
                    ax.scatter(neg[:, 0], neg[:, 1],
                            color='red', marker='*',
                            s=marker_size, edgecolor='white', linewidth=1.2)
            return

        raise ValueError(f"coords must be shape (P,2) or (B,P,2). Got {coords.shape}")

        
    @staticmethod
    def show_box(box: Union[List, np.ndarray],
                  ax: plt.Axes,
                  labels: Optional[List[str]] = None) -> None:
        """Display one or multiple bounding boxes on an axis."""
        
        box = np.array(box, dtype=np.float32)

        if box.ndim == 1:
            box = box.reshape(1, 4)

        if labels is not None:
            if len(labels) == 1 and len(box) > 1:
                labels = labels * len(box)
            elif len(labels) != len(box):
                labels = labels + [""] * (len(box) - len(labels))
        else:
            labels = [""] * len(box)

        for (x1, y1, x2, y2), label in zip(box, labels):
            w, h = x2 - x1, y2 - y1

            ax.add_patch(
                plt.Rectangle(
                    (x1, y1),
                    w,
                    h,
                    edgecolor='green',
                    facecolor=(0, 0, 0, 0),
                    lw=2
                )
            )

            if label:
                ax.text(
                    x1,
                    y1 - 5,
                    label,
                    fontsize=12,
                    color='green',
                    bbox=dict(
                        facecolor='black',
                        alpha=0.5,
                        edgecolor='none',
                        pad=2
                    )
                )

class ImagePredictor:
    """Class for image prediction using SAM2."""

    def __init__(self, model_cfg: str, sam2_checkpoint: str, device: str = 'cpu'):
        """Initialize ImagePredictor."""
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.build_sam import build_sam2

        self.predictor = SAM2ImagePredictor(build_sam2(model_cfg, sam2_checkpoint, device=device))
        self.image = None

    def set_image(self, image: Union[str, np.ndarray]) -> None:
        """Set the image for prediction."""
        if isinstance(image, str):
            image = cv2.imread(image)
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            self.image = image
        self.predictor.set_image(self.image)

    def predict_item(self, 
                    bbox: Optional[List[List[float]]] = None,
                    points: Optional[List[List[float]]] = None,
                    point_labels: Optional[List[int]] = None,
                    labels_names: Optional[List[str]] = None,
                    show: bool = True, 
                    gemini_bbox: bool = True,
                    **kwargs):

        all_masks = []
        all_scores = []
        all_logits = []
        all_boxes = []
        all_points = []
        all_labels = []

        if bbox is not None:
            bbox = np.array(bbox, dtype=np.float32)
            if gemini_bbox:
                bbox = bbox[:, [1, 0, 3, 2]]   # reorder
        else:
            bbox = []

        for i, single_box in enumerate(bbox if len(bbox) > 0 else [None]):

            predict_args = {}

            if points is not None and point_labels is not None:
                predict_args["point_coords"] = np.array(points, dtype=np.float32)
                predict_args["point_labels"] = np.array(point_labels, np.int32)
                all_points.append(np.array(points, dtype=np.float32))
                all_labels.append(np.array(point_labels, np.int32))

            if single_box is not None:
                predict_args["box"] = single_box.astype(np.float32)
                all_boxes.append(single_box.astype(np.float32))

            masks, scores, logits = self.predictor.predict(
                **predict_args, multimask_output=False, **kwargs
            )

            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]
            logits = logits[sorted_ind]

            all_masks.append(masks[0])
            all_scores.append(scores[0])
            all_logits.append(logits[0])

        all_masks = np.array(all_masks)
        all_scores = np.array(all_scores)
        all_logits = np.array(all_logits)
        all_boxes = np.array(all_boxes)
        if len(all_points)==0:
            all_points = None
            all_labels = None
        else:
            all_points = np.array(all_points)
            all_labels = np.array(all_labels)

        if show:
            self._visualize_prediction(all_masks, all_scores, all_points, all_boxes, all_labels, labels_names)

        return all_masks, all_scores, all_logits
    
    def _visualize_prediction(self, masks: np.ndarray, scores: np.ndarray,points: Optional[np.ndarray] = None,
                            bbox: Optional[np.ndarray] = None, point_labels: Optional[np.ndarray] = None,
                            labels_names: Optional[List[str]] = None) -> None:
        """Visualize the prediction."""
        # plt.figure(figsize=(10, 10))
        # plt.imshow(self.image)
        SAM2Processor.show_masks_image(self.image, masks, scores, point_coords=points,
                                     box_coords=bbox, point_labels=point_labels, labels_names=labels_names)