## Docling data extract

from typing import List
from mb_rag.utils.extra import check_package

__all__ = ['DocumentExtractor']

class DocumentExtractor:
    """
    DocumentExtractor class for extracting data from documents using Docling.
    """
    
    def __init__(self):
        """
        Initialize the DocumentExtractor class.
        Checking for Docling package.
        """
        if not check_package("docling"):
            raise ImportError("Docling package not found. Please install it using: pip install docling")
        from docling import Docling
        self.Docling = Docling
    
    def _extract_data(self, file_path: str, **kwargs) -> List[str]:
        """
        Extract data from a document using Docling.
        """
        try:
            docling = self.Docling(file_path, **kwargs)
            return docling.extract()
        except Exception as e:
            raise Exception(f"Error extracting data from {file_path}: {str(e)}")

    def get_data(self,file_path: str, save_path: str = None, data_store_type: str = "markdown",**kwargs) -> List[str]:
        """
        Get data from a document using Docling.
        Args:
            file_path (str): Path to the document
            save_path (str): Path to save the extracted data. Default is None. If None, data saved as Markdown file as docling_{file_name}.md
            data_store_type (str): Saving document as markdown, txt or html. Default is markdown
            **kwargs: Additional arguments for Docling
        Returns:
            List[str]: Extracted data
        """
        data = self._extract_data(file_path, **kwargs)   
        if data_store_type == "markdown":
            data_type = "md"
        elif data_store_type == "txt":
            data_type = "txt"
        elif data_store_type == "html":
            data_type = "html"
        else:
            print("Invalid data store type. Defaulting to text (txt)")
            data_type = "txt"
        if save_path is None:
            save_path = f"docling_{file_path.split('/')[-1].split('.')[0]}.{data_type}"
            print(f"Saving extracted data to {save_path}")
        if data_store_type == "markdown":
            data_with_type = data.document.export_to_markdown()
        elif data_store_type == "txt":
            data_with_type = data.document.export_to_text()
        elif data_store_type == "html":
            data_with_type = data.document.export_to_html()
        with open(save_path, 'w') as f:
            f.write(data_with_type)
        return data