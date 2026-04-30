"""Document loaders."""
from .text import load_text_file, load_html_file, supports_text, supports_html, iter_supported_text_files, html_to_text
from .pdf import load_pdf_file
from .docx import load_docx_file

__all__ = [
    "load_text_file",
    "load_html_file",
    "load_pdf_file",
    "load_docx_file",
    "supports_text",
    "supports_html",
    "iter_supported_text_files",
    "html_to_text",
]
