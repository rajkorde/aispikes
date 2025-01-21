from typing import Optional

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document


def scrape_webpage(url: str) -> Optional[Document]:
    """Use requests and bs4 to scrape the provided web pages for detailed information."""
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs[0]


def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b


# This will be a tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b


def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b
