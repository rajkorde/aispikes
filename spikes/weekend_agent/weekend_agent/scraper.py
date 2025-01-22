from typing import Optional

from langchain_community.document_loaders import WebBaseLoader


def scrape_webpage(url: str) -> Optional[str]:
    """Use requests and bs4 to scrape the provided web pages for detailed information."""
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs[0].page_content if docs and len(docs) >= 1 else None
