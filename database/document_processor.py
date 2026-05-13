from typing import List, Dict, Any, Tuple
from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Chapter"),
    ("##", "Section")
]

class MathAwareDocumentProcessor:
    def __init__(self):
        """
        - Reason: Simplified document processor focusing only on Structural Parsing (TOC & Sections).
        - Function: Initializes without max_chunk_size since we now preserve full section macro-contexts.
        """
        pass

    def process_markdown(self, markdown_text: str) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
        """
        - Reason: To parse the structural hierarchy of the textbook without unnecessarily breaking math blocks.
        - Function: Splits markdown by headers, builds a Table of Contents (TOC), and yields full sections.
        - Usage: Called by the data ingestion pipeline.
        - Parameters:
            - markdown_text (str): The raw string content of the markdown file.
        - Returns: A tuple containing the list of section documents and the TOC dictionary.
        """
        print("[*] Section-Level Structural Parsing & TOC Extraction")
        final_documents = []
        toc = {}

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        splits = markdown_splitter.split_text(markdown_text)

        global_seq_id = 0

        for split in splits:
            # --- IDEAL TABLE OF CONTENTS (TOC) EXTRACTION ---
            chapter = split.metadata.get("Chapter", "General Chapter: Introduction")
            section = split.metadata.get("Section", "General Section: Introduction")

            if chapter not in toc:
                toc[chapter] = []
            if section not in toc[chapter]:
                toc[chapter].append(section)
            # -------------------------------------

            meta = split.metadata.copy()
            meta["Chapter"] = chapter
            meta["Section"] = section
            meta["seq_id"] = global_seq_id
            final_documents.append({
                "page_content": split.page_content.strip(),
                "metadata": meta
            })
            global_seq_id += 1

        print(f"[*] Successfully extracted {len(final_documents)} Sections and built the TOC tree.")
        return final_documents, toc