import re
from typing import List, Dict, Any, Tuple
from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Chapter"),
    ("##", "Section"),
    ("###", "Subsection"),
]


class MathAwareDocumentProcessor:
    def __init__(self, max_chunk_size: int = 1500):
        self.max_chunk_size = max_chunk_size
        self.math_block_pattern = re.compile(r'(\$\$[\s\S]*?\$\$)', re.MULTILINE)

    def _safe_math_split(self, text_content: str) -> List[str]:
        tokens = self.math_block_pattern.split(text_content)
        chunks = []
        current_chunk = ""
        for token in tokens:
            if not token.strip(): continue
            if token.startswith("$$"):
                if len(current_chunk) + len(token) > self.max_chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = token
                else:
                    current_chunk += "\n" + token
            else:
                paragraphs = token.split('\n\n')
                for p in paragraphs:
                    if not p.strip(): continue
                    if len(current_chunk) + len(p) > self.max_chunk_size and current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = p
                    else:
                        current_chunk += "\n\n" + p if current_chunk else p
        if current_chunk: chunks.append(current_chunk.strip())
        return chunks

    def process_markdown(self, markdown_text: str) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
        print("[*] Safe Chunking & TOC Extraction")
        final_documents = []
        toc = {}  # Khởi tạo Cây mục lục rỗng

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        splits = markdown_splitter.split_text(markdown_text)

        global_seq_id = 0

        for split in splits:
            # --- LOGIC RÚT TRÍCH MỤC LỤC (TOC) ---
            chapter = split.metadata.get("Chapter")
            section = split.metadata.get("Section")

            if chapter:
                if chapter not in toc:
                    toc[chapter] = []
                if section and section not in toc[chapter]:
                    toc[chapter].append(section)
            # -------------------------------------

            safe_chunks = self._safe_math_split(split.page_content)
            for chunk in safe_chunks:
                meta = split.metadata.copy()
                meta["seq_id"] = global_seq_id
                final_documents.append({
                    "page_content": chunk,
                    "metadata": meta
                })
                global_seq_id += 1

        print(f"[*] Đã tạo thành công {len(final_documents)} chunks và Cây mục lục.")
        # Trả về 2 giá trị: Danh sách chunk và Cây mục lục (TOC)
        return final_documents, toc