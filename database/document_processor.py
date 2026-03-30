import re
from typing import List, Dict
from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Chapter"),
    ("##", "Section"),
    ("###", "Subsection"),
]

class MathAwareDocumentProcessor:
    def __init__(self, max_chunk_size: int = 1500):
        self.max_chunk_size = max_chunk_size

        # Regex bắt chính xác các khối toán học nhiều dòng (Display Math)
        # Sử dụng [\s\S]*? để bắt mọi ký tự kể cả dấu xuống dòng (Non-greedy)
        self.math_block_pattern = re.compile(r'(\$\$[\s\S]*?\$\$)', re.MULTILINE)

    def _safe_math_split(self, text_content: str) -> List[str]:
        """
        Thuật toán cắt văn bản an toàn: Tách đoạn (Paragraph) nhưng
        tuyệt đối không băm ngang công thức Toán học.
        """
        # Phân tách văn bản thành mảng luân phiên: [text_thường, khối_toán, text_thường...]

        tokens = self.math_block_pattern.split(text_content)

        chunks = []
        current_chunk = ""

        for token in tokens:
            if not token.strip():
                continue

            # Kịch bản 1: Token hiện tại LÀ một khối Toán học ($$ ... $$)
            if token.startswith("$$"):
                # Buộc phải nhét toàn bộ khối toán vào chunk. Nếu tràn, ngắt chunk cũ.
                if len(current_chunk) + len(token) > self.max_chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = token
                else:
                    current_chunk += "\n" + token

            # Kịch bản 2: Token là Text thông thường (Có thể cắt bằng \n\n)
            else:
                paragraphs = token.split('\n\n')
                for p in paragraphs:
                    if not p.strip(): continue

                    if len(current_chunk) + len(p) > self.max_chunk_size and current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = p
                    else:
                        current_chunk += "\n\n" + p if current_chunk else p

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def process_markdown(self, markdown_text: str) -> List[Dict[str, Any]]:
        print("[*] Safe Chunking")
        final_documents = []

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        splits = markdown_splitter.split_text(markdown_text)

        global_seq_id = 0

        for split in splits:
            safe_chunks = self._safe_math_split(split.page_content)

            for chunk in safe_chunks:
                meta = split.metadata.copy()
                meta["seq_id"] = global_seq_id

                final_documents.append({
                    "page_content": chunk,
                    "metadata": meta
                })

                global_seq_id += 1

        print(f"[*] Đã tạo thành công {len(final_documents)} chunks an toàn.")
        return final_documents