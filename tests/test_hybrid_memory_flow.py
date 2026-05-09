import pytest
import os
import sys
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).parent.parent.resolve()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from core.container import Container
from config.settings import settings


def test_long_interactive_learning_session(capsys):
    """
    - Lí do tại sao dùng: Mô phỏng một người học liên tục hỏi nhiều câu liên tiếp.
    - Chức năng: In ra toàn bộ thông tin Semantic Memory, Recent History gửi vào LLM ở mỗi turn.
    """
    container = Container()
    container.config.from_pydantic(settings)
    engine = container.runtime_engine()

    import uuid
    user_id = f"student_demo_long_flow_{uuid.uuid4().hex[:4]}"
    
    queries = [
        "Graph Database (Đồ thị tri thức) là gì?",
        "Nó có giống với Vector Database mà tôi hay dùng để build RAG không?",
        "Vậy khi nào tôi nên dùng Graph thay vì Vector?",
        "Bạn có thể tóm tắt lại 3 câu hỏi vừa rồi tôi hỏi gì không?",
        "Tóm lại, nếu làm Chatbot tra cứu tài liệu y khoa thì tôi nên dùng loại DB nào?"
    ]

    print(f"\n\n{'='*60}")
    print(f"[START] BAT DAU PHIEN HOC DAI (User: {user_id})")
    print(f"{'='*60}\n")

    for i, query in enumerate(queries, 1):
        print(f"\n\n{'='*50}")
        print(f"👉 [LUOT {i}] NGUOI HOC HOI: {query}")
        print(f"{'='*50}")

        # 1. Lấy và in ra những gì sẽ gửi cho LLM từ Qdrant
        semantic_mem = engine.vector_db.search_semantic_memory(user_id, query, limit=3)
        print("\n[QDRANT] LICH SU NGU NGHIA SE GUI CHO LLM:")
        if not semantic_mem:
            print("   -> (Trong - Chua co tri nho ngu nghia nao phu hop)")
        else:
            for mem in semantic_mem:
                print(f"   - {mem.get('summary')}")

        # 2. Lấy và in ra những gì sẽ gửi cho LLM từ Neo4j
        recent_mem = engine.graph_db.get_recent_history(user_id, limit=5)
        print("\n[NEO4J] LICH SU CHAT GAN NHAT SE GUI CHO LLM:")
        if not recent_mem:
            print("   -> (Trong - Chua co lich su chat nao)")
        else:
            for mem in recent_mem:
                print(f"   - [Q: {mem.get('query')[:30]}...] -> {mem.get('summary')}")

        # 3. Gọi RuntimeEngine xử lý
        print("\n[ENGINE] Dang dua vao LLM Router de suy luan...")
        answer = engine.process_action(
            action_mode="GLOBAL_QA",
            query=query,
            target_file="",
            target_section="",
            user_id=user_id
        )

        # 4. In ra Response của LLM
        print("\n[LLM] PHAN HOI TU AI:")
        print(f"   {answer}")

        # 5. Xem lịch sử thật sự đã hình thành sau câu hỏi này (Neo4j raw_history)
        raw_history = engine.graph_db.get_raw_chat_turns_by_user(user_id)
        print("\n[NEO4J RAW] CAU TRUC CHUOI THOI GIAN HIEN TAI (Luu trong DB):")
        for turn in raw_history:
            print(f"   + Turn ID: {turn.get('id')} | Query: {turn.get('query')}")

    assert len(raw_history) == len(queries)

def test_fetch_raw_interactive_session(capsys):
    """
    - Lí do tại sao dùng: Mô phỏng người dùng hỏi một câu yêu cầu LLM phải truy cập lại văn bản thô (fetch_raw).
    - Chức năng: In ra nguyên văn Prompt và Response để xem LLM quyết định thế nào.
    """
    from unittest.mock import patch

    container = Container()
    container.config.from_pydantic(settings)
    engine = container.runtime_engine()

    import uuid
    user_id = f"student_fetch_raw_{uuid.uuid4().hex[:4]}"
    
    # 1. Tạo một lịch sử cực kỳ chi tiết
    q1 = "Giải thích thuật toán PageRank với 3 bước: 1. Khởi tạo, 2. Phân phối (rất quan trọng, hãy nói cực kỳ chi tiết), 3. Hội tụ."
    print("\n\n" + "="*80)
    print(f"👉 [TAO LICH SU] User: {q1}")
    
    # Run Q1 normal
    ans1 = engine.process_action("GLOBAL_QA", q1, "", "", user_id)
    print(f"🤖 [AI Tra loi]: {ans1[:150]}...\n")

    # 2. Hook vào LLM invoke để in ra Prompt và Response của luồng QA
    class LLMWrapper:
        def __init__(self, llm):
            self._llm = llm

        def invoke(self, messages, *args, **kwargs):
            print("\n\n" + "-"*80)
            print("IN: [PROMPT HOAN CHINH GUI CHO LLM]")
            if isinstance(messages, list):
                for msg in messages:
                    print(f"\n--- {getattr(msg, 'type', 'UNKNOWN').upper()} PROMPT ---")
                    print(getattr(msg, 'content', str(msg)))
            else:
                print(str(messages))
            print("-" * 80)

            response = self._llm.invoke(messages, *args, **kwargs)
            
            print("\nOUT: [RESPONSE TU LLM (RAW JSON)]")
            print(getattr(response, 'content', str(response)))
            print("-" * 80 + "\n")
                
            return response

        def __getattr__(self, name):
            return getattr(self._llm, name)

    engine.support_agent.llm = LLMWrapper(engine.support_agent.llm)

    # 3. Ép LLM phải lấy lại văn bản thô (vì summary bị thiếu chi tiết)
    q2 = "Trong buoc Phan phoi ma ban giai thich chi tiet o cau hoi truoc, co doan nao noi ve cong thuc tinh diem khong? Hay nhac lai chinh xac nhung gi ban da noi ve buoc do. Chac chan tom tat ngan se khong co du chu, hay tu dong lay raw text."
    
    print("\n\n" + "="*80)
    print(f"[LUOT 2 - EP FETCH_RAW] User: {q2}")
    
    ans2 = engine.process_action("GLOBAL_QA", q2, "", "", user_id)
        
    print("\n\n" + "="*80)
    print(f"[KET QUA CUOI CUNG SAU KHI LAY RAW]:")
    print(ans2)

    captured = capsys.readouterr()
    print(captured.out)
