import pytest
from unittest.mock import MagicMock, patch
import json

# ==========================================
# FIXTURES (MOCKING)
# ==========================================
@pytest.fixture
def mock_dependencies():
    mock_llm = MagicMock()
    mock_graph_db = MagicMock()
    
    # Mock Graph DB Returns
    mock_graph_db.get_raw_chat_turns.return_value = [
        {"id": "turn_123", "raw_query": "What is GNN?", "raw_answer": "GNN is..."}
    ]
    
    # Mock LLM Returns (Default to needing raw data)
    mock_response = MagicMock()
    mock_response.content = '{"action": "fetch_raw", "turn_ids": ["turn_123"]}'
    mock_llm.invoke.return_value = mock_response
    
    return mock_llm, mock_graph_db

# ==========================================
# MOCK CLASSES (TDD INTERFACES)
# ==========================================
# These classes represent the expected interface we will build later.
class MockSupportAgent:
    def __init__(self, llm):
        self.llm = llm
        
    def route_and_answer(self, query: str, summary_history: str, graph_context: list):
        # Simulates the Router Agent logic
        response = self.llm.invoke("prompt")
        data = json.loads(response.content)
        
        if data.get("action") == "fetch_raw":
            return "NEEDS_RAW", data.get("turn_ids", [])
        return "ANSWERED", data.get("response", "")

class MockGraphMemoryManager:
    def __init__(self, graph_db):
        self.graph_db = graph_db
        
    def save_chat_turn(self, query: str, raw_answer: str, summary: str, concept_id: str):
        self.graph_db.save_chat_turn(query, raw_answer, summary, concept_id)
        
    def fetch_raw_history(self, turn_ids: list):
        return self.graph_db.get_raw_chat_turns(turn_ids)

# ==========================================
# UNIT TESTS: ROUTER AGENT
# ==========================================
def test_router_agent_decides_to_answer(mock_dependencies):
    """Happy Path: LLM has enough info from summaries and answers directly."""
    mock_llm, _ = mock_dependencies
    mock_response = MagicMock()
    mock_response.content = '{"action": "answer", "response": "GNN applies to graphs."}'
    mock_llm.invoke.return_value = mock_response
    
    agent = MockSupportAgent(mock_llm)
    status, result = agent.route_and_answer("What does it apply to?", "Summary...", [])
    
    assert status == "ANSWERED"
    assert result == "GNN applies to graphs."

def test_router_agent_decides_to_fetch(mock_dependencies):
    """Happy Path: LLM realizes it needs raw details and requests fetch."""
    mock_llm, _ = mock_dependencies
    
    agent = MockSupportAgent(mock_llm)
    status, turn_ids = agent.route_and_answer("Can you show me the math again?", "Summary...", [])
    
    assert status == "NEEDS_RAW"
    assert "turn_123" in turn_ids

# ==========================================
# UNIT TESTS: GRAPH MEMORY MANAGER
# ==========================================
def test_fetch_raw_history(mock_dependencies):
    """Integration Mock: Ensure Memory Manager correctly queries Neo4j for raw texts."""
    _, mock_graph_db = mock_dependencies
    
    memory = MockGraphMemoryManager(mock_graph_db)
    raw_data = memory.fetch_raw_history(["turn_123"])
    
    mock_graph_db.get_raw_chat_turns.assert_called_once_with(["turn_123"])
    assert len(raw_data) == 1
    assert raw_data[0]["raw_query"] == "What is GNN?"

def test_save_chat_turn(mock_dependencies):
    """Integration Mock: Ensure full turn (raw + summary) is pushed to Neo4j."""
    _, mock_graph_db = mock_dependencies
    
    memory = MockGraphMemoryManager(mock_graph_db)
    memory.save_chat_turn("Q", "A", "Sum", "Concept_A")
    
    mock_graph_db.save_chat_turn.assert_called_once_with("Q", "A", "Sum", "Concept_A")
