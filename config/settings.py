from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # LLM Configuration
    llm_provider: str = Field(default="ollama", description="Provider for LLM (e.g. ollama, openai, gemini)")
    llm_model_name: str = Field(default="qwen2.5:7b")
    llm_base_url: str = Field(default="http://localhost:11434/v1")
    ingestion_temperature: float = Field(default=0.0)
    ingestion_top_p: float = Field(default=0.2)
    ingestion_max_token: int = Field(default=2048)
    learning_temperature: float = Field(default=0.3)
    learning_top_p: float = Field(default=0.7)
    learning_max_token: int = Field(default=4096)
    qa_temperature: float = Field(default=0.3)
    qa_top_p: float = Field(default=0.7)
    qa_max_token: int = Field(default=2048)

    # API Keys for Cloud Deployment
    groq_api_key: str = Field(default="")
    google_api_key: str = Field(default="")
    openai_api_key: str = Field(default="")

    # Qdrant Vector DB Configuration
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_api_key: str = Field(default="")
    vector_collection_name: str = Field(default="math_curriculum_v4")

    # Neo4j Graph DB Configuration
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="ExpertMentor2026")

    # Multi-Agent Runtime Configuration
    use_multi_agent_runtime: bool = Field(default=False)
    critic_enabled: bool = Field(default=False)
    max_revision_loops: int = Field(default=1)
    stream_agent_outputs: bool = Field(default=True)
    planner_can_add_agents: bool = Field(default=True)
    planner_can_remove_required_agents: bool = Field(default=False)
    history_restore_limit: int = Field(default=50)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
