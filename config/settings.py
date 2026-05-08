from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # LLM Configuration
    llm_provider: str = Field(default="ollama", description="Provider for LLM (e.g. ollama, openai, gemini)")
    llm_model_name: str = Field(default="qwen2.5:7b")
    llm_base_url: str = Field(default="http://localhost:11434/v1")
    temperature: float = Field(default=0.3)
    top_p: float = Field(default=0.2)
    max_token: int = Field(default=2048)

    # Qdrant Vector DB Configuration
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    vector_collection_name: str = Field(default="math_curriculum_v4")

    # Neo4j Graph DB Configuration
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="ExpertMentor2026")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
