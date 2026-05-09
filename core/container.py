from dependency_injector import containers, providers
from config.settings import settings
from database.structural_db import QdrantVectorStore
from database.semantic_dag import Neo4jManager
from orchestrator.llm_factory import LLMFactory
from orchestrator.llm_service import LLMService
from runtime.engine import SupportAgent, RuntimeEngine
from runtime.queue import QueueOrchestrator


class Container(containers.DeclarativeContainer):
    # Pass settings object to configuration
    config = providers.Configuration()

    # Database Providers
    vector_db = providers.Singleton(
        QdrantVectorStore,
        host=config.qdrant_host,
        port=config.qdrant_port,
        collection_name=config.vector_collection_name
    )

    graph_db = providers.Singleton(
        Neo4jManager,
        uri=config.neo4j_uri,
        user=config.neo4j_user,
        password=config.neo4j_password
    )

    # LLM Providers using Factory
    ingestion_llm = providers.Singleton(
        LLMFactory.create_llm,
        provider=config.llm_provider,
        model_name=config.llm_model_name,
        base_url=config.llm_base_url,
        temperature=config.ingestion_temperature,
        max_tokens=config.ingestion_max_token,
        top_p=config.ingestion_top_p,
        model_kwargs={"response_format": {"type": "json_object"}}
    )

    learning_llm = providers.Singleton(
        LLMFactory.create_llm,
        provider=config.llm_provider,
        model_name=config.llm_model_name,
        base_url=config.llm_base_url,
        temperature=config.learning_temperature,
        max_tokens=config.learning_max_token,
        top_p=config.learning_top_p,
    )

    qa_llm = providers.Singleton(
        LLMFactory.create_llm,
        provider=config.llm_provider,
        model_name=config.llm_model_name,
        base_url=config.llm_base_url,
        temperature=config.qa_temperature,
        max_tokens=config.qa_max_token,
        top_p=config.qa_top_p,
        model_kwargs={"response_format": {"type": "json_object"}}
    )

    # Services
    llm_service = providers.Singleton(
        LLMService,
        llm=ingestion_llm,
        chat_llm=learning_llm
    )

    support_agent = providers.Singleton(
        SupportAgent,
        llm=qa_llm
    )

    queue_orchestrator = providers.Singleton(
        QueueOrchestrator,
        llm_service=llm_service
    )

    # Main Engine
    runtime_engine = providers.Singleton(
        RuntimeEngine,
        orchestrator=queue_orchestrator,
        vector_db=vector_db,
        graph_db=graph_db,
        support_agent=support_agent
    )
