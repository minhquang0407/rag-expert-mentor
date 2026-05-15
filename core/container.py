from dependency_injector import containers, providers
from config.settings import settings
from database.structural_db import QdrantVectorStore
from database.semantic_dag import Neo4jManager
from orchestrator.llm_factory import LLMFactory
from orchestrator.llm_service import LLMService
from runtime.engine import SupportAgent, RuntimeEngine
from runtime.queue import QueueOrchestrator
from runtime.agent_runtime import MultiAgentRuntime


class Container(containers.DeclarativeContainer):
    # Pass settings object to configuration
    config = providers.Configuration()

    # Database Providers
    vector_db = providers.Singleton(
        QdrantVectorStore,
        host=config.qdrant_host,
        port=config.qdrant_port,
        api_key=config.qdrant_api_key,
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
        openai_api_key=config.openai_api_key,
        groq_api_key=config.groq_api_key,
        google_api_key=config.google_api_key,
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
        openai_api_key=config.openai_api_key,
        groq_api_key=config.groq_api_key,
        google_api_key=config.google_api_key,
    )

    qa_llm = providers.Singleton(
        LLMFactory.create_llm,
        provider=config.llm_provider,
        model_name=config.llm_model_name,
        base_url=config.llm_base_url,
        temperature=config.qa_temperature,
        max_tokens=config.qa_max_token,
        top_p=config.qa_top_p,
        openai_api_key=config.openai_api_key,
        groq_api_key=config.groq_api_key,
        google_api_key=config.google_api_key,
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

    multi_agent_runtime = providers.Singleton(
        MultiAgentRuntime,
        llm_service=llm_service,
        vector_db=vector_db,
        graph_db=graph_db,
        critic_enabled=config.critic_enabled,
        max_revision_loops=config.max_revision_loops,
        stream_agent_outputs=config.stream_agent_outputs,
    )

    # Main Engine
    runtime_engine = providers.Singleton(
        RuntimeEngine,
        orchestrator=queue_orchestrator,
        vector_db=vector_db,
        graph_db=graph_db,
        support_agent=support_agent,
        multi_agent_runtime=multi_agent_runtime,
        use_multi_agent_runtime=config.use_multi_agent_runtime,
    )
