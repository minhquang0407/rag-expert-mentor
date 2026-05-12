from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

class LLMFactory:
    """
    Factory to instantiate the appropriate LangChain ChatModel based on configuration.
    """
    @staticmethod
    def create_llm(provider: str, model_name: str, base_url: str, temperature: float, max_tokens: int = 2048, top_p: float = 0.2, **kwargs) -> BaseChatModel:
        import streamlit as st
        st.write(f"🧪 LLMFactory: Creating model for provider '{provider}'...")
        
        # Capture internal keys before cleaning up kwargs
        openai_key = kwargs.get("openai_api_key", "not-needed")
        groq_key = kwargs.get("groq_api_key")
        google_key = kwargs.get("google_api_key")
        
        if provider.lower() == "groq" and not groq_key:
            st.error("⚠️ GROQ_API_KEY is missing! Check your environment/secrets.")
        elif provider.lower() == "openai" and not openai_key:
             st.error("⚠️ OPENAI_API_KEY is missing!")
        elif provider.lower() == "gemini" and not google_key:
             st.error("⚠️ GOOGLE_API_KEY is missing!")
             
        # Clean up kwargs to avoid passing internal config keys to LangChain models
        internal_keys = ["openai_api_key", "google_api_key", "groq_api_key"]
        for key in internal_keys:
            kwargs.pop(key, None)

        if provider.lower() == "ollama":
            # For local Ollama via OpenAI API compatibility
            return ChatOpenAI(
                base_url=base_url,
                api_key="not-needed",
                model=model_name,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                **kwargs
            )
        elif provider.lower() == "groq":
            from langchain_groq import ChatGroq
            return ChatGroq(
                groq_api_key=groq_key,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        elif provider.lower() == "openai":
            return ChatOpenAI(
                base_url=base_url if base_url else None,
                api_key=openai_key,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        elif provider.lower() == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                google_api_key=google_key,
                model=model_name,
                temperature=temperature,
                max_output_tokens=max_tokens,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
