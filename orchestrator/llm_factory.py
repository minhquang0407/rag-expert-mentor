from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

class LLMFactory:
    """
    Factory to instantiate the appropriate LangChain ChatModel based on configuration.
    """
    @staticmethod
    def create_llm(provider: str, model_name: str, base_url: str, temperature: float, max_tokens: int = 2048, top_p: float = 0.2, **kwargs) -> BaseChatModel:
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
        elif provider.lower() == "openai":
            # Abstracted for future implementation
            raise NotImplementedError("OpenAI provider is prepared abstractly but not fully implemented.")
        elif provider.lower() == "gemini":
            # Abstracted for future implementation
            raise NotImplementedError("Gemini provider is prepared abstractly but not fully implemented.")
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
