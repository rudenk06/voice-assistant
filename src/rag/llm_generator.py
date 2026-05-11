import logging

from src.utils.memory import force_gc, log_memory_usage

logger = logging.getLogger(__name__)

NO_CONTEXT_ANSWER = "В материалах кафедры нет информации по этому вопросу."


class LocalLLMGenerator:
    """Local GGUF answer generator using llama.cpp."""

    def __init__(
        self,
        model_path: str,
        max_tokens: int = 80,
        temperature: float = 0.1,
        context_size: int = 2048,
    ):
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.context_size = context_size
        self._llm = None

    def load(self):
        if self._llm is not None:
            return

        try:
            log_memory_usage("before LLM load")
            from llama_cpp import Llama

            self._llm = Llama(
                model_path=self.model_path,
                n_ctx=self.context_size,
                n_threads=4,
                verbose=False,
            )
            log_memory_usage("after LLM load")
            logger.info("Local LLM loaded: %s", self.model_path)
        except ImportError as e:
            raise RuntimeError(
                "llama-cpp-python is required when rag.use_llm=true"
            ) from e

    def generate(self, question: str, chunks: list[dict]) -> str:
        if not chunks:
            return NO_CONTEXT_ANSWER
        if self._llm is None:
            self.load()

        prompt = self._build_prompt(question, chunks)
        output = self._llm(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=["\n\n", "Вопрос:", "Контекст:"],
            echo=False,
        )
        answer = output["choices"][0]["text"].strip().strip('"')
        return answer or NO_CONTEXT_ANSWER

    def _build_prompt(self, question: str, chunks: list[dict]) -> str:
        context_parts = []
        for idx, chunk in enumerate(chunks, start=1):
            source = chunk.get("document_name", "unknown")
            text = chunk.get("text", "").strip()
            if text:
                context_parts.append(f"[{idx}] {source}\n{text}")

        context = "\n\n".join(context_parts)
        return (
            "Ты отвечаешь на вопросы по материалам кафедры.\n"
            "Используй только приведенный контекст. "
            "Не добавляй сведения, которых нет в контексте.\n"
            "Ответ должен быть кратким и пригодным для озвучивания: "
            "обычно одно предложение, максимум несколько предложений.\n"
            f"Если в контексте нет ответа, верни ровно: {NO_CONTEXT_ANSWER}\n\n"
            f"Контекст:\n{context}\n\n"
            f"Вопрос: {question}\n\n"
            "Ответ:"
        )

    def unload(self):
        if self._llm is not None:
            del self._llm
            self._llm = None
            force_gc()
            log_memory_usage("after LLM unload")
            logger.info("Local LLM unloaded")
