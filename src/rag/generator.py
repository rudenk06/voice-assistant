import logging
from src.utils.memory import force_gc, log_memory_usage

logger = logging.getLogger(__name__)


class Generator:
    """Answer generator: template mode (MVP) or LLM mode (enhanced)."""

    def __init__(self, model_path: str = None, mode: str = "template",
                 max_tokens: int = 100, context_size: int = 512):
        self.model_path = model_path
        self.mode = mode
        self.max_tokens = max_tokens
        self.context_size = context_size
        self._llm = None

    def load(self):
        """Load LLM model if in llm mode."""
        if self.mode != "llm" or not self.model_path:
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
            logger.info("LLM loaded for generation")
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}. Falling back to template mode.")
            self._llm = None

    def generate(self, query: str, context: list[dict]) -> str:
        """Generate answer from query and retrieved context chunks.

        Args:
            query: user's question text
            context: list of {text, score, document_name} from retriever

        Returns:
            Answer text string
        """
        if not context:
            return "К сожалению, я не нашёл информацию по вашему вопросу в базе знаний кафедры."

        if self.mode == "llm" and self._llm is not None:
            return self._generate_llm(query, context)
        return self._generate_template(query, context)

    def _generate_template(self, query: str, context: list[dict]) -> str:
        """Simple template-based answer: return best matching chunk."""
        best = context[0]
        return f"По данным кафедры: {best['text']}"

    def _generate_llm(self, query: str, context: list[dict]) -> str:
        """Generate answer using local LLM."""
        context_text = "\n\n".join(c["text"] for c in context[:2])

        prompt = (
            f"Ты — ассистент кафедры. Отвечай кратко и точно на русском языке, "
            f"используя только предоставленный контекст.\n\n"
            f"Контекст:\n{context_text}\n\n"
            f"Вопрос: {query}\n\n"
            f"Ответ:"
        )

        try:
            output = self._llm(
                prompt,
                max_tokens=self.max_tokens,
                stop=["\n\n", "Вопрос:"],
                echo=False,
            )
            answer = output["choices"][0]["text"].strip()
            if answer:
                return answer
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")

        # Fallback to template
        return self._generate_template(query, context)

    def unload(self):
        """Free LLM from RAM."""
        if self._llm is not None:
            del self._llm
            self._llm = None
            force_gc()
            log_memory_usage("after LLM unload")
            logger.info("LLM unloaded")
