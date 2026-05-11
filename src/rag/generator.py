import logging

from src.rag.llm_generator import LocalLLMGenerator, NO_CONTEXT_ANSWER

logger = logging.getLogger(__name__)


class Generator:
    """Answer generator: template baseline or retrieval + local LLM."""

    def __init__(
        self,
        use_llm: bool = False,
        llm_backend: str = "llama_cpp",
        llm_model_path: str = None,
        llm_max_tokens: int = 80,
        llm_temperature: float = 0.1,
        context_size: int = 2048,
        model_path: str = None,
        mode: str = "template",
    ):
        self.use_llm = use_llm or mode == "llm"
        self.llm_backend = llm_backend
        self._llm_generator = None

        if self.use_llm:
            if self.llm_backend != "llama_cpp":
                raise ValueError(f"Unsupported LLM backend: {self.llm_backend}")
            resolved_model_path = llm_model_path or model_path
            if not resolved_model_path:
                raise ValueError("LLM model path is required when use_llm=true")
            self._llm_generator = LocalLLMGenerator(
                model_path=resolved_model_path,
                max_tokens=llm_max_tokens,
                temperature=llm_temperature,
                context_size=context_size,
            )

    def load(self):
        """Load local LLM when LLM mode is enabled."""
        if self._llm_generator is not None:
            self._llm_generator.load()

    def generate(self, query: str, context: list[dict]) -> str:
        """Generate answer from query and retrieved context chunks.

        Args:
            query: user's question text
            context: list of {text, score, document_name} from retriever

        Returns:
            Answer text string
        """
        if not context:
            if self.use_llm:
                return NO_CONTEXT_ANSWER
            return "К сожалению, я не нашёл информацию по вашему вопросу в базе знаний кафедры."

        if self._llm_generator is not None:
            return self._llm_generator.generate(query, context)
        return self._generate_template(query, context)

    def _generate_template(self, query: str, context: list[dict]) -> str:
        """Simple template-based answer: return best matching chunk."""
        best = context[0]
        return f"По данным кафедры: {best['text']}"

    def unload(self):
        """Free LLM from RAM."""
        if self._llm_generator is not None:
            self._llm_generator.unload()
