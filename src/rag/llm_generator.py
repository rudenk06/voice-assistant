import logging
import re

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
        threads: int = 4,
    ):
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.context_size = context_size
        self.threads = threads
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
                n_threads=self.threads,
                verbose=False,
            )
            log_memory_usage("after LLM load")
            logger.info("Local LLM loaded: %s", self.model_path)
        except ImportError as e:
            raise RuntimeError(
                "llama-cpp-python is required when rag.use_llm=true"
            ) from e

    def generate(
        self,
        question: str,
        chunks: list[dict],
        is_contact: bool = False,
        is_person: bool = False,
        is_overview: bool = False,
        is_department_name: bool = False,
    ) -> str:
        if not chunks:
            return NO_CONTEXT_ANSWER
        if self._llm is None:
            self.load()

        prompt = self._build_prompt(
            question,
            chunks,
            is_contact=is_contact,
            is_person=is_person,
            is_overview=is_overview,
            is_department_name=is_department_name,
        )
        output = self._llm(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=["<|im_end|>", "<|endoftext|>"],
            echo=False,
        )
        answer = self._postprocess(output["choices"][0]["text"])
        return answer or NO_CONTEXT_ANSWER

    def _build_prompt(
        self,
        question: str,
        chunks: list[dict],
        is_contact: bool = False,
        is_person: bool = False,
        is_overview: bool = False,
        is_department_name: bool = False,
    ) -> str:
        context_parts = []
        for idx, chunk in enumerate(chunks, start=1):
            source = chunk.get("document_name", "unknown")
            text = chunk.get("text", "").strip()
            if text:
                context_parts.append(f"[{idx}] {source}\n{text}")

        context = "\n\n".join(context_parts)
        contact_rule = ""
        if is_contact:
            contact_rule = (
                "Если вопрос про почту, email, контакт или как связаться, "
                "ответ должен содержать email из контекста. "
                "Не заменяй email на имя, отчество, должность или описание. "
                "Если email есть в контексте, обязательно скопируй email. "
                f"Если email нет в контексте, верни ровно: {NO_CONTEXT_ANSWER}\n"
            )
        person_rule = ""
        if is_person:
            person_rule = (
                "Если вопрос про человека, должность или кем он является, "
                "не отвечай email вместо должности или описания. "
                "Опиши человека по контексту: ФИО, роль, должность или кабинет, если они есть. "
                "Email добавляй только если пользователь спрашивает про контакт или почту.\n"
            )
        overview_rule = ""
        if is_overview:
            overview_rule = (
                "Если вопрос общий про кафедру, не возвращай fallback, если в контексте "
                "есть хоть какие-то сведения. Ответь максимум двумя короткими предложениями. "
                "Обязательно используй несколько разных фактов из контекста, если они есть: "
                "год основания, заведующего и научные направления. "
                "Не ограничивайся только годом, если в контексте есть заведующий или направления. "
                "Не отвечай одним фактом: overview-ответ должен кратко объединять сведения "
                "о заведующем, направлениях и годе основания, если они есть в контексте. "
                "Структура overview-ответа: первое короткое предложение про кафедру и год основания; "
                "второе короткое предложение про заведующего и несколько научных направлений. "
                "Ответ неполный, если в контексте есть заведующий и направления, а ты их не упомянул. "
                "Не перечисляй длинные списки полностью, выбери только несколько ключевых направлений. "
                "Не включай email. Не используй внешние знания.\n"
            )
        department_rule = ""
        if is_department_name:
            department_rule = (
                "Не используй внешние знания. Если пользователь спрашивает про "
                "'технологию программирования' или 'технологии программирования' "
                "в контексте ассистента, трактуй это как название кафедры. "
                "Отвечай о кафедре, если контекст содержит сведения о кафедре. "
                "Запрещено отвечать, что это область знаний, набор практик, процесс разработки "
                "или программные продукты. Начни ответ с того, что Технологии программирования "
                "это кафедра, затем кратко перечисли, какие сведения есть в материалах. "
                "Если есть сведения о заведующем или научных направлениях, кратко упомяни их. "
                f"Не добавляй фразу \"{NO_CONTEXT_ANSWER}\", если уже назвал сведения из контекста. "
                "Не давай общее определение из внешних знаний.\n"
            )
        if is_overview or is_department_name:
            answer_length_rule = (
                "Ответ должен быть пригодным для озвучивания: максимум два коротких "
                "предложения, без длинных рассуждений.\n"
            )
        else:
            answer_length_rule = (
                "Ответ должен быть пригодным для озвучивания: коротко, без рассуждений, "
                "обычно одно предложение, максимум два-три предложения.\n"
            )

        return (
            "<|im_start|>system\n"
            "Ты голосовой ассистент кафедры. Отвечай только на русском языке.\n"
            "Используй только факты из найденного контекста, ничего не выдумывай.\n"
            "Если ответ есть в контексте, ответь по контексту.\n"
            "Если вопрос общий, кратко перечисли найденные сведения из контекста.\n"
            f"Фразу \"{NO_CONTEXT_ANSWER}\" используй только если в контексте "
            "действительно нет подходящей информации.\n"
            "Не повторяй одну и ту же фразу. Не пиши \"согласно контексту\".\n"
            f"{contact_rule}"
            f"{person_rule}"
            f"{overview_rule}"
            f"{department_rule}"
            f"{answer_length_rule}"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"Контекст:\n{context}\n\n"
            f"Вопрос: {question}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def _postprocess(self, text: str) -> str:
        answer = text.strip().strip('"')
        for token in ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "assistant"):
            answer = answer.replace(token, "")
        answer = re.sub(r"\s+", " ", answer).strip().strip('"')

        if answer.startswith(NO_CONTEXT_ANSWER):
            return NO_CONTEXT_ANSWER

        if NO_CONTEXT_ANSWER in answer:
            answer = self._remove_fallback_sentences(answer)

        answer = self._dedupe_repeated_sentences(answer)
        sentences = re.findall(r"[^.!?]+[.!?]+|[^.!?]+$", answer)
        limited = " ".join(s.strip() for s in sentences[:3]).strip()
        return self._trim_to_complete_sentence(limited or answer)

    def _dedupe_repeated_sentences(self, text: str) -> str:
        sentences = re.findall(r"[^.!?]+[.!?]+|[^.!?]+$", text)
        result = []
        seen = set()
        for sentence in sentences:
            normalized = re.sub(r"\s+", " ", sentence).strip()
            if not normalized:
                continue
            key = normalized.rstrip(".!?").casefold()
            if key in seen:
                continue
            seen.add(key)
            result.append(normalized)
        return " ".join(result).strip()

    def _remove_fallback_sentences(self, text: str) -> str:
        sentences = re.findall(r"[^.!?]+[.!?]+|[^.!?]+$", text)
        kept = [sentence.strip() for sentence in sentences if NO_CONTEXT_ANSWER not in sentence]
        return " ".join(kept).strip()

    def _trim_to_complete_sentence(self, text: str) -> str:
        if not text or text.endswith((".", "!", "?")):
            return text

        last_end = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
        if last_end == -1:
            return text
        return text[: last_end + 1].strip()

    def unload(self):
        if self._llm is not None:
            del self._llm
            self._llm = None
            force_gc()
            log_memory_usage("after LLM unload")
            logger.info("Local LLM unloaded")
