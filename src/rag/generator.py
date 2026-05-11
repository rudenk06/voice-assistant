import logging
import re
import time

from src.rag.llm_generator import LocalLLMGenerator, NO_CONTEXT_ANSWER

logger = logging.getLogger(__name__)

CONTACT_WORDS = (
    "почта",
    "email",
    "e-mail",
    "электронная почта",
    "связаться",
    "контакт",
    "контакты",
)

EMAIL_CONTACT_WORDS = (
    "почта",
    "email",
    "e-mail",
    "электронная почта",
)

PERSON_WORDS = (
    "кто такой",
    "кто такая",
    "кем является",
    "кто является",
    "должность",
)

OVERVIEW_PHRASES = (
    "какая информация о кафедре",
    "какая есть информация о кафедре",
    "что известно о кафедре",
    "расскажи о кафедре",
    "расскажи про кафедру",
    "информация о кафедре",
    "что есть о кафедре",
    "чем занимается кафедра",
)

EMAIL_RE = re.compile(
    r"[A-Za-zА-Яа-я0-9._%+\-]+(?:\s*[._%+\-]\s*[A-Za-zА-Яа-я0-9]+)*"
    r"\s*@\s*"
    r"[A-Za-zА-Яа-я0-9.-]+(?:\s*\.\s*[A-Za-zА-Яа-я]{0,3})?",
    re.IGNORECASE,
)

CONTACT_STOP_WORDS = {
    "какая",
    "какой",
    "какие",
    "почта",
    "email",
    "e-mail",
    "электронная",
    "связаться",
    "контакт",
    "контакты",
}


def is_contact_intent(text: str) -> bool:
    lowered = text.casefold()
    return any(word in lowered for word in CONTACT_WORDS)


def is_email_contact_intent(text: str) -> bool:
    lowered = text.casefold()
    return any(word in lowered for word in EMAIL_CONTACT_WORDS)


def is_person_intent(text: str) -> bool:
    lowered = text.casefold()
    return any(phrase in lowered for phrase in PERSON_WORDS)


def is_overview_intent(text: str) -> bool:
    lowered = text.casefold()
    return any(phrase in lowered for phrase in OVERVIEW_PHRASES)


def is_department_name_question(text: str) -> bool:
    lowered = text.casefold()
    has_name = (
        "технология программирования" in lowered
        or "технологии программирования" in lowered
    )
    has_question = (
        "что такое" in lowered
        or "расскажи про" in lowered
        or "расскажи о" in lowered
        or "что известно о" in lowered
        or "что я могу узнать" in lowered
    )
    return has_name and has_question


def detect_intent(text: str) -> str:
    if is_contact_intent(text):
        return "contact"
    if is_department_name_question(text):
        return "department"
    if is_person_intent(text):
        return "person"
    if is_overview_intent(text):
        return "overview"
    return "other"


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
        llm_threads: int = 4,
        llm_unload_after_generate: bool = True,
        model_path: str = None,
        mode: str = "template",
    ):
        self.use_llm = use_llm or mode == "llm"
        self.llm_backend = llm_backend
        self.llm_unload_after_generate = llm_unload_after_generate
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
                threads=llm_threads,
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
            try:
                intent = detect_intent(query)
                llm_context = self._context_for_intent(query, context, intent)
                compact_used = _is_compact_context(llm_context)

                if intent == "contact":
                    email = best_contact_email(query, llm_context)
                    if email:
                        logger.info(
                            "RAG generation debug: intent=%s retrieved_chunks=%s "
                            "compact_context=%s compact_chars=%s llm_called=%s generation_seconds=%.2f",
                            intent,
                            len(context),
                            compact_used,
                            _context_text_length(llm_context) if compact_used else 0,
                            False,
                            0.0,
                        )
                        return f"Электронная почта: {email}."
                    return NO_CONTEXT_ANSWER

                start = time.perf_counter()
                answer = self._llm_generator.generate(
                    query,
                    llm_context,
                    is_contact=intent == "contact",
                    is_person=intent == "person",
                    is_overview=intent == "overview",
                    is_department_name=intent == "department",
                )
                final_answer = self._postprocess_answer(query, llm_context, answer, intent)
                logger.info(
                    "RAG generation debug: intent=%s retrieved_chunks=%s "
                    "compact_context=%s compact_chars=%s llm_called=%s generation_seconds=%.2f",
                    intent,
                    len(context),
                    compact_used,
                    _context_text_length(llm_context) if compact_used else 0,
                    True,
                    time.perf_counter() - start,
                )
                return final_answer
            finally:
                if self.llm_unload_after_generate:
                    self._llm_generator.unload()
        return self._generate_template(query, context)

    def _context_for_intent(self, query: str, context: list[dict], intent: str) -> list[dict]:
        if intent == "contact":
            return self._contact_context(query, context)[:1]
        if intent == "person":
            return self._contact_context(query, context)[:2]
        if intent in ("overview", "department"):
            return [_compact_context_chunk(context)]
        return context

    def _contact_context(self, query: str, context: list[dict]) -> list[dict]:
        return sorted(
            context,
            key=lambda chunk: _chunk_name_score(query, chunk.get("text", "")),
            reverse=True,
        )

    def _postprocess_answer(
        self,
        query: str,
        context: list[dict],
        answer: str,
        intent: str,
    ) -> str:
        if intent == "contact":
            return self._postprocess_contact_answer(query, context, answer)
        if intent == "person":
            return self._postprocess_person_answer(context, answer)
        if intent == "overview":
            return self._postprocess_overview_answer(context, answer)
        if intent == "department":
            return self._postprocess_department_name_answer(context, answer)
        return answer

    def _postprocess_contact_answer(
        self,
        query: str,
        context: list[dict],
        answer: str,
    ) -> str:
        if not is_contact_intent(query):
            return answer
        if "@" in answer:
            return answer

        email = best_contact_email(query, context)
        if email:
            return f"Электронная почта: {email}."
        return NO_CONTEXT_ANSWER

    def _postprocess_person_answer(self, context: list[dict], answer: str) -> str:
        lowered = answer.casefold()
        if "@" in answer or "электронной почт" in lowered or "электронная почта" in lowered:
            person_answer = _person_answer_from_context(context)
            if person_answer:
                return person_answer
        return answer

    def _postprocess_overview_answer(self, context: list[dict], answer: str) -> str:
        return _limit_to_two_sentences(answer)

    def _postprocess_department_name_answer(self, context: list[dict], answer: str) -> str:
        return answer

    def _generate_template(self, query: str, context: list[dict]) -> str:
        """Simple template-based answer: return best matching chunk."""
        best = context[0]
        return f"По данным кафедры: {best['text']}"

    def unload(self, force: bool = False):
        """Free LLM from RAM."""
        if self._llm_generator is not None and (force or self.llm_unload_after_generate):
            self._llm_generator.unload()


def best_contact_email(query: str, chunks: list[dict]) -> str:
    best_email = ""
    best_score = -1

    for index, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        emails = extract_emails(text)
        if not emails:
            continue

        name_score = _chunk_name_score(query, text)
        # Prefer surname match first, then retriever order.
        score = name_score * 1000 - index
        if score > best_score:
            best_score = score
            best_email = emails[0]

    return best_email


def extract_emails(text: str) -> list[str]:
    emails = []
    for match in EMAIL_RE.finditer(text):
        email = _normalize_email(match.group(0))
        if email:
            emails.append(email)
    return emails


def _normalize_email(email: str) -> str:
    compact = re.sub(r"\s+", "", email).strip().strip(".,;:")
    if "@" not in compact:
        return ""

    local, domain = compact.split("@", 1)
    domain = domain.lower()
    domain = re.sub(r"\.+", ".", domain).strip(".")
    if domain == "spbu":
        domain = "spbu.ru"
    elif domain.startswith("spbu.") and domain != "spbu.ru":
        domain = "spbu.ru"
    return f"{local.lower()}@{domain}"


def _chunk_name_score(query: str, text: str) -> int:
    text_lower = text.casefold()
    best = 0
    for variant, score, mode in _query_name_variants(query):
        if mode == "word":
            if re.search(rf"\b{re.escape(variant)}\b", text_lower):
                best = max(best, score)
        elif variant in text_lower:
            best = max(best, score)
    return best


def _query_name_variants(query: str) -> list[tuple[str, int, str]]:
    words = re.findall(r"[A-Za-zА-Яа-яЁё-]+", query.casefold())
    variants = []
    for word in words:
        if len(word) <= 4 or word in CONTACT_STOP_WORDS:
            continue
        variants.append((word, 1, "word"))

        if word.endswith("а") and len(word) > 5:
            variants.append((word[:-1], 4, "word"))

        for suffix in ("ого", "ова", "ева", "ым", "им", "ом", "ем"):
            if word.endswith(suffix) and len(word) - len(suffix) > 4:
                variants.append((word[: -len(suffix)], 4, "word"))

        for suffix in ("ой", "ая", "у", "е"):
            if word.endswith(suffix) and len(word) - len(suffix) > 4:
                variants.append((word[: -len(suffix)], 3, "prefix"))
    return variants


def _person_answer_from_context(chunks: list[dict]) -> str:
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue

        sentence = re.split(r"[.!?]\s*", text, maxsplit=1)[0].strip()
        sentence = re.sub(r",?\s*электронная почта\s+\S+.*$", "", sentence, flags=re.IGNORECASE)
        sentence = re.sub(r",?\s*почта\s+\S+.*$", "", sentence, flags=re.IGNORECASE)
        sentence = sentence.strip(" ,;")
        if sentence:
            return sentence + "."
    return ""


def _limit_to_two_sentences(answer: str) -> str:
    sentences = [s.strip() for s in re.findall(r"[^.!?]+[.!?]+|[^.!?]+$", answer) if s.strip()]
    if len(sentences) <= 2:
        return answer
    first = sentences[0].rstrip(".!?")
    second = sentences[1].strip()
    merged = f"{first}; {second[0].lower() + second[1:] if second else second}"
    return f"{merged} {' '.join(sentences[2:3])}".strip()


def _compact_context_chunk(chunks: list[dict]) -> dict:
    compact_text = _compact_context_text(chunks)
    return {
        "text": compact_text,
        "score": 1.0,
        "document_name": "compact_context",
    }


def _compact_context_text(chunks: list[dict]) -> str:
    facts = {
        "title": "",
        "year": "",
        "head": "",
        "science": "",
        "contacts": "",
    }

    for chunk in chunks:
        text = chunk.get("text", "").strip()
        lowered = text.casefold()
        if not facts["title"] and "кафедра технологии программирования" in lowered:
            facts["title"] = (
                "Название кафедры: технологии программирования; это название кафедры, "
                "а не общее понятие."
            )
        if not facts["year"] and "основана" in lowered:
            facts["year"] = _year_fact(text)
        if not facts["head"] and "заведующий кафедрой" in lowered:
            facts["head"] = _remove_email_details(_clean_fact(text))
        if not facts["science"] and "научные направления" in lowered:
            facts["science"] = _short_science_fact(text)
        if not facts["contacts"] and extract_emails(text):
            facts["contacts"] = "Есть сведения о преподавателях и контактах."

    if not any(facts.values()):
        return "Есть сведения о кафедре."

    summary_parts = ["кафедра технологии программирования"]
    if facts["year"]:
        summary_parts.append(facts["year"])
    if facts["head"]:
        summary_parts.append(facts["head"])
    if facts["science"]:
        summary_parts.append(facts["science"])
    if facts["contacts"]:
        summary_parts.append("есть сведения о преподавателях и контактах")

    return (
        "Сводка фактов из найденных материалов: "
        + "; ".join(part.rstrip(".") for part in summary_parts)
        + ". Технологии программирования — название кафедры, а не общее понятие."
    )


def _is_compact_context(chunks: list[dict]) -> bool:
    return len(chunks) == 1 and chunks[0].get("document_name") == "compact_context"


def _context_text_length(chunks: list[dict]) -> int:
    return sum(len(chunk.get("text", "")) for chunk in chunks)


def _clean_fact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().rstrip(".")


def _remove_email_details(text: str) -> str:
    text = re.sub(
        r",?\s*(?:электронная\s+почта|почта)\s+\S+(?:\s*\.\s*\S+)?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip(" ,;")


def _year_fact(text: str) -> str:
    match = re.search(r"\b(19\d{2}|20\d{2})\b", text)
    if match:
        return f"год основания — {match.group(1)}"
    return _clean_fact(text)


def _short_science_fact(text: str) -> str:
    _, _, tail = text.partition(":")
    if not tail:
        return ""

    items = [item.strip() for item in tail.split(",") if item.strip()]
    if not items:
        return ""
    selected = items[:4]
    if len(selected) == 1:
        joined = selected[0]
    else:
        joined = ", ".join(selected[:-1]) + " и " + selected[-1]
    return f"научные направления: {joined}"
