import re


MANUAL_REPLACEMENTS = {
    "СПбГУ": "эс пэ бэ гэ у",
    "GigaAM": "гига эм",
    "ONNX": "онникс",
}

LATIN_LETTERS = {
    "a": "а",
    "b": "бэ",
    "c": "цэ",
    "d": "дэ",
    "e": "е",
    "f": "эф",
    "g": "гэ",
    "h": "аш",
    "i": "и",
    "j": "джей",
    "k": "ка",
    "l": "эль",
    "m": "эм",
    "n": "эн",
    "o": "о",
    "p": "пэ",
    "q": "ку",
    "r": "эр",
    "s": "эс",
    "t": "тэ",
    "u": "у",
    "v": "вэ",
    "w": "дабл ю",
    "x": "икс",
    "y": "игрек",
    "z": "зэт",
}

TRANSLIT_MULTI = (
    ("sch", "щ"),
    ("sh", "ш"),
    ("ch", "ч"),
    ("zh", "ж"),
    ("yu", "ю"),
    ("ya", "я"),
    ("yo", "ё"),
    ("ts", "ц"),
)

TRANSLIT_SINGLE = {
    "a": "а",
    "b": "б",
    "c": "к",
    "d": "д",
    "e": "е",
    "f": "ф",
    "g": "г",
    "h": "х",
    "i": "и",
    "j": "дж",
    "k": "к",
    "l": "л",
    "m": "м",
    "n": "н",
    "o": "о",
    "p": "п",
    "q": "к",
    "r": "р",
    "s": "с",
    "t": "т",
    "u": "у",
    "v": "в",
    "w": "в",
    "x": "кс",
    "y": "ы",
    "z": "з",
}

HUNDREDS_1900 = {
    19: "тысяча девятьсот",
    20: "две тысячи",
}

ORDINALS_0_19 = {
    0: "",
    1: "первом",
    2: "втором",
    3: "третьем",
    4: "четвёртом",
    5: "пятом",
    6: "шестом",
    7: "седьмом",
    8: "восьмом",
    9: "девятом",
    10: "десятом",
    11: "одиннадцатом",
    12: "двенадцатом",
    13: "тринадцатом",
    14: "четырнадцатом",
    15: "пятнадцатом",
    16: "шестнадцатом",
    17: "семнадцатом",
    18: "восемнадцатом",
    19: "девятнадцатом",
}

TENS_CARDINAL = {
    2: "двадцать",
    3: "тридцать",
    4: "сорок",
    5: "пятьдесят",
    6: "шестьдесят",
    7: "семьдесят",
    8: "восемьдесят",
    9: "девяносто",
}

TENS_ORDINAL = {
    2: "двадцатом",
    3: "тридцатом",
    4: "сороковом",
    5: "пятидесятом",
    6: "шестидесятом",
    7: "семидесятом",
    8: "восьмидесятом",
    9: "девяностом",
}


def normalize_for_tts(text: str) -> str:
    """Normalize generated answer text for Piper speech synthesis."""
    normalized = text
    normalized = _replace_emails(normalized)
    normalized = _replace_domains(normalized)
    normalized = _replace_years(normalized)
    normalized = _apply_manual_replacements(normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _replace_emails(text: str) -> str:
    local_part = r"[A-Za-z0-9]+(?:\s*[._%+\-]\s*[A-Za-z0-9]+)*"
    domain_part = r"(?:spbu(?:\s*\.\s*(?:ru)?)?|[A-Za-z0-9.-]+\.[A-Za-z]{2,})"
    pattern = re.compile(rf"\b({local_part})@({domain_part})", re.IGNORECASE)
    return pattern.sub(lambda match: _normalize_email(match.group(1), match.group(2)), text)


def _normalize_email(local: str, domain: str) -> str:
    return f"{_normalize_email_local(local)} собака {_normalize_email_domain(domain)}"


def _normalize_email_local(local: str) -> str:
    compact = re.sub(r"\s*([._%+\-])\s*", r"\1", local)
    parts = re.split(r"([._%+\-])", compact)
    spoken = []
    for part in parts:
        if not part:
            continue
        if part == ".":
            spoken.append("точка")
        elif part == "-":
            spoken.append("дефис")
        elif part == "_":
            spoken.append("нижнее подчёркивание")
        elif part in ("%","+"):
            spoken.append("плюс" if part == "+" else "процент")
        else:
            spoken.append(_normalize_latin_word(part))
    return " ".join(spoken)


def _normalize_latin_word(word: str) -> str:
    if len(word) <= 2 or not word.isalpha():
        return _spell_latin_token(word)

    transliterated = _transliterate_latin(word)
    return transliterated or _spell_latin_token(word)


def _normalize_email_domain(domain: str) -> str:
    cleaned = re.sub(r"\s+", "", domain).strip().rstrip(".").lower()
    if cleaned == "spbu":
        cleaned = "spbu.ru"
    return _normalize_domain(cleaned)


def _replace_domains(text: str) -> str:
    return re.sub(
        r"\bspbu\.(?:ru)?\b|\bspbu\.",
        lambda match: _normalize_domain("spbu.ru"),
        text,
        flags=re.IGNORECASE,
    )


def _normalize_domain(domain: str) -> str:
    parts = [part for part in domain.strip().rstrip(".").split(".") if part]
    if len(parts) == 1 and parts[0].lower() == "spbu":
        parts.append("ru")

    spoken_parts = []
    for part in parts:
        lower = part.lower()
        if lower == "spbu":
            spoken_parts.append("эс пэ бэ у")
        elif lower == "ru":
            spoken_parts.append("ру")
        else:
            spoken_parts.append(_normalize_latin_word(part))

    if len(spoken_parts) == 1:
        return spoken_parts[0]
    return f"{spoken_parts[0]} точка {' точка '.join(spoken_parts[1:])}"


def _replace_years(text: str) -> str:
    def repl(match):
        prefix = match.group("prefix")
        year = int(match.group("year"))
        spoken = _year_prepositional(year)
        if not spoken:
            return match.group(0)

        if not prefix:
            return f"{spoken} году"

        replacement_prefix = "В" if prefix[0].isupper() else "в"
        return f"{replacement_prefix} {spoken} году"

    return re.sub(
        r"\b(?:(?P<prefix>[Вв]|[Нн]а)\s+)?(?P<year>19\d{2}|20\d{2})\s+году\b",
        repl,
        text,
    )


def _year_prepositional(year: int) -> str | None:
    if year < 1900 or year > 2099:
        return None

    century = year // 100
    prefix = HUNDREDS_1900.get(century)
    if not prefix:
        return None

    rest = year % 100
    if rest == 0:
        return f"{prefix}овом" if century == 19 else "двухтысячном"
    if rest < 20:
        return f"{prefix} {ORDINALS_0_19[rest]}"

    tens = rest // 10
    ones = rest % 10
    if ones == 0:
        return f"{prefix} {TENS_ORDINAL[tens]}"
    return f"{prefix} {TENS_CARDINAL[tens]} {ORDINALS_0_19[ones]}"


def _apply_manual_replacements(text: str) -> str:
    result = text
    for source, target in MANUAL_REPLACEMENTS.items():
        result = result.replace(source, target)
    return result


def _transliterate_latin(word: str) -> str:
    source = word.lower()
    result = []
    idx = 0
    while idx < len(source):
        matched = False
        for latin, cyrillic in TRANSLIT_MULTI:
            if source.startswith(latin, idx):
                result.append(cyrillic)
                idx += len(latin)
                matched = True
                break
        if matched:
            continue

        char = source[idx]
        if char not in TRANSLIT_SINGLE:
            return ""
        result.append(TRANSLIT_SINGLE[char])
        idx += 1
    return "".join(result)


def _spell_latin_token(token: str) -> str:
    spoken = []
    for char in token:
        lower = char.lower()
        if lower in LATIN_LETTERS:
            spoken.append(LATIN_LETTERS[lower])
        elif char.isdigit():
            spoken.append(char)
        elif char == ".":
            spoken.append("точка")
        elif char == "-":
            spoken.append("дефис")
        elif char == "_":
            spoken.append("нижнее подчёркивание")
        elif char == "+":
            spoken.append("плюс")
    return " ".join(spoken)
