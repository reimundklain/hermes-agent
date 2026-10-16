"""German spoken-script rules behind ``tts.language: de`` (used by ``tools.tts_text_normalize``).

Voices read raw digits and English unit words with English pronunciation inside German replies
("um 18:12 Uhr" became "eighteen twelve"), so German scripts spell clock times, dates, money,
units and numbers out as German words. ``expand_german_symbols`` replaces the English symbol
words; ``expand_german_numbers`` runs after identifier pruning, so a silenced model ID never
half-survives as number words.
"""

from __future__ import annotations

import re
from typing import Callable

_ONES = ("null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun", "zehn",
         "elf", "zwölf", "dreizehn", "vierzehn", "fünfzehn", "sechzehn", "siebzehn", "achtzehn",
         "neunzehn")
_TENS = ("", "", "zwanzig", "dreißig", "vierzig", "fünfzig", "sechzig", "siebzig", "achtzig", "neunzig")
_SCALES = ((10**9, "Milliarde", "Milliarden"), (10**6, "Million", "Millionen"))
_LARGEST_SPOKEN_NUMBER = 10**12 - 1
# Four-digit numbers in this range are spoken like years ("neunzehnhundertneunundneunzig").
_HUNDREDS_STYLE_RANGE = range(1100, 2000)

_IRREGULAR_ORDINAL_STEMS = {1: "erst", 3: "dritt", 7: "siebt", 8: "acht"}
# An ordinal's ending follows the word before it ("am dritten", "der dritte", "ein dritter").
# A date without one takes "-er" ("Freitag, dritter November"); a numbered noun without one
# stays a cardinal ("Platz 3.").
_ORDINAL_ENDINGS_AFTER = {
    **dict.fromkeys(("am", "vom", "zum", "zur", "beim", "im", "dem", "den", "des", "seit", "ab", "bis",
                     "einen", "einem", "einer"), "en"),
    **dict.fromkeys(("der", "die", "das", "eine"), "e"),
    "ein": "er",
}

_MONTHS = ("Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September",
           "Oktober", "November", "Dezember")
_MONTH_ABBREVIATIONS = {"Jan": "Januar", "Feb": "Februar", "Mär": "März", "Apr": "April", "Jun": "Juni",
                        "Jul": "Juli", "Aug": "August", "Sept": "September", "Sep": "September",
                        "Okt": "Oktober", "Nov": "November", "Dez": "Dezember"}
_MONTH_NAME_PATTERN = "|".join(sorted((*_MONTHS, *_MONTH_ABBREVIATIONS), key=len, reverse=True))

# "1" before one of these nouns is "eine", before any other capitalized noun "ein".
_FEMININE_NOUNS = frozenset({"Million", "Milliarde", "Stunde", "Minute", "Sekunde", "Kilowattstunde",
                             "Woche", "Nacht", "Person", "Seite", "Nachricht", "Datei", "Frage"})

# Unit after a number -> (singular, plural); case-sensitive so "5G" stays a network generation.
# "km/h" precedes "km", which would otherwise leave "Kilometer/h".
_UNITS = (
    (r"km\s*/\s*h", "Kilometer pro Stunde", "Kilometer pro Stunde"),
    (r"kWh", "Kilowattstunde", "Kilowattstunden"), (r"kW", "Kilowatt", "Kilowatt"),
    (r"km", "Kilometer", "Kilometer"), (r"cm", "Zentimeter", "Zentimeter"),
    (r"mm", "Millimeter", "Millimeter"), (r"m", "Meter", "Meter"),
    (r"kg", "Kilogramm", "Kilogramm"), (r"g", "Gramm", "Gramm"), (r"l", "Liter", "Liter"),
    (r"min", "Minute", "Minuten"), (r"Std\.", "Stunde", "Stunden"), (r"h", "Stunde", "Stunden"),
    (r"Sek\.", "Sekunde", "Sekunden"), (r"s", "Sekunde", "Sekunden"),
    (r"TB", "Terabyte", "Terabyte"), (r"GB", "Gigabyte", "Gigabyte"), (r"MB", "Megabyte", "Megabyte"),
    (r"Mio\.", "Million", "Millionen"), (r"Mrd\.", "Milliarde", "Milliarden"),
    (r"%", "Prozent", "Prozent"),
)
_UNIT_RULES = tuple(
    (re.compile(r"(?<![\w.,])(\d+(?:[.,]\d+)*)\s*" + pattern + r"(?!\w)"),
     re.compile(r"(?<=[\w€$£%])\s*/\s*" + pattern + r"(?!\w)"), singular, plural)
    for pattern, singular, plural in _UNITS)
_SYMBOL_WORDS = (("&", " und "), ("≈", " etwa "), ("→", " nach "), ("⇒", " nach "))

# (symbols and codes written before or after an amount, words written after it, major, minor)
_CURRENCIES = (
    (("€", "EUR"), ("Euro",), "Euro", "Cent"),
    (("US$", "$", "USD"), ("Dollar",), "Dollar", "Cent"),
    (("£", "GBP"), ("Pfund",), "Pfund", "Pence"),
    (("CHF",), ("Franken",), "Franken", "Rappen"),
)
_AMOUNT = r"(\d{1,3}(?:\.\d{3})+|\d+)(?:[.,](\d{1,2}|-{1,2}))?"


def is_german_speech(language: str) -> bool:
    """True for a German ``tts.language`` tag ("de", "de-DE", "de_AT")."""
    return re.split(r"[-_]", language.strip().lower(), maxsplit=1)[0] == "de"


def _below_hundred(number: int) -> str:
    if number < 20:
        return _ONES[number]
    tens, unit = divmod(number, 10)
    return f"{_compound_prefix(unit)}und{_TENS[tens]}" if unit else _TENS[tens]


def _below_thousand(number: int) -> str:
    hundreds, rest = divmod(number, 100)
    head = f"{_compound_prefix(hundreds)}hundert" if hundreds else ""
    return head + (_below_hundred(rest) if rest or not hundreds else "")


def _compound_prefix(number: int) -> str:
    """``number`` (1-999) as the head of a compound: a trailing "eins" becomes "ein"."""
    words = _below_thousand(number)
    return words[:-1] if words.endswith("eins") else words


def german_cardinal(number: int) -> str:
    """German cardinal words for 0 <= ``number`` <= 999,999,999,999."""
    if not 0 <= number <= _LARGEST_SPOKEN_NUMBER:
        raise ValueError(f"no German cardinal for {number}")
    for scale, singular, plural in _SCALES:
        if number >= scale:
            count, rest = divmod(number, scale)
            head = f"eine {singular}" if count == 1 else f"{german_cardinal(count)} {plural}"
            return f"{head} {german_cardinal(rest)}" if rest else head
    thousands, rest = divmod(number, 1000)
    if not thousands:
        return _below_thousand(number)
    return f"{_compound_prefix(thousands)}tausend" + (_below_thousand(rest) if rest else "")


def german_ordinal(number: int, ending: str) -> str:
    """German ordinal for 1 <= ``number`` <= 999 with its inflection ``ending`` ("e", "en", "er")."""
    hundreds, rest = divmod(number, 100)
    head = german_cardinal(hundreds * 100) if hundreds else ""
    if not rest:
        return f"{head}st{ending}"
    if rest < 20:
        return head + (_IRREGULAR_ORDINAL_STEMS.get(rest) or f"{_ONES[rest]}t") + ending
    return f"{head}{_below_hundred(rest)}st{ending}"


def _year_words(year: int) -> str:
    if year in _HUNDREDS_STYLE_RANGE:
        century, rest = divmod(year, 100)
        return f"{german_cardinal(century)}hundert" + (_below_hundred(rest) if rest else "")
    return german_cardinal(year)


def _digit_words(digits: str) -> str:
    return " ".join(_ONES[int(digit)] for digit in digits if digit.isdigit())


def _spoken_integer(digits: str) -> str:
    """Integer words; leading zeros ("007") and out-of-range values are read digit by digit."""
    value = int(digits.replace(".", ""))
    if (len(digits) > 1 and digits.startswith("0")) or value > _LARGEST_SPOKEN_NUMBER:
        return _digit_words(digits)
    return german_cardinal(value)


def _lead_word(text_before: str) -> str:
    lead = re.search(r"(\w+)\s+$", text_before)
    return lead.group(1).lower() if lead else ""


def _ordinal_ending(text_before: str) -> str:
    """Ending of a day ordinal in a date."""
    return _ORDINAL_ENDINGS_AFTER.get(_lead_word(text_before), "er")


def _sentence_dot(text_after: str) -> str:
    """The date's closing dot doubles as a full stop at the end of a sentence; keep it there."""
    return "." if re.match(r"\s*$|\s+[A-ZÄÖÜ]", text_after) else ""


def _spoken_date(match: re.Match[str], day: int, month: int) -> str | None:
    if not (1 <= day <= 31 and 1 <= month <= 12):
        return None
    return f"{german_ordinal(day, _ordinal_ending(match.string[:match.start()]))} {_MONTHS[month - 1]}"


def _speak_iso_date(match: re.Match[str]) -> str:
    spoken = _spoken_date(match, int(match.group(3)), int(match.group(2)))
    return f"{spoken} {_year_words(int(match.group(1)))}" if spoken else match.group(0)


def _speak_numeric_date(match: re.Match[str]) -> str:
    spoken = _spoken_date(match, int(match.group(1)), int(match.group(2)))
    if spoken is None:
        return match.group(0)
    if match.group(3):
        return f"{spoken} {_year_words(int(match.group(3)))}"
    return spoken + _sentence_dot(match.string[match.end():])


def _speak_day_and_month_name(match: re.Match[str]) -> str:
    month = _MONTH_ABBREVIATIONS.get(match.group(2), match.group(2))
    spoken = _spoken_date(match, int(match.group(1)), _MONTHS.index(month) + 1)
    if spoken is None:
        return match.group(0)
    return spoken + (_sentence_dot(match.string[match.end():]) if match.group(3) else "")


def _speak_ordinal_before_noun(match: re.Match[str]) -> str:
    """"der 3. Platz", "zum 2. Mal": a numbered noun after an article or preposition."""
    ending = _ORDINAL_ENDINGS_AFTER.get(_lead_word(match.string[:match.start()]))
    return german_ordinal(int(match.group(1)), ending) if ending else match.group(0)


def _speak_clock(match: re.Match[str]) -> str:
    hour, minute = int(match.group("hour")), int(match.group("minute"))
    second = int(match.groupdict().get("second") or 0)
    spoken = f"{'ein' if hour == 1 else german_cardinal(hour)} Uhr"
    if minute:
        spoken += f" {german_cardinal(minute)}"
    if second:
        spoken += " und eine Sekunde" if second == 1 else f" und {german_cardinal(second)} Sekunden"
    return spoken


def _speak_first_day_of_range(match: re.Match[str]) -> str:
    """"16.–18. Oktober": the first day becomes "sechzehnten bis"; the date rules speak the rest."""
    day = int(match.group(1))
    if not 1 <= day <= 31:
        return match.group(0)
    return f"{german_ordinal(day, _ordinal_ending(match.string[:match.start()]))} bis "


def _speak_money(major: str, minor: str) -> Callable[[re.Match[str]], str]:
    def speak(match: re.Match[str]) -> str:
        units = int(match.group(1).replace(".", ""))
        fraction = match.group(2) or ""
        cents = int(fraction.ljust(2, "0")) if fraction.isdigit() else 0
        if units > _LARGEST_SPOKEN_NUMBER:
            return match.group(0)
        if cents and not units:
            return f"{german_cardinal(cents)} {minor}"
        spoken = f"{'ein' if units == 1 else german_cardinal(units)} {major}"
        return f"{spoken} {german_cardinal(cents)}" if cents else spoken
    return speak


def _one_before(text_after: str) -> str:
    noun = re.match(r"\s+([A-ZÄÖÜ]\w*)", text_after)
    if noun is None:
        return "eins"
    return "eine" if noun.group(1) in _FEMININE_NOUNS else "ein"


def _speak_number(match: re.Match[str]) -> str:
    integer, fraction = match.group(1), match.group(2)
    if fraction is not None:
        return f"{_spoken_integer(integer)} Komma {_digit_words(fraction)}"
    if integer == "1":
        return _one_before(match.string[match.end():])
    if len(integer) == 4 and int(integer) in _HUNDREDS_STYLE_RANGE:
        return _year_words(int(integer))
    return _spoken_integer(integer)


def _speak_dotted_number(match: re.Match[str]) -> str:
    return " Punkt ".join(_spoken_integer(part) for part in match.group(0).split("."))


def _money_rules() -> list[tuple[re.Pattern[str], Callable[[re.Match[str]], str]]]:
    rules = []
    for symbols, words, major, minor in _CURRENCIES:
        written_before = "|".join(re.escape(symbol) for symbol in symbols)
        written_after = "|".join(re.escape(symbol) for symbol in (*symbols, *words))
        speak = _speak_money(major, minor)
        rules.append((re.compile(r"(?<![\w.,])" + _AMOUNT + r"\s*(?:" + written_after + r")(?!\w)"), speak))
        rules.append((re.compile(r"(?<!\w)(?:" + written_before + r")\s*" + _AMOUNT + r"(?!\d|[.,]\d)"), speak))
    return rules


# Order matters: dates and clock times claim their digits before ranges, money and plain numbers.
_NUMBER_RULES: tuple[tuple[re.Pattern[str], Callable[[re.Match[str]], str] | str], ...] = (
    (re.compile(r"(?<![\w.-])(\d{4})-(\d{2})-(\d{2})(?![\w-])"), _speak_iso_date),
    (re.compile(r"(?<=\d)\s*[-–—]\s*(?=\d)"), " bis "),
    (re.compile(r"(?<![\w.:,])(?P<hour>[01]?\d|2[0-4]):(?P<minute>[0-5]\d)(?::(?P<second>[0-5]\d))?(?![\d:])"
                r"(?:\s*Uhr(?!\w))?"), _speak_clock),
    (re.compile(r"(?<![\w.:,])(?P<hour>[01]?\d|2[0-4])\.(?P<minute>[0-5]\d)\s*Uhr(?!\w)"), _speak_clock),
    (re.compile(r"(?<![\w.:,])(\d+):(\d+)(?![\d:])"), r"\1 zu \2"),
    (re.compile(r"(?<=\d)\s*[x×]\s*(?=\d|\s|$)"), " mal "),
    (re.compile(r"(?<![\w.])(\d{1,2})\.\s*[-–—]\s*(?=\d{1,2}\.(?:\d{1,2}\.|\s*(?:" + _MONTH_NAME_PATTERN + r")))"),
     _speak_first_day_of_range),
    (re.compile(r"(?<![\w.])(\d{1,2})\.(\d{1,2})\.(\d{4}(?!\d)|\d{2}(?!\d))?(?!\d)"), _speak_numeric_date),
    (re.compile(r"(?<![\w.])(\d{1,2})\.\s*(" + _MONTH_NAME_PATTERN + r")(\.)?(?!\w)"),
     _speak_day_and_month_name),
    (re.compile(r"(?<![\w.])([1-9]\d{0,2})\.(?=\s+[A-ZÄÖÜ])"), _speak_ordinal_before_noun),
    *_money_rules(),
    (re.compile(r"(?<![\w.,%)])-(?=\d)"), "minus "),
    (re.compile(r"(?<![\w.,])(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d+))?(?!\w|[.,]\d)"), _speak_number),
    (re.compile(r"(?<![\w.,])\d+(?:\.\d+)+(?!\w|[.,]\d)"), _speak_dotted_number),
)
_LEFTOVER_CURRENCY_WORDS = (("US$", " Dollar "), ("€", " Euro "), ("$", " Dollar "), ("£", " Pfund "))


def expand_german_symbols(text: str) -> str:
    """Degrees, units, rates and operator symbols as German words; digits stay for
    ``expand_german_numbers``."""
    for unit, word in (("C", "Celsius"), ("F", "Fahrenheit")):
        text = re.sub(r"°\s*" + unit + r"(?!\w)", f" Grad {word}", text)
    text = text.replace("°", " Grad ")
    for after_number, after_slash, singular, plural in _UNIT_RULES:
        text = after_number.sub(lambda m, s=singular, p=plural: f"{m.group(1)} {s if m.group(1) == '1' else p}",
                                text)
        text = after_slash.sub(f" pro {singular}", text)
    text = re.sub(r"(?<=\d)\s*/\s*(?=[A-Za-zÄÖÜäöü])", " pro ", text)
    text = re.sub(r"~\s*(?=\d)", "etwa ", text)
    for symbol, word in _SYMBOL_WORDS:
        text = text.replace(symbol, word)
    return re.sub("[•◦▪▫]", " ", text)


def expand_german_numbers(text: str) -> str:
    """Clock times, dates, money, ranges and numbers as German words."""
    for pattern, speak in _NUMBER_RULES:
        text = pattern.sub(speak, text)
    for symbol, word in _LEFTOVER_CURRENCY_WORDS:
        text = text.replace(symbol, word)
    return text
