"""German spoken scripts (``tts.language: de``).

Voices read raw digits and the English symbol words ("degrees", "percent", "euros") with English
pronunciation inside German replies: "um 18:12 Uhr" was spoken as "eighteen twelve". German
scripts spell clock times, dates, money, units and numbers out as German words.
"""

import pytest

import hermes_yaml as yaml
from tools.tts_streaming import SentenceChunker
from tools.tts_text_normalize import _strip_markdown_for_tts, prepare_spoken_text
from tools.tts_text_normalize_de import german_cardinal


def _german(text: str) -> str:
    return prepare_spoken_text(text, language="de")


@pytest.mark.parametrize(("raw", "spoken"), [
    ("Das Meeting ist um 18:12 Uhr.", "Das Meeting ist um achtzehn Uhr zwölf."),
    ("Der Wecker klingelt um 1:05.", "Der Wecker klingelt um ein Uhr fünf."),
    ("Ab 18 Uhr ist Ruhe.", "Ab achtzehn Uhr ist Ruhe."),
    ("Abfahrt 7.45 Uhr", "Abfahrt sieben Uhr fünfundvierzig"),
    ("Geöffnet 9:00–17:30 Uhr.", "Geöffnet neun Uhr bis siebzehn Uhr dreißig."),
    ("Der Job startet um 18:12:30.", "Der Job startet um achtzehn Uhr zwölf und dreißig Sekunden."),
    ("Bayern hat 3:2 gewonnen.", "Bayern hat drei zu zwei gewonnen."),
])
def test_clock_times_are_spoken_as_german_times(raw, spoken):
    assert _german(raw) == spoken


@pytest.mark.parametrize(("raw", "spoken"), [
    ("Am 3. November 2026 kommt das Update.",
     "Am dritten November zweitausendsechsundzwanzig kommt das Update."),
    ("Der 1. Mai ist frei.", "Der erste Mai ist frei."),
    ("Freitag, 16.10.2026", "Freitag, sechzehnter Oktober zweitausendsechsundzwanzig"),
    ("Termin am 24.12. um 20 Uhr", "Termin am vierundzwanzigsten Dezember um zwanzig Uhr"),
    ("Das Paket kommt am 2.3.", "Das Paket kommt am zweiten März."),
    ("Release 2026-10-16", "Release sechzehnter Oktober zweitausendsechsundzwanzig"),
    ("Vom 16.–18. Oktober bin ich weg.", "Vom sechzehnten bis achtzehnten Oktober bin ich weg."),
    ("Er wurde 1999 geboren.", "Er wurde neunzehnhundertneunundneunzig geboren."),
])
def test_dates_take_german_ordinals_and_month_names(raw, spoken):
    assert _german(raw) == spoken


@pytest.mark.parametrize(("raw", "spoken"), [
    ("Bayern wurde zum 2. Mal Meister.", "Bayern wurde zum zweiten Mal Meister."),
    ("Der 3. Platz geht an Polaris.", "Der dritte Platz geht an Polaris."),
    ("Im 4. Quartal klappt es beim 101. Versuch.",
     "Im vierten Quartal klappt es beim einhundertersten Versuch."),
    ("Das ist ein 2. Versuch und eine 3. Chance.", "Das ist ein zweiter Versuch und eine dritte Chance."),
    ("Wir sind auf Platz 3. Danach kommt nichts.", "Wir sind auf Platz drei. Danach kommt nichts."),
])
def test_numbered_nouns_after_an_article_or_preposition_take_ordinals(raw, spoken):
    assert _german(raw) == spoken


@pytest.mark.parametrize(("raw", "spoken"), [
    ("Das kostet 12,50 €.", "Das kostet zwölf Euro fünfzig."),
    ("Strom kostet 0,32 €/kWh.", "Strom kostet zweiunddreißig Cent pro Kilowattstunde."),
    ("Nur 1 € pro Tag", "Nur ein Euro pro Tag"),
    ("Budget: 1.500 Euro", "Budget: eintausendfünfhundert Euro"),
    ("Morgen werden es 18 °C.", "Morgen werden es achtzehn Grad Celsius."),
    ("Zwischen -5 und 3 °C", "Zwischen minus fünf und drei Grad Celsius"),
    ("Etwa 12 % mehr", "Etwa zwölf Prozent mehr"),
    ("Die Fahrt dauert 45 min bei 120 km/h.",
     "Die Fahrt dauert fünfundvierzig Minuten bei einhundertzwanzig Kilometer pro Stunde."),
    ("Noch 1 h", "Noch eine Stunde"),
    ("Tom & Jerry, ~5 Folgen", "Tom und Jerry, etwa fünf Folgen"),
])
def test_money_units_and_symbols_are_spoken_as_german_words(raw, spoken):
    assert _german(raw) == spoken


@pytest.mark.parametrize(("raw", "spoken"), [
    ("Ein Wert von 3,5", "Ein Wert von drei Komma fünf"),
    ("Es kamen 2.500.000 Besucher.", "Es kamen zwei Millionen fünfhunderttausend Besucher."),
    ("Platz 1.", "Platz eins."),
    ("Seiten 10-12", "Seiten zehn bis zwölf"),
    ("Version 2.1.0 ist da.", "Version zwei Punkt eins Punkt null ist da."),
    ("Agent 007", "Agent null null sieben"),
    ("Danach 2x täglich.", "Danach zwei mal täglich."),
])
def test_numbers_are_spoken_as_german_words(raw, spoken):
    assert _german(raw) == spoken


@pytest.mark.parametrize(("number", "words"), [
    (0, "null"), (1, "eins"), (16, "sechzehn"), (17, "siebzehn"), (21, "einundzwanzig"),
    (30, "dreißig"), (99, "neunundneunzig"), (100, "einhundert"), (101, "einhunderteins"),
    (999, "neunhundertneunundneunzig"), (1000, "eintausend"), (2026, "zweitausendsechsundzwanzig"),
    (21_000, "einundzwanzigtausend"), (101_000, "einhunderteintausend"), (1_000_000, "eine Million"),
    (2_000_001, "zwei Millionen eins"), (3_000_000_000, "drei Milliarden"),
])
def test_german_cardinal_words(number, words):
    assert german_cardinal(number) == words


def test_identifier_tokens_are_silenced_before_numbers_are_spelled_out():
    assert _german("Das Modell Llama-3.3-70B läuft.") == "Das Modell läuft."


@pytest.mark.parametrize("raw", [
    "Das Meeting ist um 18:12 Uhr, es kostet 12,50 € und es werden 18 °C.",
    "Am 3. November 2026 sind es ~5 % mehr & 1.500 Besucher.",
])
def test_german_script_is_stable_when_normalized_twice(raw):
    # The gateway cleans a reply once for voice delivery and the TTS tool cleans it again.
    once = _german(raw)
    assert _german(once) == once


@pytest.mark.parametrize("language", ["de", "de-DE", "de_AT", "DE"])
def test_german_language_tags_select_german_speech(language):
    assert prepare_spoken_text("18 °C", language=language) == "achtzehn Grad Celsius"


@pytest.mark.parametrize("language", ["", "en", "en-US"])
def test_other_languages_keep_the_english_symbol_words(language):
    assert prepare_spoken_text("Um 18:12 Uhr sind es 18 °C.", language=language) == (
        "Um 18:12 Uhr sind es 18 degrees Celsius.")


def _streamed_sentences(reply: str, tts_config: dict) -> list[str]:
    chunker = SentenceChunker.from_config(tts_config)
    sentences = [sentence for token in reply.split(" ") for sentence in chunker.feed(token + " ")]
    return sentences + chunker.flush()


def test_streaming_keeps_a_german_date_in_one_spoken_sentence():
    # Streaming TTS cleans each sentence on its own; cut after "17." the month is gone.
    reply = "Heute ist Freitag, der 17. Oktober 2026. Um 18:12 Uhr ist der Termin am 21.10. um 9 Uhr."
    spoken = [prepare_spoken_text(s, language="de") for s in _streamed_sentences(reply, {"language": "de"})]

    assert " ".join(spoken) == ("Heute ist Freitag, der siebzehnte Oktober zweitausendsechsundzwanzig. "
                                "Um achtzehn Uhr zwölf ist der Termin am einundzwanzigsten Oktober um neun Uhr.")


def test_streaming_still_cuts_german_sentences_that_end_in_a_word():
    sentences = _streamed_sentences("Es sind heute 18 Grad. Morgen wird es wärmer.", {"language": "de"})

    assert [s.strip() for s in sentences] == ["Es sind heute 18 Grad.", "Morgen wird es wärmer."]


def test_configured_tts_language_drives_every_cleanup_entry_point(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"tts": {"language": "de"}}), encoding="utf-8")

    assert prepare_spoken_text("Um 18:12 Uhr sind es 18 °C.") == (
        "Um achtzehn Uhr zwölf sind es achtzehn Grad Celsius.")
    assert _strip_markdown_for_tts("**18:12 Uhr**") == "achtzehn Uhr zwölf"
