"""Tests for preprocessing utilities."""

import pytest

from sentiment_toolkit.preprocessing import read_csv_texts, split_sentences


def test_split_sentences_ignores_blank_lines():
    text = "Great product\n\nBad support\n  \n"
    assert split_sentences(text) == ["Great product", "Bad support"]


def test_read_csv_texts_uses_named_column():
    content = "text,score\nI love it,5\nI hate it,1\n"
    records = read_csv_texts(content)
    assert len(records) == 2
    assert records[0].text == "I love it"


def test_read_csv_texts_falls_back_to_first_column():
    content = "review\nWorks well\n"
    records = read_csv_texts(content, text_column="text")
    assert records[0].text == "Works well"


def test_read_csv_texts_raises_on_empty_values():
    with pytest.raises(ValueError, match="No non-empty values"):
        read_csv_texts("text\n\n")
