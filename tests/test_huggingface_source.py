"""Tests for HuggingFaceSource metadata extraction."""

from src.huggingface_source import HuggingFaceSource


def make_source():
    return HuggingFaceSource()  # no token needed for unit tests


def test_extract_metadata_filters_null_and_empty():
    source = make_source()
    row = {"title": "Test Title", "url": "", "nature": None, "category": "Law"}
    meta = source._extract_metadata(row, "AgentPublic/legi")
    assert meta["title"] == "Test Title"
    assert meta["category"] == "Law"
    assert "url" not in meta  # empty string filtered out
    assert "nature" not in meta  # None filtered out


def test_extract_metadata_truncates_long_strings():
    source = make_source()
    long_value = "x" * 300
    row = {"title": long_value, "url": "http://example.com"}
    meta = source._extract_metadata(row, "AgentPublic/legi")
    assert len(meta["title"]) == 255
    assert meta["title"].endswith("...")


def test_get_document_name_uses_dataset_specific_field():
    source = make_source()

    row = {"name": "Admin Name", "title": "Some Title"}

    # local-administrations uses "name"
    assert source._get_document_name(row, "AgentPublic/local-administrations-directory") == "Admin Name"
    # legi uses "title"
    assert source._get_document_name(row, "AgentPublic/legi") == "Some Title"


def test_get_document_name_returns_none_if_field_missing():
    source = make_source()
    assert source._get_document_name({}, "AgentPublic/legi") is None
    assert source._get_document_name({"title": ""}, "AgentPublic/legi") is None
