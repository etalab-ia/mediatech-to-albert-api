"""Tests for HuggingFaceSource metadata extraction."""

from src.huggingface_source import HuggingFaceSource


def make_source():
    return HuggingFaceSource()  # no token needed for unit tests


def test_extract_metadata_filters_null_and_empty():
    source = make_source()
    row = {"category": "ARRETE", "ministry": "", "status": None, "start_date": "2007-01-01"}
    meta = source._extract_metadata(row, "AgentPublic/legi")
    assert meta["category"] == "ARRETE"
    assert meta["start_date"] == "2007-01-01"
    assert "ministry" not in meta  # empty string filtered out
    assert "status" not in meta  # None filtered out


def test_extract_metadata_truncates_long_strings():
    source = make_source()
    long_value = "x" * 300
    row = {"category": long_value, "start_date": "2007-01-01"}
    meta = source._extract_metadata(row, "AgentPublic/legi")
    assert len(meta["category"]) == 255
    assert meta["category"].endswith("...")


def test_get_document_name_uses_dataset_specific_field():
    source = make_source()

    row = {"name": "Admin Name", "full_title": "Full Legal Title"}

    # local-administrations uses "name"
    assert source._get_document_name(row, "AgentPublic/local-administrations-directory") == "Admin Name"
    # legi uses "full_title"
    assert source._get_document_name(row, "AgentPublic/legi") == "Full Legal Title"


def test_get_document_name_returns_none_if_field_missing():
    source = make_source()
    assert source._get_document_name({}, "AgentPublic/legi") is None
    assert source._get_document_name({"title": ""}, "AgentPublic/legi") is None


def test_extract_metadata_converts_list_to_str():
    source = make_source()
    row = {"name": "Admin", "types": ["type1", "type2"], "directory_url": "http://example.com"}
    meta = source._extract_metadata(row, "AgentPublic/local-administrations-directory")
    assert meta["types"] == "['type1', 'type2']"


def test_extract_metadata_truncates_list_converted_to_long_str():
    source = make_source()
    row = {"name": "Admin", "types": ["x" * 200, "y" * 200], "directory_url": "http://example.com"}
    meta = source._extract_metadata(row, "AgentPublic/local-administrations-directory")
    assert len(meta["types"]) == 255
    assert meta["types"].endswith("...")
