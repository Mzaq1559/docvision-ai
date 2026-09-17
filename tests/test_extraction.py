from app.extraction.entities import extract_entities


SAMPLE_TEXT = """
Invoice INV-20458
Date: 12/09/2026
Bill To: John Doe
Email: john.doe@example.com
Phone: +92 300 1234567
Website: https://example.com/invoice
CNIC: 12345-1234567-1
Amount Due: Rs. 4,500.00
"""


def test_extract_entities_finds_email():
    result = extract_entities(SAMPLE_TEXT)
    assert "john.doe@example.com" in result.emails


def test_extract_entities_finds_url():
    result = extract_entities(SAMPLE_TEXT)
    assert any("example.com/invoice" in u for u in result.urls)


def test_extract_entities_finds_date():
    result = extract_entities(SAMPLE_TEXT)
    assert any("12/09/2026" in d for d in result.dates)


def test_extract_entities_finds_identifier():
    result = extract_entities(SAMPLE_TEXT)
    assert any("12345-1234567-1" in i for i in result.identifiers)
    assert any("INV-20458" in i for i in result.identifiers)


def test_extract_entities_finds_phone():
    result = extract_entities(SAMPLE_TEXT)
    assert len(result.phone_numbers) >= 1


def test_extract_entities_empty_text_returns_empty_result():
    result = extract_entities("")
    assert result.total_fields == 0


def test_extract_entities_no_false_positive_on_plain_text():
    result = extract_entities("This is just a plain sentence with no data in it.")
    assert result.emails == []
    assert result.urls == []
