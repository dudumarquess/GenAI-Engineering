from app.services.classify_service import classify_text

VALID_LABELS = {"Question", "Complaint", "Request", "Other"}

def test_classify_returns_valid_label():
    result = classify_text("What is the capital of France?")
    
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "label" in result, "Result should contain 'label'"
    assert "reason" in result, "Result should contain 'reason'"
    assert result["label"] in VALID_LABELS, f"Label should be one of {VALID_LABELS}"
    
def test_classify_multiple_inputs():
    test_cases = [
        ("Can you tell me when the office opens?", "Question"),
        ("Your support team never replied.", "Complaint"),
        ("Please send me the invoice.", "Request"),
        ("The sky is very clear today.", "Other"),
    ]
    
    for text, expected_label in test_cases:
        result = classify_text(text)
        assert result["label"] == expected_label, f"Expected label '{expected_label}' but got '{result['label']}'"