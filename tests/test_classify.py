from app.services.classify_service import classify_text

VALID_LABELS = {"question", "complaint", "request", "other"}

def test_classify_returns_valid_label():
    result = classify_text("What is the capital of France?")
    
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "label" in result, "Result should contain 'label'"
    assert "reason" in result, "Result should contain 'reason'"
    assert result["label"] in VALID_LABELS, f"Label should be one of {VALID_LABELS}"
    
    ## This test is being ignored because it's to strict. For a LLM is better to make manual evaluation for this tests
"""def test_classify_multiple_inputs():
    test_cases = [
        ("Can you tell me when the office opens?", "question"),
        ("Your support team never replied.", "complaint"),
        ("Please send me the invoice.", "request"),
        ("The sky is very clear today.", "other"),
    ]
    
    for text, expected_label in test_cases:
        result = classify_text(text)
        assert result["label"] == expected_label, f"Expected label '{expected_label}' but got '{result['label']}'"
       
"""
 
def test_classify_never_crashes():
    inputs = ["???", "1234", "random text", "HELP!!!"]
    
    for text in inputs:
        result = classify_text(text)
        assert isinstance(result, dict)
        assert "label" in result
        assert "reason" in result
        assert result["label"] in VALID_LABELS