"""
test_construction_detector.py — Unit tests for construction chapter detection.
"""
import pytest
from construction_detector import (
    is_construction_chapter,
    should_use_construction_prompt,
    validate_question_construction_type
)


def test_is_construction_chapter_assamese():
    """Test detection of construction chapters in Assamese."""
    # Explicit construction keywords
    assert is_construction_chapter("অংকন") == True
    assert is_construction_chapter("অঙ্কন") == True
    assert is_construction_chapter("জ্যামিতি অংকন") == True
    assert is_construction_chapter("ত্রিভুজ অংকন") == True
    
    # Non-construction chapters
    assert is_construction_chapter("ত্রিভুজ") == False
    assert is_construction_chapter("বৃত্ত") == False
    assert is_construction_chapter("স্থানাংক জ্যামিতি") == False


def test_is_construction_chapter_english():
    """Test detection of construction chapters in English."""
    # Explicit construction keywords
    assert is_construction_chapter("Constructions") == True
    assert is_construction_chapter("Geometric Constructions") == True
    assert is_construction_chapter("Triangle Construction") == True
    
    # Non-construction chapters
    assert is_construction_chapter("Triangles") == False
    assert is_construction_chapter("Circles") == False
    assert is_construction_chapter("Coordinate Geometry") == False


def test_is_construction_chapter_edge_cases():
    """Test edge cases for construction detection."""
    # Empty title
    assert is_construction_chapter("") == False
    assert is_construction_chapter(None) == False
    
    # Mixed titles
    assert is_construction_chapter("ত্রিভুজ অংকন") == True  # Has construction keyword
    assert is_construction_chapter("Triangles and Constructions") == True  # Has construction keyword


def test_should_use_construction_prompt():
    """Test construction prompt configuration."""
    # Construction chapter
    config = should_use_construction_prompt("অংকন", 11)
    assert config["is_construction"] == True
    assert config["force_instruments"] == True
    assert "construction" in config["reason"].lower()
    
    # Non-construction chapter
    config = should_use_construction_prompt("ত্রিভুজ", 7)
    assert config["is_construction"] == False
    assert config["force_instruments"] == False
    assert "not a construction" in config["reason"].lower()


def test_validate_question_construction_type_non_construction_chapter():
    """Test validation for non-construction chapters."""
    # Question marked as construction in non-construction chapter
    question = {
        "question_number": 1,
        "question_text": "Find the area of the triangle",
        "required": "Calculate the area",
        "construction_instruments": ["কম্পাছ", "স্কেল"],
        "diagram_spec": {"diagram_type": "construction"}
    }
    
    validation = validate_question_construction_type(question, chapter_is_construction=False)
    assert validation["is_valid"] == False
    assert validation["should_correct"] == True
    assert validation["correction_action"] == "remove_construction"
    
    # Question NOT marked as construction in non-construction chapter (valid)
    question2 = {
        "question_number": 2,
        "question_text": "Find the area of the triangle",
        "required": "Calculate the area",
        "construction_instruments": None,
        "diagram_spec": {"diagram_type": "triangle"}
    }
    
    validation2 = validate_question_construction_type(question2, chapter_is_construction=False)
    assert validation2["is_valid"] == True
    assert validation2["should_correct"] == False


def test_validate_question_construction_type_construction_chapter():
    """Test validation for construction chapters."""
    # Question asking for construction but not marked as such
    question = {
        "question_number": 1,
        "question_text": "অংকন কৰা: এটা ত্রিভুজ অংকন কৰা",
        "required": "তিনিটা বাহুৰ দৈর্ঘ্য দিয়া আছে",
        "construction_instruments": None,
        "diagram_spec": {"diagram_type": "triangle"}
    }
    
    validation = validate_question_construction_type(question, chapter_is_construction=True)
    assert validation["is_valid"] == False
    assert validation["should_correct"] == True
    assert validation["correction_action"] == "add_construction"
    
    # Question properly marked as construction in construction chapter (valid)
    question2 = {
        "question_number": 2,
        "question_text": "অংকন কৰা: এটা ত্রিভুজ অংকন কৰা",
        "required": "তিনিটা বাহুৰ দৈর্ঘ্য দিয়া আছে",
        "construction_instruments": ["কম্পাছ", "স্কেল"],
        "diagram_spec": {"diagram_type": "construction"}
    }
    
    validation2 = validate_question_construction_type(question2, chapter_is_construction=True)
    assert validation2["is_valid"] == True
    assert validation2["should_correct"] == False


def test_validate_question_with_actual_construction_keywords():
    """Test that actual construction keywords are detected."""
    # Question with construction verbs in non-construction chapter
    question = {
        "question_number": 1,
        "question_text": "Construct a triangle with sides 3, 4, 5",
        "required": "Draw using compass and ruler",
        "construction_instruments": ["কম্পাছ", "স্কেল"],
        "diagram_spec": {"diagram_type": "construction"}
    }
    
    # Even with construction verbs, if it's not a construction chapter,
    # and the question doesn't explicitly ask for construction in the text,
    # it should be flagged
    validation = validate_question_construction_type(question, chapter_is_construction=False)
    # This should be valid because the question explicitly asks to "construct"
    assert validation["is_valid"] == True  # Actually asks for construction


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
