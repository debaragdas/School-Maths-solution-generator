"""
test_construction_integration.py - Integration test for construction detection with real config.
"""
import sys
from construction_detector import is_construction_chapter, should_use_construction_prompt


def test_with_current_config():
    """Test construction detection with the current config.py settings."""
    try:
        import config
        print(f"Testing with config: Class {config.CLASS}, Chapter {config.CHAPTER}")
        
        # Simulate getting chapter title (in real run, this comes from chapter_detector)
        # For testing, we'll test both construction and non-construction scenarios
        
        test_cases = [
            ("অংকন", 11, True, "Construction chapter in Assamese"),
            ("Constructions", 11, True, "Construction chapter in English"),
            ("ত্রিভুজ অংকন", 11, True, "Triangle Construction (mixed)"),
            ("ত্রিভুজ", 7, False, "Triangles (non-construction)"),
            ("বৃত্ত", 9, False, "Circles (non-construction)"),
            ("জ্যামিতি অংকন", 10, True, "Geometric Construction"),
        ]
        
        print("\n=== Construction Detection Test Results ===")
        all_passed = True
        
        for title, chapter_num, expected, description in test_cases:
            result = is_construction_chapter(title, chapter_num)
            passed = result == expected
            all_passed = all_passed and passed
            
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status}: {description}")
            print(f"  Title: '{title}', Chapter: {chapter_num}")
            print(f"  Expected: {expected}, Got: {result}")
            
            if not passed:
                print(f"  ERROR: Mismatch detected!")
        
        print("\n=== Prompt Configuration Test ===")
        for title, chapter_num, expected, description in test_cases:
            config_result = should_use_construction_prompt(title, chapter_num)
            passed = config_result["is_construction"] == expected
            all_passed = all_passed and passed
            
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status}: {description}")
            print(f"  Force instruments: {config_result['force_instruments']}")
            print(f"  Reason: {config_result['reason']}")
        
        if all_passed:
            print("\n🎉 All tests passed!")
            return 0
        else:
            print("\n❌ Some tests failed!")
            return 1
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return 1


def test_question_validation():
    """Test question-level construction validation."""
    from construction_detector import validate_question_construction_type
    
    print("\n=== Question Validation Test ===")
    
    # Test case 1: Non-construction chapter, question marked as construction
    question1 = {
        "question_number": 1,
        "question_text": "Find the area of triangle ABC",
        "required": "Calculate the area",
        "construction_instruments": ["কম্পাছ", "স্কেল"],
        "diagram_spec": {"diagram_type": "construction"}
    }
    
    validation1 = validate_question_construction_type(question1, chapter_is_construction=False)
    print(f"Test 1 (non-construction chapter, construction question):")
    print(f"  Valid: {validation1['is_valid']}")
    print(f"  Should correct: {validation1['should_correct']}")
    print(f"  Action: {validation1['correction_action']}")
    print(f"  Reason: {validation1['reason']}")
    
    # Test case 2: Construction chapter, question not marked as construction
    question2 = {
        "question_number": 2,
        "question_text": "অংকন কৰা: এটা ত্রিভুজ অংকন কৰা",
        "required": "তিনিটা বাহুৰ দৈর্ঘ্য দিয়া আছে",
        "construction_instruments": None,
        "diagram_spec": {"diagram_type": "triangle"}
    }
    
    validation2 = validate_question_construction_type(question2, chapter_is_construction=True)
    print(f"\nTest 2 (construction chapter, non-construction question):")
    print(f"  Valid: {validation2['is_valid']}")
    print(f"  Should correct: {validation2['should_correct']}")
    print(f"  Action: {validation2['correction_action']}")
    print(f"  Reason: {validation2['reason']}")
    
    # Test case 3: Valid construction question in construction chapter
    question3 = {
        "question_number": 3,
        "question_text": "অংকন কৰা: এটা ত্রিভুজ অংকন কৰা",
        "required": "তিনিটা বাহুৰ দৈর্ঘ্য দিয়া আছে",
        "construction_instruments": ["কম্পাছ", "স্কেল"],
        "diagram_spec": {"diagram_type": "construction"}
    }
    
    validation3 = validate_question_construction_type(question3, chapter_is_construction=True)
    print(f"\nTest 3 (construction chapter, construction question - valid):")
    print(f"  Valid: {validation3['is_valid']}")
    print(f"  Should correct: {validation3['should_correct']}")
    print(f"  Action: {validation3['correction_action']}")
    print(f"  Reason: {validation3['reason']}")
    
    all_valid = (not validation1['is_valid'] and validation1['should_correct'] and
                 not validation2['is_valid'] and validation2['should_correct'] and
                 validation3['is_valid'] and not validation3['should_correct'])
    
    if all_valid:
        print("\n🎉 All validation tests passed!")
        return 0
    else:
        print("\n❌ Some validation tests failed!")
        return 1


if __name__ == "__main__":
    print("=" * 60)
    print("Construction Detection Integration Test")
    print("=" * 60)
    
    result1 = test_with_current_config()
    result2 = test_question_validation()
    
    print("\n" + "=" * 60)
    if result1 == 0 and result2 == 0:
        print("✅ All integration tests passed!")
        sys.exit(0)
    else:
        print("❌ Some integration tests failed!")
        sys.exit(1)
