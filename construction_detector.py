"""
construction_detector.py — Smart detection of construction chapters.

Determines whether a chapter is a "Constructions" (অংকন) chapter that requires
step-by-step drawing instructions, versus a regular geometry chapter that needs
only mathematical solutions. This prevents the AI from mistakenly adding
construction instructions to non-construction chapters.
"""
import re
from utils import logger

# Keywords that indicate a construction chapter in Assamese/Bengali
_CONSTRUCTION_KEYWORDS_ASSAMESE = [
    "অংকন", "অঙ্কন",  # Construction
    "ৰেখাখণ্ড", "সৰলৰেখা",  # Line segment, straight line
    "কোণ", "কোণৰ সমদ্বিখণ্ডক",  # Angle, angle bisector
    "ত্রিভুজ অংকন", "চতুর্ভুজ অংকন",  # Triangle construction, quadrilateral construction
    "জ্যামিতি অংকন",  # Geometric construction
]

# Keywords that indicate a construction chapter in English
_CONSTRUCTION_KEYWORDS_ENGLISH = [
    "construction", "constructions",
    "geometric construction",
    "line segment", "angle bisector",
    "triangle construction", "quadrilateral construction",
]

# Regular chapters that should NEVER be treated as construction chapters
_NON_CONSTRUCTION_CHAPTERS = [
    "ত্রিভুজ", "triangle",  # Triangle (proofs/properties, not construction)
    "বৃত্ত", "circle",  # Circle (theorems, not construction)
    "স্থানাংক জ্যামিতি", "coordinate geometry",  # Coordinate geometry
    "ত্রিকোণমিতি", "trigonometry",  # Trigonometry
    "পৰিসংখ্যা", "statistics",  # Statistics
    "বীজগণিত", "algebra",  # Algebra
]


def is_construction_chapter(chapter_title: str, chapter_number: int = None) -> bool:
    """
    Determines if a chapter is a construction chapter based on its title.
    
    Args:
        chapter_title: The title of the chapter (e.g., "অংকন", "Constructions")
        chapter_number: Optional chapter number for additional heuristics
    
    Returns:
        True if this is a construction chapter, False otherwise
    """
    if not chapter_title:
        return False
    
    title_lower = chapter_title.lower()
    
    # First check: explicit construction keywords (highest priority)
    for keyword in _CONSTRUCTION_KEYWORDS_ASSAMESE + _CONSTRUCTION_KEYWORDS_ENGLISH:
        if keyword.lower() in title_lower:
            # If it has a construction keyword, it's a construction chapter
            # UNLESS it's clearly a non-construction topic that happens to contain the word
            # Example: "ত্রিভুজ" (Triangle) vs "ত্রিভুজ অংকন" (Triangle Construction)
            # The latter should be construction, the former should not
            
            # Check if this is a pure non-construction topic (no construction word present)
            # If the title is EXACTLY a non-construction keyword, reject it
            if title_lower.strip() in [nc.lower() for nc in _NON_CONSTRUCTION_CHAPTERS]:
                logger.info(f"🔍 Chapter '{chapter_title}' is exactly a non-construction topic")
                return False
            
            # If it has BOTH a construction keyword AND a non-construction keyword,
            # the construction keyword wins (e.g., "ত্রিভুজ অংকন" = Triangle Construction)
            logger.info(f"🔍 Chapter '{chapter_title}' identified as a CONSTRUCTION chapter")
            return True
    
    # Second check: common construction chapter numbers (varies by board)
    # SEBA Class 9: Chapter 11 is typically Constructions
    # SEBA Class 10: Chapter 10 or 11 is typically Constructions
    if chapter_number:
        if chapter_number in [10, 11]:
            # Additional check: if the title doesn't explicitly say "construction",
            # be conservative and don't treat it as construction
            logger.info(f"🔍 Chapter {chapter_number} ('{chapter_title}') - not explicitly a construction chapter")
            return False
    
    logger.info(f"🔍 Chapter '{chapter_title}' - NOT a construction chapter")
    return False


def should_use_construction_prompt(chapter_title: str, chapter_number: int = None,
                                   class_level: int = None) -> dict:
    """
    Returns configuration for whether to use construction-specific prompting.
    
    Args:
        chapter_title: The chapter title
        chapter_number: Optional chapter number
        class_level: Optional class level (9, 10, etc.) for board-specific rules
    
    Returns:
        dict with keys:
        - is_construction: bool
        - reason: str explaining the decision
        - force_instruments: bool (whether to require construction_instruments)
    """
    is_const = is_construction_chapter(chapter_title, chapter_number)
    
    if is_const:
        return {
            "is_construction": True,
            "reason": f"Chapter '{chapter_title}' is identified as a construction chapter",
            "force_instruments": True,  # Require construction_instruments for all questions
        }
    
    # Even for non-construction chapters, some individual questions might be constructions
    # The AI will still detect these at question level, but we don't force it
    return {
        "is_construction": False,
        "reason": f"Chapter '{chapter_title}' is not a construction chapter",
        "force_instruments": False,  # Let AI decide per question
    }


def validate_question_construction_type(question: dict, chapter_is_construction: bool) -> dict:
    """
    Validates that a question's construction classification matches the chapter type.
    
    This catches cases where the AI mistakenly marks a question as a construction
    question in a non-construction chapter (or vice versa).
    
    Args:
        question: The question dict with construction_instruments and diagram_spec
        chapter_is_construction: Whether the chapter is a construction chapter
    
    Returns:
        dict with keys:
        - is_valid: bool
        - should_correct: bool (whether to auto-correct)
        - correction_action: str or None
        - reason: str
    """
    has_instruments = question.get("construction_instruments") is not None
    diagram_spec = question.get("diagram_spec", {})
    diagram_type = diagram_spec.get("diagram_type") if isinstance(diagram_spec, dict) else None
    
    is_construction_question = has_instruments or diagram_type == "construction"
    
    # Case 1: Non-construction chapter, but question marked as construction
    if not chapter_is_construction and is_construction_question:
        # Check if the question text actually asks for construction
        question_text = f"{question.get('question_text', '')} {question.get('required', '')}".lower()
        construction_verbs = ["অংকন কৰা", "নিৰ্মাণ কৰা", "construct", "draw using", "draw with compass"]
        
        actually_construction = any(verb in question_text for verb in construction_verbs)
        
        if not actually_construction:
            return {
                "is_valid": False,
                "should_correct": True,
                "correction_action": "remove_construction",
                "reason": f"Question marked as construction but doesn't ask for construction in non-construction chapter"
            }
    
    # Case 2: Construction chapter, but question not marked as construction
    if chapter_is_construction and not is_construction_question:
        question_text = f"{question.get('question_text', '')} {question.get('required', '')}".lower()
        construction_verbs = ["অংকন কৰা", "নিৰ্মাণ কৰা", "construct", "draw using", "draw with compass"]
        
        actually_construction = any(verb in question_text for verb in construction_verbs)
        
        if actually_construction:
            return {
                "is_valid": False,
                "should_correct": True,
                "correction_action": "add_construction",
                "reason": f"Question asks for construction but not marked as such in construction chapter"
            }
    
    return {
        "is_valid": True,
        "should_correct": False,
        "correction_action": None,
        "reason": "Question construction type matches chapter type"
    }
