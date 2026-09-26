"""
batch_correction.py — Batch correction and republishing workflow.

Allows correcting multiple questions across exercises without regenerating
the entire chapter. Uses the existing correction_engine.py functions but
adds batch processing and automatic republishing.
"""
import os
import json
from typing import List, Dict
from correction_engine import regenerate_question, attach_manual_figure_to_question
from review_state import load_state, approve, STATUS_APPROVED, chapter_publish_status, publish_chapter
from utils import logger


def apply_corrections_from_file(corrections_file: str, output_dir: str = "output") -> Dict:
    """
    Applies multiple corrections from a JSON file without regenerating chapters.
    
    The corrections file should have this format:
    {
        "corrections": [
            {
                "exercise_label": "7.1",
                "question_number": 1,
                "sub_part": null,
                "correction_instruction": "Fix the calculation in step 3",
                "reviewer": "admin"
            },
            ...
        ]
    }
    
    Args:
        corrections_file: Path to JSON file with corrections
        output_dir: Output directory containing the exercise PDFs
    
    Returns:
        Dict with summary of applied corrections
    """
    with open(corrections_file, 'r', encoding='utf-8') as f:
        corrections_data = json.load(f)
    
    corrections = corrections_data.get("corrections", [])
    results = {
        "total": len(corrections),
        "success": 0,
        "failed": 0,
        "errors": []
    }
    
    for correction in corrections:
        exercise_label = correction.get("exercise_label")
        question_number = correction.get("question_number")
        sub_part = correction.get("sub_part")
        instruction = correction.get("correction_instruction")
        reviewer = correction.get("reviewer", "batch_correction")
        
        # Build the PDF path
        # Need to determine class/chapter from context or assume from config
        import config
        class_name = config.CLASS
        chapter = config.CHAPTER
        
        pdf_path = os.path.join(
            output_dir,
            f"class_{class_name}",
            f"chapter_{chapter}",
            f"ex_{exercise_label.replace('.', '_')}.pdf"
        )
        
        if not os.path.exists(pdf_path):
            results["failed"] += 1
            results["errors"].append({
                "exercise": exercise_label,
                "question": question_number,
                "error": f"PDF not found at {pdf_path}"
            })
            logger.error(f"❌ PDF not found for exercise {exercise_label}: {pdf_path}")
            continue
        
        try:
            regenerate_question(
                pdf_output_path=pdf_path,
                question_number=question_number,
                correction_instruction=instruction,
                sub_part=sub_part,
                reviewer=reviewer
            )
            results["success"] += 1
            logger.info(f"✅ Applied correction to Exercise {exercise_label} Q{question_number}")
        except Exception as e:
            results["failed"] += 1
            results["errors"].append({
                "exercise": exercise_label,
                "question": question_number,
                "error": str(e)
            })
            logger.error(f"❌ Failed to correct Exercise {exercise_label} Q{question_number}: {e}")
    
    return results


def republish_corrected_chapter(class_name: int, chapter: int, 
                                output_dir: str = "output",
                                published_dir: str = "published") -> Dict:
    """
    Republishes a chapter after corrections have been applied.
    
    This function:
    1. Checks if all exercises are approved
    2. If yes, publishes to the published directory
    3. If no, returns status of which exercises still need approval
    
    Args:
        class_name: Class level (9, 10, etc.)
        chapter: Chapter number
        output_dir: Directory containing draft PDFs
        published_dir: Directory for published PDFs
    
    Returns:
        Dict with publish status
    """
    status = chapter_publish_status(class_name, chapter, output_dir)
    
    if status["ready_to_publish"]:
        try:
            publish_result = publish_chapter(class_name, chapter, output_dir, published_dir)
            return {
                "success": True,
                "message": f"Chapter {chapter} published successfully",
                "published_files": publish_result["published_files"],
                "dest_dir": publish_result["dest_dir"],
                "status": status
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to publish: {e}",
                "status": status
            }
    else:
        return {
            "success": False,
            "message": f"Chapter {chapter} not ready to publish",
            "pending_exercises": status["pending"],
            "status": status
        }


def export_review_state(class_name: int, chapter: int, 
                       output_dir: str = "output",
                       export_file: str = "review_state_export.json") -> str:
    """
    Exports the current review state for a chapter to a JSON file.
    
    This allows reviewing the state offline and creating correction files.
    
    Args:
        class_name: Class level
        chapter: Chapter number
        output_dir: Output directory
        export_file: Path to export file
    
    Returns:
        Path to the exported file
    """
    import review_state as rs
    from review_state import list_states
    
    all_states = list_states(output_dir)
    chapter_states = [
        s for s in all_states
        if str(s.get("class_name")) == str(class_name) 
        and str(s.get("chapter")) == str(chapter)
    ]
    
    export_data = {
        "class_name": class_name,
        "chapter": chapter,
        "exercises": chapter_states,
        "export_timestamp": rs._now() if hasattr(rs, '_now') else "unknown"
    }
    
    with open(export_file, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📤 Exported review state for Chapter {chapter} to {export_file}")
    return export_file


def import_and_apply_corrections(corrections_file: str, 
                                  class_name: int = None,
                                  chapter: int = None,
                                  output_dir: str = "output",
                                  auto_republish: bool = False) -> Dict:
    """
    Imports a corrections file and applies all corrections.
    
    This is a convenience function that combines:
    1. Loading corrections from file
    2. Applying each correction
    3. Optionally republishing if all exercises become approved
    
    Args:
        corrections_file: Path to corrections JSON file
        class_name: Class level (if None, uses config.CLASS)
        chapter: Chapter number (if None, uses config.CHAPTER)
        output_dir: Output directory
        auto_republish: If True, automatically republish after corrections
    
    Returns:
        Dict with results
    """
    import config
    
    if class_name is None:
        class_name = config.CLASS
    if chapter is None:
        chapter = config.CHAPTER
    
    logger.info(f"🔧 Starting batch correction for Class {class_name} Chapter {chapter}")
    
    # Apply corrections
    correction_results = apply_corrections_from_file(corrections_file, output_dir)
    
    result = {
        "corrections": correction_results,
        "republish": None
    }
    
    # Optionally republish
    if auto_republish and correction_results["success"] > 0:
        logger.info("🚀 Attempting to republish chapter after corrections...")
        republish_result = republish_corrected_chapter(class_name, chapter, output_dir)
        result["republish"] = republish_result
    
    return result


def create_correction_template(output_file: str = "corrections_template.json") -> str:
    """
    Creates a template corrections file for manual editing.
    
    Args:
        output_file: Path to write the template
    
    Returns:
        Path to the created template file
    """
    template = {
        "corrections": [
            {
                "exercise_label": "7.1",
                "question_number": 1,
                "sub_part": None,
                "correction_instruction": "Describe the correction needed here",
                "reviewer": "your_name"
            }
        ],
        "instructions": [
            "Edit this file to add corrections you want to apply",
            "exercise_label: should match the exercise (e.g., '7.1', '7.2')",
            "question_number: the question number to correct",
            "sub_part: null for main question, or 'a', 'b', etc. for sub-parts",
            "correction_instruction: what should be fixed",
            "reviewer: who is making this correction"
        ]
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(template, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📝 Created correction template at {output_file}")
    return output_file
