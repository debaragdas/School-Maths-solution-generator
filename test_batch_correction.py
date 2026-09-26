"""
test_batch_correction.py — Unit tests for batch correction workflow.
"""
import pytest
import json
import os
import tempfile
from unittest.mock import patch, MagicMock
from batch_correction import (
    apply_corrections_from_file,
    republish_corrected_chapter,
    export_review_state,
    import_and_apply_corrections,
    create_correction_template
)


@pytest.fixture
def temp_corrections_file():
    """Create a temporary corrections file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        corrections = {
            "corrections": [
                {
                    "exercise_label": "7.1",
                    "question_number": 1,
                    "sub_part": None,
                    "correction_instruction": "Fix the calculation",
                    "reviewer": "test_user"
                }
            ]
        }
        json.dump(corrections, f)
        return f.name


@pytest.fixture
def mock_pdf_path():
    """Mock PDF path for testing."""
    return "output/class_9/chapter_7/ex_7_1.pdf"


def test_create_correction_template():
    """Test creation of correction template."""
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        template_path = f.name
    
    result_path = create_correction_template(template_path)
    
    assert os.path.exists(result_path)
    
    with open(result_path, 'r') as f:
        template = json.load(f)
    
    assert "corrections" in template
    assert "instructions" in template
    assert len(template["corrections"]) == 1
    assert template["corrections"][0]["exercise_label"] == "7.1"
    
    os.unlink(result_path)


def test_apply_corrections_from_file_missing_pdf(temp_corrections_file):
    """Test applying corrections when PDF doesn't exist."""
    # This should fail gracefully
    with patch('os.path.exists', return_value=False):
        result = apply_corrections_from_file(temp_corrections_file)
        
        assert result["total"] == 1
        assert result["failed"] == 1
        assert result["success"] == 0
        assert len(result["errors"]) == 1


@patch('batch_correction.regenerate_question')
def test_apply_corrections_from_file_success(mock_regenerate, temp_corrections_file):
    """Test successful application of corrections."""
    mock_regenerate.return_value = {"version": 2}
    
    with patch('os.path.exists', return_value=True):
        result = apply_corrections_from_file(temp_corrections_file)
        
        assert result["total"] == 1
        assert result["success"] == 1
        assert result["failed"] == 0
        assert len(result["errors"]) == 0
        mock_regenerate.assert_called_once()


@patch('batch_correction.chapter_publish_status')
@patch('batch_correction.publish_chapter')
def test_republish_corrected_chapter_ready(mock_publish, mock_status):
    """Test republishing when chapter is ready."""
    mock_status.return_value = {
        "ready_to_publish": True,
        "total_exercises": 5,
        "approved": 5,
        "pending": []
    }
    mock_publish.return_value = {
        "published_files": ["ex_7_1.pdf", "ex_7_2.pdf"],
        "dest_dir": "published/class_9/chapter_7"
    }
    
    result = republish_corrected_chapter(9, 7)
    
    assert result["success"] == True
    assert "published successfully" in result["message"]
    assert result["published_files"] == ["ex_7_1.pdf", "ex_7_2.pdf"]


@patch('batch_correction.chapter_publish_status')
def test_republish_corrected_chapter_not_ready(mock_status):
    """Test republishing when chapter is not ready."""
    mock_status.return_value = {
        "ready_to_publish": False,
        "total_exercises": 5,
        "approved": 3,
        "pending": ["7.4", "7.5"]
    }
    
    result = republish_corrected_chapter(9, 7)
    
    assert result["success"] == False
    assert "not ready to publish" in result["message"]
    assert result["pending_exercises"] == ["7.4", "7.5"]


@patch('review_state.list_states')
def test_export_review_state(mock_list_states):
    """Test exporting review state."""
    mock_list_states.return_value = [
        {
            "pdf_path": "output/class_9/chapter_7/ex_7_1.pdf",
            "exercise_label": "7.1",
            "class_name": 9,
            "chapter": 7,
            "status": "approved",
            "version": 1
        }
    ]
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        export_path = f.name
    
    result_path = export_review_state(9, 7, export_file=export_path)
    
    assert os.path.exists(result_path)
    
    with open(result_path, 'r') as f:
        export_data = json.load(f)
    
    assert export_data["class_name"] == 9
    assert export_data["chapter"] == 7
    assert len(export_data["exercises"]) == 1
    
    os.unlink(export_path)


@patch('batch_correction.apply_corrections_from_file')
@patch('batch_correction.republish_corrected_chapter')
def test_import_and_apply_corrections_no_republish(mock_republish, mock_apply):
    """Test importing corrections without auto-republish."""
    mock_apply.return_value = {
        "total": 2,
        "success": 2,
        "failed": 0,
        "errors": []
    }
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        corrections_file = f.name
    
    result = import_and_apply_corrections(corrections_file, auto_republish=False)
    
    assert result["corrections"]["success"] == 2
    assert result["republish"] is None
    mock_republish.assert_not_called()
    
    os.unlink(corrections_file)


@patch('batch_correction.apply_corrections_from_file')
@patch('batch_correction.republish_corrected_chapter')
def test_import_and_apply_corrections_with_republish(mock_republish, mock_apply):
    """Test importing corrections with auto-republish."""
    mock_apply.return_value = {
        "total": 2,
        "success": 2,
        "failed": 0,
        "errors": []
    }
    mock_republish.return_value = {
        "success": True,
        "message": "Published successfully"
    }
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        corrections_file = f.name
    
    result = import_and_apply_corrections(corrections_file, auto_republish=True)
    
    assert result["corrections"]["success"] == 2
    assert result["republish"] is not None
    mock_republish.assert_called_once()
    
    os.unlink(corrections_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
