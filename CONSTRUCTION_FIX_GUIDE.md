# Construction Chapter Detection & Batch Correction - Implementation Guide

## Overview

This implementation fixes two critical issues in the math solution factory:

1. **Smart Construction Chapter Detection**: Prevents the AI from mistakenly adding construction instructions to regular geometry chapters
2. **Batch Correction Workflow**: Allows correcting errors without regenerating entire chapters

## New Files Added

### 1. `construction_detector.py`
Smart detection of construction chapters based on chapter titles.

**Key Functions:**
- `is_construction_chapter(title, chapter_number)` - Detects if a chapter is a construction chapter
- `should_use_construction_prompt(title, chapter_number, class_level)` - Returns configuration for prompting
- `validate_question_construction_type(question, chapter_is_construction)` - Validates question-level construction type

**Detection Logic:**
- Detects construction keywords in Assamese (অংকন, অঙ্কন, etc.) and English (construction, geometric construction)
- Handles mixed titles (e.g., "ত্রিভুজ অংকন" = Triangle Construction)
- Prevents false positives for non-construction chapters (ত্রিভুজ, বৃত্ত, etc.)

### 2. `batch_correction.py`
Batch correction and republishing workflow.

**Key Functions:**
- `apply_corrections_from_file(corrections_file, output_dir)` - Applies corrections from JSON file
- `republish_corrected_chapter(class_name, chapter, output_dir, published_dir)` - Republishes after corrections
- `export_review_state(class_name, chapter, output_dir, export_file)` - Exports current review state
- `import_and_apply_corrections(corrections_file, ...)` - Convenience function for full workflow
- `create_correction_template(output_file)` - Creates a template for manual editing

### 3. Test Files
- `test_construction_detector.py` - Unit tests for construction detection
- `test_batch_correction.py` - Unit tests for batch correction workflow

## Modified Files

### 1. `prompts.py`
**Changes:**
- Updated `build_solve_prompt()` to accept `chapter_title` and `is_construction_chapter` parameters
- Adds chapter-specific guidance to the AI prompt:
  - For construction chapters: Forces construction instruments and construction-style steps
  - For non-construction chapters: Prevents construction instructions unless explicitly asked

### 2. `solver.py`
**Changes:**
- Added import for `construction_detector`
- Updated `solve_exercise()` to accept `chapter_title` parameter
- Added construction detection and validation
- Auto-corrects construction type mismatches between questions and chapter type

### 3. `main.py`
**Changes:**
- Updated `process_exercise()` to pass `chapter_name` to `solve_exercise()`

## Usage

### Running with Construction Detection

The system now automatically detects construction chapters and adjusts the AI prompt accordingly. No configuration changes needed - it works automatically based on chapter titles.

**Example:**
```python
# If chapter title is "অংকন" (Constructions)
# The AI will automatically:
# - Force construction_instruments for all questions
# - Use construction-style steps (ruler/compass actions)
# - Use diagram_type "construction" for diagrams

# If chapter title is "ত্রিভুজ" (Triangles)
# The AI will automatically:
# - Use proof-style reasoning
# - Use appropriate diagram_type (triangle, circle, etc.)
# - Only add construction instructions if explicitly asked
```

### Batch Correction Workflow

#### Step 1: Create a Corrections File

```python
from batch_correction import create_correction_template

# Create a template file
create_correction_template("my_corrections.json")
```

Edit the generated JSON file:

```json
{
  "corrections": [
    {
      "exercise_label": "7.1",
      "question_number": 1,
      "sub_part": null,
      "correction_instruction": "Fix the calculation in step 3 - the answer should be 42, not 24",
      "reviewer": "your_name"
    },
    {
      "exercise_label": "7.2",
      "question_number": 3,
      "sub_part": "a",
      "correction_instruction": "The diagram should show angle A as 60°, not 45°",
      "reviewer": "your_name"
    }
  ]
}
```

#### Step 2: Apply Corrections

```python
from batch_correction import import_and_apply_corrections

# Apply corrections without republishing
result = import_and_apply_corrections(
    "my_corrections.json",
    class_name=9,
    chapter=7,
    auto_republish=False
)

print(f"Applied {result['corrections']['success']} corrections successfully")
```

#### Step 3: Republish (Optional)

```python
# Apply corrections and auto-republish if all exercises are approved
result = import_and_apply_corrections(
    "my_corrections.json",
    class_name=9,
    chapter=7,
    auto_republish=True
)

if result['republish']['success']:
    print(f"Chapter published to: {result['republish']['dest_dir']}")
```

### Export Review State

```python
from batch_correction import export_review_state

# Export current review state for offline review
export_file = export_review_state(
    class_name=9,
    chapter=7,
    output_dir="output",
    export_file="chapter_7_review_state.json"
)
```

### Manual Correction via CLI

You can also use the existing correction engine directly:

```python
from correction_engine import regenerate_question

# Correct a single question
regenerate_question(
    pdf_output_path="output/class_9/chapter_7/ex_7_1.pdf",
    question_number=1,
    correction_instruction="Fix the calculation error in step 3",
    reviewer="admin"
)
```

## Testing

### Run Unit Tests

```bash
# Test construction detection
python -m pytest test_construction_detector.py -v

# Test batch correction
python -m pytest test_batch_correction.py -v
```

### Test with Real Vertex AI

To test with real Vertex AI:

1. Ensure your GCP credentials are set up:
```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

2. Configure `config.py` with your test chapter:
```python
CLASS = 9
CHAPTER = 11  # Construction chapter
# or
CHAPTER = 7   # Non-construction chapter
```

3. Run the pipeline:
```bash
python main.py
```

4. Check the logs for construction detection:
```
🔍 Chapter 'অংকন' identified as a CONSTRUCTION chapter
🤖 Solving Exercise 11.1 (single Gemini call, construction_chapter=True)
```

## Validation Checklist

- [ ] Construction chapters (অংকন) are detected correctly
- [ ] Non-construction chapters (ত্রিভুজ, বৃত্ত) are not treated as construction
- [ ] Mixed titles (ত্রিভুজ অংকন) are handled correctly
- [ ] AI prompts include chapter-specific guidance
- [ ] Construction type mismatches are auto-corrected
- [ ] Batch correction applies changes without full regeneration
- [ ] Republishing works after corrections
- [ ] Review state export/import works correctly

## Troubleshooting

### Issue: Chapter incorrectly detected as construction

**Solution:** Check the chapter title in the PDF. The detection is based on the title returned by `chapter_detector.get_chapter_title()`. If the title is incorrect, you may need to improve the title detection or manually override.

### Issue: Corrections not applying

**Solution:** Ensure:
1. The PDF path in the corrections file matches the actual file location
2. The exercise label matches exactly (e.g., "7.1" not "7.1 ")
3. The question number exists in the exercise
4. The review state file exists alongside the PDF

### Issue: Auto-republish fails

**Solution:** Check that:
1. All exercises in the chapter are approved
2. The review state files are not corrupted
3. You have write permissions to the published directory

## Integration with Existing Workflow

The changes are fully backward compatible:

- Existing `main.py` workflow continues to work unchanged
- Construction detection is automatic based on chapter titles
- The human review workflow (review_webapp.py) continues to work as before
- Existing correction_engine.py functions are unchanged, just wrapped by batch_correction.py

## Performance Impact

- **Construction detection**: Negligible (simple string matching)
- **Validation per question**: Minimal (runs after AI response, before rendering)
- **Batch correction**: Only runs when explicitly invoked
- **No impact** on normal solve pipeline performance
