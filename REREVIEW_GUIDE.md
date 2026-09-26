# How to Re-Review an Exercise by Editing Review File

## Quick Method (What You Requested)

### Step 1: Find the Review File

For an exercise like `ex_7_1.pdf`, the review file is located at:
```
output/class_9/chapter_7/Review/ex_7_1.review.json
```

### Step 2: Edit the Review File

Open the JSON file and add/set this field:
```json
{
  "exercise_label": "7.1",
  "status": "approved",
  "force_rereview": true,
  ...
}
```

Just change `"force_rereview": false` to `"force_rereview": true` (or add it if it doesn't exist).

### Step 3: Run main.py

```bash
python main.py
```

The system will:
1. Detect `force_rereview: true`
2. Reset the exercise to `pending_review` status
3. Open the human review interface (web or CLI)
4. You can now correct specific questions using AI regeneration

### Step 4: Correct Questions

In the review interface:
- Select the question you want to fix
- Choose "Regenerate with correction"
- Type your correction instruction
- AI will regenerate just that question's solution
- Accept or reject the corrected version

## Configuration for Construction Chapters

### Set in config.py

```python
# For construction chapters (অংকন)
IS_CONSTRUCTION_CHAPTER = True

# For regular geometry chapters (ত্রিভুজ, বৃত্ত, etc.)
IS_CONSTRUCTION_CHAPTER = False  # Default
```

### What This Does

**When IS_CONSTRUCTION_CHAPTER = True:**
- AI is instructed that EVERY question is a construction question
- Forces `construction_instruments` for all questions
- Uses construction-style steps (ruler/compass actions)
- Uses `diagram_type: "construction"` for diagrams

**When IS_CONSTRUCTION_CHAPTER = False:**
- AI is instructed this is NOT a construction chapter
- Uses proof-style reasoning for geometry questions
- Only adds construction instructions if explicitly asked
- Uses appropriate diagram types (triangle, circle, etc.)

## Example Workflow

### Scenario: You want to fix Question 3 in Exercise 7.1

```bash
# 1. Edit the review file
# output/class_9/chapter_7/Review/ex_7_1.review.json
# Set: "force_rereview": true

# 2. Run main.py
python main.py

# 3. The review interface opens
# Navigate to Question 3
# Click "Regenerate with correction"
# Type: "Remove construction steps - this is a proof question, not construction"
# AI regenerates the question
# Review and accept the corrected version
```

### Scenario: Processing a Construction Chapter

```python
# In config.py
IS_CONSTRUCTION_CHAPTER = True
CHAPTER = 11  # Construction chapter
SOLVE_SINGLE_EXERCISE = True
EXERCISE_FILTER = ["11.1"]

# Run
python main.py

# All questions will be treated as construction questions
# No risk of AI mistakenly adding construction to geometry
```

## Important Notes

1. **force_rereview is one-time**: After you run main.py, the flag is automatically reset to false
2. **No full regeneration**: Only the specific questions you correct are regenerated
3. **Preserves history**: All previous corrections and acceptances are kept
4. **Safe editing**: The system uses file locks to prevent corruption

## Review File Structure

```json
{
  "exercise_label": "7.1",
  "class_name": 9,
  "chapter": 7,
  "chapter_name": "ত্রিভুজ",
  "status": "approved",
  "version": 2,
  "force_rereview": false,  // <-- Set this to true to trigger re-review
  "solved": { ... },
  "history": [ ... ],
  "accepted_questions": ["1", "2", "3"],
  "reviewer_notes": []
}
```

## Troubleshooting

### Issue: force_rereview not working

**Check:**
1. The JSON syntax is valid (no trailing commas)
2. The file path is correct
3. You're running main.py from the correct directory

### Issue: Exercise still skipped

**Check:**
1. The status was "approved" before setting force_rereview
2. The JSON file was saved correctly
3. Check the logs for "force_rereview=True" message

### Issue: Construction detection not working

**Check:**
1. IS_CONSTRUCTION_CHAPTER is set correctly in config.py
2. You're not relying on auto-detection (which is now disabled by default)
3. Check the logs for "construction_chapter=True/False" message
