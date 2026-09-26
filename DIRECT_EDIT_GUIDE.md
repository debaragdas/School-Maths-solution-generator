# Direct Edit Feature - User Guide

## Overview

The Direct Edit feature allows you to manually edit solution text during human review without using AI regeneration. This is perfect for:
- Fixing typos
- Correcting formatting issues
- Replacing entire solution text
- Adding custom explanations
- Adjusting mathematical notation

## How to Use

### Method 1: Web Interface (Recommended)

1. Open the human review web interface (runs automatically when you run `python main.py`)
2. Navigate to the question you want to edit
3. Click the **✏️ Direct Edit** button
4. An edit form will appear with four fields:
   - **Given (প্ৰদত্ত)**: The given information
   - **Required (প্ৰয়োজনীয়)**: What to prove/find
   - **Steps (সমাধান)**: The solution steps
   - **Final Answer (চূড়ান্ত উত্তৰ)**: The final answer

5. Edit the fields as needed
6. Click **💾 Save Edits** to save your changes
7. The PDF will be automatically re-rendered with your edits

### Method 2: Keyboard Shortcut

- Press **E** on the current question to open the edit form
- Press **Enter** to save edits when the form is open

## MathJax Support

The edit form supports MathJax mathematical notation:

- **Inline math**: Use `\(x^2\)` for inline equations
- **Display math**: Use `$$x^2$$` for centered equations
- **Fractions**: `$$\frac{a}{b}$$`
- **Square roots**: `$$\sqrt{x}$$`
- **Greek letters**: `$$\alpha, \beta, \theta$$`

## Assamese Font

All edited text will be rendered with the correct Assamese font (Tiro Bangla) in the output PDF, matching the AI-generated content.

## Important Notes

- **No AI regeneration**: Direct edits are saved without calling the AI
- **PDF auto-renders**: The PDF is automatically regenerated after saving edits
- **Version tracking**: Each edit increments the version number and adds a history entry
- **Approval reset**: Editing a question resets its approval status (you must approve it again)
- **Formatting preserved**: MathJax delimiters and Assamese text render correctly in the PDF

## Example Usage

### Fixing a Typo

**Original:**
```
Given: প্ৰদত্ত: AB = 5 cm, BC = 3 cm
```

**Edit:**
```
Given: প্ৰদত্ত: AB = 5 cm, BC = 4 cm
```

### Adding Mathematical Notation

**Original:**
```
Steps: Calculate the area using formula
```

**Edit:**
```
Steps: Calculate the area using formula $$A = \frac{1}{2} \times b \times h$$
```

### Replacing Entire Solution

You can completely replace the solution text by editing all fields:

```
Given: প্ৰদত্ত: Triangle ABC with AB = 6 cm, BC = 8 cm, ∠B = 90°
Required: প্ৰয়োজনীয়: Find AC
Steps: 
1. Using Pythagoras theorem: $$AC^2 = AB^2 + BC^2$$
2. $$AC^2 = 6^2 + 8^2 = 36 + 64 = 100$$
3. $$AC = \sqrt{100} = 10$$ cm
Final Answer: চূড়ান্ত উত্তৰ: AC = 10 cm
```

## Comparison with AI Regeneration

| Feature | Direct Edit | AI Regeneration |
|---------|-------------|----------------|
| Speed | Instant | Slower (AI call) |
| Cost | Free | Uses AI quota |
| Control | Full manual control | AI interprets instruction |
| Use case | Typos, formatting, known fixes | Complex corrections, new solutions |

## Troubleshooting

### Issue: Edits not appearing in PDF

**Solution:** 
- Make sure you clicked "Save Edits"
- Check that the field wasn't empty (empty fields are not saved)
- Refresh the review page to see the updated content

### Issue: MathJax not rendering

**Solution:**
- Ensure you use the correct delimiters: `\( \)` for inline, `$$ $$` for display
- Check for escaped backslashes: use `\\(` not `\(` in the JSON

### Issue: Assamese text not displaying correctly

**Solution:**
- The Assamese font is automatically applied during PDF rendering
- If you see incorrect characters, ensure your text encoding is UTF-8
- The web interface uses the same font as the PDF, so it should match

## API Usage (Advanced)

If you need to use the direct edit feature programmatically:

```python
from correction_engine import edit_question_directly

# Edit a question directly
edit_question_directly(
    pdf_output_path="output/class_9/chapter_7/ex_7_1.pdf",
    question_number=1,
    field_updates={
        "given": "Updated given text",
        "steps": ["Step 1", "Step 2"],
        "final_answer": "Updated answer"
    },
    reviewer="your_name"
)
```
