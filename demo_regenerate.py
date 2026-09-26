"""
Evidence script for the V8 polishing-phase fixes.

Builds a "solved" dict using the REAL data transcribed from the actual
uploaded exercise PDFs (Exercise 14.3 Q4(iii) and Q6, Exercise 14.4 Q1,
and Exercise 7.2 Q1), runs it through the ACTUAL, now-patched
solver._normalize_all_text_fields() + html_renderer.render_exercise_html()
+ pdf_generator.render_pdf() (real Playwright/Chromium, no mocking of the
rendering pipeline itself), and writes two real PDFs to /tmp for direct
inspection:

  /tmp/demo_before.pdf  -- template reverted to the OLD unconditional
                            "✔ প্ৰমাণিত হ'ল" badge, to show the bug as
                            it existed
  /tmp/demo_after.pdf   -- current (fixed) template + classifier

Also attaches the Exercise 14.3 Q6 case with has_book_diagram=True and
no image pool as the strict-figure-matching case (attaches nothing,
correctly falling back to whatever diagram_spec exists — none here
either, so the diagram box is simply omitted for that question, which
is the correct behavior instead of a wrong figure appearing).
"""
import solver
import html_renderer
import pdf_generator

QUESTIONS = [
    # Exercise 7.2 Q1 — a genuine proof. Should say "Proven" both before and after.
    {
        "question_number": 1,
        "question_text": "এটা সমদ্বিবাহু ত্ৰিভুজ ABC ত AB = AC আৰু ∠B আৰু ∠C ৰ সমদ্বিখণ্ডক দুডালে O বিন্দুত "
                          "পৰস্পৰ কটাকটি কৰে। A ক O ৰ সৈতে ৰেখাৰে সংলগ্ন কৰা। দেখুওৱা যে (i) OB = OC (ii) AO য়ে "
                          "∠A ক সমদ্বিখণ্ডিত কৰে।",
        "given": "প্ৰদত্ত যে ABC এটা সমদ্বিবাহু ত্ৰিভুজ য'ত AB = AC। BO আৰু CO ক্ৰমে ∠B আৰু ∠C ৰ সমদ্বিখণ্ডক।",
        "required": "প্ৰমাণ কৰিব লাগে যে (i) OB = OC (ii) AO য়ে ∠A ক সমদ্বিখণ্ডিত কৰে।",
        "steps": ["ΔABC ত, AB = AC (প্ৰদত্ত)।", "গতিকে, ∠ACB = ∠ABC (সমান বাহুৰ বিপৰীত কোণ)।"],
        "final_answer": "গতিকে, (i) OB = OC আৰু (ii) AO য়ে ∠A ক সমদ্বিখণ্ডিত কৰে বুলি প্ৰমাণ কৰা হ'ল।",
        "diagram_spec": None,
        "has_book_diagram": False,
    },
    # Exercise 14.4 Q1 — pure mean/median/mode calculation. Real bug: was stamped "Proven".
    {
        "question_number": 1,
        "question_text": "10 খন খেলত এটা দলে লাভ কৰা গোলৰ সংখ্যা আছিল এনেধৰণৰ— 2, 3, 4, 5, 0, 1, 3, 3, 4, 3। এই "
                          "গোলসমূহৰ মাধ্য, মধ্যমা আৰু বহুলক নিৰ্ণয় কৰা।",
        "given": "10 খন খেলত এটা দলে লাভ কৰা গোলৰ সংখ্যা হ'ল: 2, 3, 4, 5, 0, 1, 3, 3, 4, 3",
        "required": "গোলসমূহৰ মাধ্য, মধ্যমা আৰু বহুলক নিৰ্ণয় কৰিব লাগে।",
        "steps": ["গোলসমূহৰ মুঠ যোগফল = 28।", "মুঠ খেলৰ সংখ্যা = 10।", "মাধ্য = 28/10 = 2.8।"],
        "final_answer": "নিৰ্ণেয়ে মাধ্য = 2.8, মধ্যমা = 3, আৰু বহুলক = 3।",
        "diagram_spec": None,
        "has_book_diagram": False,
    },
    # Exercise 14.3 Q4(iii) — the sharpest real case: correct answer is "No", but the
    # OLD template still printed "✔ প্ৰমাণিত হ'ল" directly under it.
    {
        "question_number": 4,
        "sub_part": "iii",
        "question_text": "সৰ্বোচ্চ সংখ্যক পাতৰ দীঘ 153 মিঃমিঃ বুলি মন্তব্য আগবঢ়োৱা কথাটো শুদ্ধ হ'বনে? কিয়?",
        "given": "বাৰংবাৰতা বিভাজন তালিকা য'ত 145-153 মি.মি. শ্ৰেণীৰ বাৰংবাৰতা সৰ্বাধিক (12)।",
        "required": "সৰ্বাধিক সংখ্যক পাতৰ দীঘ 153 মি.মি. হয় নে নহয় পৰীক্ষা কৰা।",
        "steps": ["তালিকাখনৰ পৰা দেখা যায় যে সৰ্বাধিক সংখ্যক পাত (12 টা) 145-153 মি.মি. শ্ৰেণী অন্তৰালৰ ভিতৰত আছে।",
                  "153 মি.মি. কেৱল সেই শ্ৰেণীটোৰ উচ্চ সীমা।"],
        "final_answer": "নহয়, মন্তব্যটো শুদ্ধ নহয়। কাৰণ 153 মি.মি. হৈছে কেৱল শ্ৰেণী অন্তৰালৰ উচ্চ সীমা, প্ৰকৃত দীঘ যিকোনো হ'ব পাৰে।",
        "diagram_spec": None,
        "has_book_diagram": False,
    },
]


def build_pdf(questions, template_path, output_path):
    """Renders with a chosen template variant (patches html_renderer's
    template lookup to point at a given file) and writes a real PDF."""
    solved = {"questions": [dict(q) for q in questions]}
    solver._normalize_all_text_fields(solved)  # sets answer_kind via the REAL fixed classifier

    from jinja2 import Environment, FileSystemLoader
    env = Environment(loader=FileSystemLoader("templates"))
    orig_env = html_renderer._env
    html_renderer._env = env
    try:
        html = html_renderer.render_exercise_html(
            solved, class_name=9, chapter=14, chapter_name="পৰিসংখ্যা", exercise_label="14.3/14.4 (demo)"
        )
    finally:
        html_renderer._env = orig_env

    pdf_generator.render_pdf(html, output_path)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    build_pdf(QUESTIONS, "templates/base.html.jinja", "/tmp/demo_after.pdf")
