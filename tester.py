"""
Stage 4 — Automated Testing
For each question, generates sample answers for all 10 evaluation categories,
runs them through the evaluator (GPT-4o mini), and records results in the sheet.
"""

import os
import re
import json
import time
import gspread
from openai import OpenAI
from google.oauth2 import service_account
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GSHEET_ID      = os.getenv("GSHEET_ID")
GOOGLE_SA_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")

COL_QUESTION  = 3
COL_MARKS     = 4
COL_ANSWER    = 5
COL_REWRITING = 8
COL_PROMPTS   = 9
COL_IDENTITY  = 10
COL_SCORING   = 11
COL_PAPER     = 13
COL_QNUM      = 14

client = OpenAI(api_key=OPENAI_API_KEY)

CATEGORIES = [
    "Correct Answer",
    "Incorrect Answer",
    "Incomplete Answer",
    "Hallucinations",
    "Correct Answer but outside of Points to Discuss",
    "Partially Correct with Incorrect Information",
    "Invalid Answer",
    "Correct Answer with New Line",
    "Incorrect Answer with Formatting/Grammar Issue",
    "Correct Answer with Formatting/Grammar Issue",
]


def get_sheet():
    creds = service_account.Credentials.from_service_account_info(
        json.loads(GOOGLE_SA_JSON),
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
    )
    gc = gspread.authorize(creds)
    return gc.open_by_key(GSHEET_ID).sheet1


def generate_sample_answers(question: str, answer: str, marks: str, rewriting: str) -> dict:
    """Generate one sample answer per category using GPT-4o mini."""

    system_prompt = """You are an expert GCSE exam answer generator.
Given a GCSE question and its correct answer, generate realistic sample student WRITTEN answers for each of the 10 evaluation categories.

IMPORTANT: Students always TYPE their answers as written text. Even for questions that list options, students write their chosen answer as text (e.g. they write "Organ" not click a box).

Return ONLY a valid JSON object with exactly these 10 keys:
{
  "Correct Answer": "...",
  "Incorrect Answer": "...",
  "Incomplete Answer": "...",
  "Hallucinations": "...",
  "Correct Answer but outside of Points to Discuss": "...",
  "Partially Correct with Incorrect Information": "...",
  "Invalid Answer": "...",
  "Correct Answer with New Line": "...",
  "Incorrect Answer with Formatting/Grammar Issue": "...",
  "Correct Answer with Formatting/Grammar Issue": "..."
}

Guidelines for each category:
- Correct Answer: A perfectly correct written answer e.g. "Organ" or "Light can reach the palisade mesophyll"
- Incorrect Answer: A plausible but completely wrong written answer
- Incomplete Answer: For 1-mark questions write something TOO VAGUE to earn the mark e.g. "It helps the plant" or "So it can work properly" — deliberately insufficient, not the actual answer
- Hallucinations: Confidently states made-up false scientific facts
- Correct Answer but outside of Points to Discuss: Correct answer PLUS extra irrelevant information e.g. "Organ. This is because organs contain cells and DNA."
- Partially Correct with Incorrect Information: Write an answer that mixes one correct element with a clearly wrong statement. The wrong statement must contradict or undermine the correct part so the overall answer does NOT earn the mark on a 1-mark question. NEVER write a fully correct answer for this category. For multi-mark questions, write something that earns some but not all marks.
- Invalid Answer: Random gibberish like "asdf hjkl" or "123 abc"
- Correct Answer with New Line: Correct answer but with extra blank lines e.g. "Organ\n\n\n"
- Incorrect Answer with Formatting/Grammar Issue: Wrong answer with spelling/grammar errors
- Correct Answer with Formatting/Grammar Issue: Correct answer with a minor typo that still makes the answer clearly recognisable e.g. "Organn" or "lite can reech the mesophyll". The core answer must be identifiable as correct despite the error. Do NOT change the meaning — only add superficial typos or spelling mistakes.

Return ONLY the JSON object, no markdown, no explanation."""

    user_msg = f"""Question: {rewriting if rewriting else question}
Correct Answer / Mark Scheme: {answer if answer else 'Not specified'}
Total Marks: {marks}

Generate sample answers for all 10 categories."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.8,
        max_tokens=4000
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r'("(?:[^"\\]|\\.)*")', lambda m: m.group(0).replace('\n', '\\n'), raw)
    raw = re.sub(r"^```json|^```|```$", "", raw, flags=re.MULTILINE).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\{[\s\S]*\}', raw)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
        print(f"      ⚠️  Could not parse sample answers JSON, retrying...")
        response2 = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Return ONLY a valid JSON object with no markdown or explanation."},
                {"role": "user", "content": user_msg}
            ],
            temperature=0,
            max_tokens=4000
        )
        raw2 = response2.choices[0].message.content.strip()
        raw2 = re.sub(r'("(?:[^"\\]|\\.)*")', lambda m: m.group(0).replace('\n', '\\n'), raw2)
        raw2 = re.sub(r"^```json|^```|```$", "", raw2, flags=re.MULTILINE).strip()
        try:
            return json.loads(raw2)
        except:
            print(f"      ⚠️  Retry failed, skipping")
            return {cat: "" for cat in CATEGORIES}


def run_evaluator(full_prompt: str, student_answer: str, max_marks: int, category: str = "") -> dict:
    """Send a student answer to the evaluator and parse the result."""

    if category == "Correct Answer with Formatting/Grammar Issue":
        extra_instruction = (
            "\n\nIMPORTANT: The student's answer may contain minor typos, spelling mistakes, "
            "or grammar errors. If the core answer is clearly identifiable as correct despite "
            "these errors, award FULL marks. Do NOT penalise for formatting or spelling alone. "
            "Only withhold marks if the meaning of the answer is changed or unclear."
        )
    elif category == "Correct Answer but outside of Points to Discuss":
        extra_instruction = (
            "\n\nIMPORTANT: The student's answer contains the correct answer plus some "
            "additional irrelevant information. Award FULL marks for the correct part. "
            "Do NOT deduct marks because extra information was included — only the "
            "correct answer portion needs to match the mark scheme."
        )
    elif category == "Correct Answer with New Line":
        extra_instruction = (
            "\n\nIMPORTANT: The student's answer may contain extra blank lines or "
            "whitespace before or after the actual answer. Ignore all leading and "
            "trailing whitespace and newlines — evaluate only the text content itself. "
            "If the core answer is correct, award FULL marks."
        )
    else:
        extra_instruction = ""

    max_marks_instruction = (
        f"\n\nIMPORTANT: The maximum mark for this question is {max_marks}. "
        f"Do NOT award more than {max_marks} mark(s) under any circumstances."
    )

    eval_prompt = full_prompt.replace(
        "**Student Response:**",
        f"**Student Response:**\n{student_answer}\n**"
    ) + extra_instruction + max_marks_instruction

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are an exam marking officer. You must NEVER award more than {max_marks} mark(s) "
                    f"total for any response. The maximum possible mark is {max_marks}. "
                    f"Any score above {max_marks} is strictly forbidden."
                )
            },
            {"role": "user", "content": eval_prompt}
        ],
        temperature=0,
        max_tokens=500
    )

    raw = response.choices[0].message.content.strip()
    score = 0
    match = re.search(r"Final Mark[:\s]+(\d+)\s*(?:/\s*\d+)?", raw, re.IGNORECASE)
    if match:
        score = int(match.group(1))

    return {"score": score, "max": max_marks, "feedback": raw}


def evaluate_category_result(category: str, score: int, max_marks: int) -> bool:
    correct_categories = {
        "Correct Answer",
        "Correct Answer but outside of Points to Discuss",
        "Correct Answer with New Line",
        "Correct Answer with Formatting/Grammar Issue",
    }
    partial_categories = {
        "Incomplete Answer",
        "Partially Correct with Incorrect Information",
    }
    zero_categories = {
        "Incorrect Answer",
        "Hallucinations",
        "Invalid Answer",
        "Incorrect Answer with Formatting/Grammar Issue",
    }

    if category in correct_categories:
        return score == max_marks
    elif category in partial_categories:
        return 0 < score < max_marks or (max_marks == 1 and score == 0)
    elif category in zero_categories:
        return score == 0
    return True


def run_tester(overwrite: bool = False, single_row: int = None):
    """Run the automated testing pipeline."""
    print("\n🧪 Stage 4 — Automated testing...")
    sheet = get_sheet()
    all_rows = sheet.get_all_values()
    data_rows = all_rows[1:]

    total = tested = skipped = 0

    for i, row in enumerate(data_rows):
        row_num = i + 2

        if single_row and row_num != single_row:
            continue

        while len(row) < 14:
            row.append("")

        question  = row[COL_QUESTION - 1].strip()
        marks_str = row[COL_MARKS - 1].strip()
        answer    = row[COL_ANSWER - 1].strip()
        rewriting = row[COL_REWRITING - 1].strip()
        identity  = row[COL_IDENTITY - 1].strip()
        scoring   = row[COL_SCORING - 1].strip()
        qnum      = row[COL_QNUM - 1].strip()

        if not question or not identity:
            skipped += 1
            continue

        if scoring and not overwrite:
            print(f"   Row {row_num} ({qnum}): ⏭️  Already tested, skipping")
            skipped += 1
            continue

        total += 1
        max_marks = int(marks_str) if marks_str.isdigit() else 1
        print(f"\n   Row {row_num} ({qnum}): 🧪 Testing ({max_marks} marks)...")

        print(f"      Generating sample answers...")
        samples = generate_sample_answers(question, answer, marks_str, rewriting)
        time.sleep(0.5)

        results = {}
        all_passed = True

        for category in CATEGORIES:

            # Skip 'Partially Correct' for 1-mark questions — no partial credit exists
            if category == "Partially Correct with Incorrect Information" and max_marks == 1:
                results[category] = {
                    "score": 0, "max": max_marks, "pass": True,
                    "answer": "N/A — skipped for 1-mark questions",
                    "feedback": "Category not applicable for 1-mark questions.",
                }
                print(f"      ⏭️  {category}: skipped (1-mark question)")
                continue

            student_answer = samples.get(category, "")
            if not student_answer:
                results[category] = {"score": 0, "max": max_marks, "pass": False, "answer": "", "feedback": ""}
                all_passed = False
                continue

            eval_result = run_evaluator(identity, student_answer, max_marks, category)
            passed = evaluate_category_result(category, eval_result["score"], max_marks)
            if not passed:
                all_passed = False

            results[category] = {
                "score":    eval_result["score"],
                "max":      max_marks,
                "pass":     passed,
                "answer":   student_answer,
                "feedback": eval_result["feedback"],
            }

            status = "✅" if passed else "❌"
            print(f"      {status} {category}: {eval_result['score']}/{max_marks}")
            time.sleep(0.3)

        scoring_data = {
            "overall_pass": all_passed,
            "status": "approved" if all_passed else "needs_review",
            "results": results,
            "samples": samples,
        }

        sheet.update_cell(row_num, COL_SCORING, json.dumps(scoring_data))
        print(f"      {'✅ ALL PASSED' if all_passed else '❌ NEEDS REVIEW'}")
        tested += 1
        time.sleep(0.5)

    print(f"\n── Testing complete ──")
    print(f"   ✅ Tested: {tested} | ⏭️  Skipped: {skipped} | Total: {total + skipped}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run automated evaluation testing")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--row", type=int, default=None)
    args = parser.parse_args()
    run_tester(overwrite=args.overwrite, single_row=args.row)