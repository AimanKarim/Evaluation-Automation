"""
Stage 4 — Automated Testing
For each question in a paper-specific sheet tab, generates sample answers
for all 10 evaluation categories, runs them through the evaluator in parallel,
and records results back to the sheet.
"""

import os
import re
import json
import gspread
from openai import OpenAI
from google.oauth2 import service_account
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GSHEET_ID      = os.getenv("GSHEET_ID")
GOOGLE_SA_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")

COL_QUESTION  = 3
COL_MARKS     = 4
COL_ANSWER    = 5
COL_REWRITING = 8
COL_IDENTITY  = 10
COL_SCORING   = 11
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

ZERO_CATEGORIES = {
    "Incorrect Answer",
    "Hallucinations",
    "Invalid Answer",
    "Incorrect Answer with Formatting/Grammar Issue",
}


def get_sa_info():
    sa = GOOGLE_SA_JSON.strip()
    if sa.endswith(".json"):
        with open(sa) as f:
            return json.load(f)
    return json.loads(sa)


def get_sheet(paper_label: str = ""):
    creds = service_account.Credentials.from_service_account_info(
        get_sa_info(),
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
    )
    gc = gspread.authorize(creds)
    spreadsheet = gc.open_by_key(GSHEET_ID)

    if paper_label:
        try:
            return spreadsheet.worksheet(paper_label)
        except gspread.exceptions.WorksheetNotFound:
            raise ValueError(f"No tab found for paper '{paper_label}'. Run extractor first.")
    return spreadsheet.sheet1


# ─────────────────────────────────────────────
# STEP 1: GENERATE SAMPLE ANSWERS
# ─────────────────────────────────────────────

def generate_sample_answers(question: str, answer: str, marks: str, rewriting: str) -> dict:
    system_prompt = """You are an expert GCSE exam answer generator.
Given a GCSE question and its correct answer, generate realistic sample student WRITTEN answers for each of the 10 evaluation categories.

IMPORTANT: Students always TYPE their answers as written text. Even for questions that list options, students write their chosen answer as text.

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
- Correct Answer: A perfectly correct written answer
- Incorrect Answer: A plausible but completely wrong written answer. Must NOT contain any correct values or facts from the mark scheme.
- Incomplete Answer: For 1-mark questions write something TOO VAGUE to earn the mark — deliberately insufficient
- Hallucinations: Confidently states made-up false scientific facts. Must NOT contain any correct values or facts from the mark scheme.
- Correct Answer but outside of Points to Discuss: Correct answer PLUS extra irrelevant information
- Partially Correct with Incorrect Information: Mix one correct element with a clearly wrong statement that undermines it. NEVER write a fully correct answer for this category.
- Invalid Answer: Random gibberish like "asdf hjkl" or "123 abc"
- Correct Answer with New Line: Correct answer with extra blank lines
- Incorrect Answer with Formatting/Grammar Issue: Wrong answer with spelling/grammar errors. Must NOT contain any correct values or facts from the mark scheme.
- Correct Answer with Formatting/Grammar Issue: Correct answer with a minor typo still clearly recognisable.

Return ONLY the JSON object, no markdown, no explanation."""

    user_msg = f"""Question: {rewriting if rewriting else question}
Correct Answer / Mark Scheme: {answer if answer else 'Not specified'}
Total Marks: {marks}

CRITICAL: For Incorrect Answer, Incorrect Answer with Formatting/Grammar Issue, and Hallucinations — do NOT use any of the correct values or facts from the mark scheme above. Use completely different numbers and facts.

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


# ─────────────────────────────────────────────
# STEP 2: RUN EVALUATOR
# ─────────────────────────────────────────────

def run_evaluator(full_prompt: str, student_answer: str, max_marks: int, category: str = "") -> dict:
    if category == "Correct Answer with Formatting/Grammar Issue":
        extra_instruction = (
            "\n\nIMPORTANT: The student's answer may contain minor typos, spelling mistakes, "
            "or grammar errors. If the core answer is clearly identifiable as correct despite "
            "these errors, award FULL marks. Do NOT penalise for formatting or spelling alone."
        )
    elif category == "Correct Answer but outside of Points to Discuss":
        extra_instruction = (
            "\n\nIMPORTANT: The student's answer contains the correct answer plus some "
            "additional irrelevant information. Award FULL marks for the correct part. "
            "Do NOT deduct marks because extra information was included."
        )
    elif category == "Correct Answer with New Line":
        extra_instruction = (
            "\n\nIMPORTANT: The student's answer may contain extra blank lines or whitespace. "
            "Ignore all leading and trailing whitespace and newlines — evaluate only the text content. "
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
                    f"You are an exam marking officer. You must NEVER award more than {max_marks} mark(s). "
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


# ─────────────────────────────────────────────
# STEP 3: TWO-PASS VERIFICATION
# ─────────────────────────────────────────────

def verify_zero_score(category: str, student_answer: str, correct_answer: str, score: int) -> int:
    if category not in ZERO_CATEGORIES or score == 0:
        return score

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a strict mark scheme checker. Reply with only YES or NO. No other text."
            },
            {
                "role": "user",
                "content": (
                    f"Mark scheme answer: {correct_answer}\n"
                    f"Student answer: {student_answer}\n\n"
                    f"Do the EXACT values, facts, and figures in the student answer match "
                    f"the mark scheme? Even if the method looks right, if any specific "
                    f"number or fact is different from the mark scheme, answer NO. "
                    f"Reply YES only if everything matches exactly."
                )
            }
        ],
        temperature=0,
        max_tokens=5
    )

    verdict = response.choices[0].message.content.strip().upper()
    if "YES" in verdict:
        return 0
    return score


# ─────────────────────────────────────────────
# STEP 4: PASS/FAIL LOGIC
# ─────────────────────────────────────────────

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

    if category in correct_categories:
        return score == max_marks
    elif category in partial_categories:
        return 0 < score < max_marks or (max_marks == 1 and score == 0)
    elif category in ZERO_CATEGORIES:
        return score == 0
    return True


# ─────────────────────────────────────────────
# STEP 5: EVALUATE SINGLE CATEGORY (parallel)
# ─────────────────────────────────────────────

def evaluate_single_category(
    category: str,
    student_answer: str,
    identity: str,
    max_marks: int,
    correct_answer: str,
) -> tuple[str, dict]:

    if category == "Partially Correct with Incorrect Information" and max_marks == 1:
        return category, {
            "score": 0, "max": max_marks, "pass": True,
            "answer": "N/A — skipped for 1-mark questions",
            "feedback": "Category not applicable for 1-mark questions.",
        }

    if not student_answer:
        return category, {
            "score": 0, "max": max_marks, "pass": False,
            "answer": "", "feedback": ""
        }

    eval_result = run_evaluator(identity, student_answer, max_marks, category)
    verified_score = verify_zero_score(category, student_answer, correct_answer, eval_result["score"])
    eval_result["score"] = verified_score
    passed = evaluate_category_result(category, eval_result["score"], max_marks)

    return category, {
        "score":    eval_result["score"],
        "max":      max_marks,
        "pass":     passed,
        "answer":   student_answer,
        "feedback": eval_result["feedback"],
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_tester(overwrite: bool = False, single_row: int = None, paper_label: str = ""):
    print("\n🧪 Stage 4 — Automated testing...")
    sheet = get_sheet(paper_label)
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

        print(f"      Running evaluations in parallel...")
        results = {}

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(
                    evaluate_single_category,
                    category,
                    samples.get(category, ""),
                    identity,
                    max_marks,
                    answer,
                ): category
                for category in CATEGORIES
            }

            for future in as_completed(futures):
                category, result = future.result()
                results[category] = result
                skip = result["answer"] == "N/A — skipped for 1-mark questions"
                if skip:
                    print(f"      ⏭️  {category}: skipped (1-mark)")
                else:
                    status = "✅" if result["pass"] else "❌"
                    print(f"      {status} {category}: {result['score']}/{result['max']}")

        all_passed = all(r["pass"] for r in results.values())

        scoring_data = {
            "overall_pass": all_passed,
            "status": "approved" if all_passed else "needs_review",
            "results": results,
            "samples": samples,
        }

        sheet.update_cell(row_num, COL_SCORING, json.dumps(scoring_data))
        print(f"      {'✅ ALL PASSED' if all_passed else '❌ NEEDS REVIEW'}")
        tested += 1

    print(f"\n── Testing complete ──")
    print(f"   ✅ Tested: {tested} | ⏭️  Skipped: {skipped} | Total: {total + skipped}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run automated evaluation testing")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--row", type=int, default=None)
    parser.add_argument("--paper", type=str, default="", help="Paper label (sheet tab name)")
    args = parser.parse_args()
    run_tester(overwrite=args.overwrite, single_row=args.row, paper_label=args.paper)