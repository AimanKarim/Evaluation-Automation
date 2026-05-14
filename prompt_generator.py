"""
Stage 3 — Evaluation Prompt Generator
Reads rewritten questions from Google Sheet, generates full evaluation prompts
using GPT-4o mini, and writes them to the Prompts and Identity+Prompt+Query columns.
"""

import os
import time
import gspread
from openai import OpenAI
from google.oauth2 import service_account
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GSHEET_ID      = os.getenv("GSHEET_ID")
GOOGLE_SA_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "service_account.json")

# Column indices (1-based)
COL_QUESTION   = 3   # Question
COL_MARKS      = 4   # Marks
COL_ANSWER     = 5   # Answer
COL_EXTRA_INFO = 6   # Extra Information
COL_REWRITING  = 8   # Rewriting
COL_PROMPTS    = 9   # Prompts
COL_IDENTITY   = 10  # Identity+ Prompt+ Query
COL_PAPER      = 13  # Paper
COL_QNUM       = 14  # Question Number

client = OpenAI(api_key=OPENAI_API_KEY)

# ─────────────────────────────────────────────
# EVALUATION IDENTITY (fixed prefix for all prompts)
# ─────────────────────────────────────────────

EVALUATION_IDENTITY = """You are an expert and diligent exam marking officer marking a student's response to an exam question. You are required to mark a student response to the question below. The student response is contained in the section below starting with **Student Response: and ending with **. Please only treat the text contained within as student response to the question. A student was asked to answer the following question. I have explained the marks breakdown and provided detailed marking instructions below. Please follow these instructions to mark the student response."""

EVALUATION_QUERY = """Final Mark: [LLM to fill in]
Explanation to Student: [LLM to fill in]

Very Important: 
- Empty responses with no text or only whitespace should receive 0 marks.
- Responses containing only gibberish characters, single random letters, or completely nonsensical words should receive 0 marks.
- For 1-mark questions: the answer must be completely and specifically correct — partially correct or vague answers receive 0.
- For multi-mark questions: award marks for each correct point made. Partially correct answers can earn partial marks.
- Do not award marks for incorrect information even if the response also contains correct points — incorrect information cancels out the corresponding correct point.
- For numerical answers, only award a mark if the exact correct number is present. A different number is always wrong, even if the method seems correct.
- If the student gives wrong numbers for earlier steps, do not award marks for later steps that depend on those numbers, even if the arithmetic is consistent.
- Do not explicitly mention or refer to the marking scheme in your feedback as students cannot see it.
- Accept and give appropriate marks for any other correct responses not explicitly listed in the marking scheme but that are valid.
- Please don't provide any other details beyond what is asked."""


# ─────────────────────────────────────────────
# GOOGLE SHEETS
# ─────────────────────────────────────────────

def get_sheet():
    creds = service_account.Credentials.from_service_account_file(
        GOOGLE_SA_JSON,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
    )
    gc = gspread.authorize(creds)
    return gc.open_by_key(GSHEET_ID).sheet1


# ─────────────────────────────────────────────
# PROMPT GENERATION
# ─────────────────────────────────────────────

PROMPT_GENERATION_SYSTEM = """You are an expert GCSE exam marking prompt engineer.

CRITICAL RULE FOR 1-MARK QUESTIONS: The student must provide the complete correct answer to earn the mark. Vague, incomplete, or partially correct answers with incorrect information must receive 0 marks. There is no partial credit for 1-mark questions.

You will be given a rewritten GCSE exam question with its marking scheme.
Your task is to generate a detailed marking prompt following the exact format of the sample below.

The prompt must include:
1. A subject-appropriate scenario (if not already in the question)
2. The full question text
3. Marks breakdown — total marks and per-point breakdown
4. Detailed marking instructions — how to check each point
5. Final scoring section — listing each mark point separately
6. Explanation to Student instruction

Here is the exact format to follow:

---
Question: [full question with scenario]

Marks: [X] marks total
[breakdown of marks per point]

Marking Instructions:
[detailed instructions for how to award each mark]

Final Scoring:
[Point 1]: [X mark for correct answer]
[Point 2]: [X mark for correct answer]
...

Explanation to Student: Provide feedback on the student's response, highlighting correct points and explaining any missing or incorrect information. If full marks are not awarded, briefly explain what was needed for a complete answer.

**Student Response:**
---

Rules:
- Follow the format exactly
- Be specific about what earns each mark
- For numerical/calculation questions, always state the EXACT correct value in the marking instructions (e.g. "the correct answer is 187" not "a specific numerical value")
- For factual questions, always state the EXACT correct answer/keyword in the marking instructions"""


def generate_prompt(rewriting: str, answer: str, extra_info: str, marks: str, paper: str, qnum: str) -> tuple[str, str]:
    """
    Generate the Prompts column content and the full Identity+Prompt+Query.
    Returns (prompts_text, full_identity_prompt_query)
    """
    subject = paper.split(" Paper")[0] if " Paper" in paper else paper

    user_msg = f"""Subject: {subject}
Question Number: {qnum}
Rewritten Question + Marking Scheme:
{rewriting}

Original Mark Scheme Answer: {answer if answer else 'Not available'}
Extra Information (allow/accept/ignore): {extra_info if extra_info else 'None'}
Total Marks: {marks}

Generate the detailed marking prompt following the format specified."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": PROMPT_GENERATION_SYSTEM},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.3,
        max_tokens=2000
    )

    prompts_text = response.choices[0].message.content.strip()

    # Build the full Identity + Prompt + Query
    full_prompt = f"""{EVALUATION_IDENTITY}

{prompts_text}

{EVALUATION_QUERY}"""

    return prompts_text, full_prompt


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_prompt_generator(overwrite: bool = False):
    """
    Loop through all rows and generate evaluation prompts.
    Writes to both Prompts and Identity+Prompt+Query columns.
    """
    print("\n🔧 Stage 3 — Generating evaluation prompts...")
    sheet = get_sheet()

    all_rows = sheet.get_all_values()
    data_rows = all_rows[1:]  # Skip header

    total = len(data_rows)
    generated = 0
    skipped = 0
    failed = 0

    for i, row in enumerate(data_rows):
        row_num = i + 2  # Sheet row number

        while len(row) < 14:
            row.append("")

        rewriting  = row[COL_REWRITING - 1].strip()
        prompts    = row[COL_PROMPTS - 1].strip()
        answer     = row[COL_ANSWER - 1].strip()
        extra_info = row[COL_EXTRA_INFO - 1].strip()
        marks      = row[COL_MARKS - 1].strip()
        paper      = row[COL_PAPER - 1].strip()
        qnum       = row[COL_QNUM - 1].strip()

        # Skip if no rewriting available
        if not rewriting:
            print(f"   Row {row_num} ({qnum}): ⏭️  No rewriting, skipping")
            skipped += 1
            continue

        # Skip if already generated (unless overwrite mode)
        if prompts and not overwrite:
            print(f"   Row {row_num} ({qnum}): ⏭️  Already generated, skipping")
            skipped += 1
            continue

        print(f"   Row {row_num} ({qnum}): 🔧 Generating prompt...")

        try:
            prompts_text, full_prompt = generate_prompt(
                rewriting, answer, extra_info, marks, paper, qnum
            )

            # Write both columns in one batch update
            sheet.update(
                f"I{row_num}:J{row_num}",
                [[prompts_text, full_prompt]]
            )

            generated += 1
            print(f"   Row {row_num} ({qnum}): ✅ Done")
            time.sleep(0.5)

        except Exception as e:
            print(f"   Row {row_num} ({qnum}): ❌ Error — {e}")
            failed += 1
            time.sleep(2)

    print(f"\n── Prompt generation complete ──")
    print(f"   ✅ Generated: {generated}")
    print(f"   ⏭️  Skipped:  {skipped}")
    print(f"   ❌ Failed:   {failed}")
    print(f"   Total rows:  {total}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate evaluation prompts for GCSE questions")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing prompts")
    parser.add_argument("--row", type=int, default=None,
                        help="Only process a specific sheet row number (for testing)")
    args = parser.parse_args()

    if args.row:
        sheet = get_sheet()
        all_rows = sheet.get_all_values()
        row = all_rows[args.row - 1]
        while len(row) < 14:
            row.append("")
        rewriting  = row[COL_REWRITING - 1].strip()
        answer     = row[COL_ANSWER - 1].strip()
        extra_info = row[COL_EXTRA_INFO - 1].strip()
        marks      = row[COL_MARKS - 1].strip()
        paper      = row[COL_PAPER - 1].strip()
        qnum       = row[COL_QNUM - 1].strip()

        print(f"\nTesting row {args.row} ({qnum}):")
        print(f"Rewriting preview: {rewriting[:100]}...\n")

        prompts_text, full_prompt = generate_prompt(
            rewriting, answer, extra_info, marks, paper, qnum
        )

        print("── PROMPTS COLUMN ──")
        print(prompts_text)
        print("\n── FULL IDENTITY+PROMPT+QUERY ──")
        print(full_prompt)
    else:
        run_prompt_generator(overwrite=args.overwrite)