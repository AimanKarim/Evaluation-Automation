"""
Stage 2 — AI Question Rewriter
Reads questions from Google Sheet, paraphrases them using GPT-4o mini,
and writes the rewritten version back to the "Rewriting" column.
"""

import os
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

COL_QUESTION   = 3
COL_MARKS      = 4
COL_ANSWER     = 5
COL_EXTRA_INFO = 6
COL_REWRITING  = 8
COL_PAPER      = 13
COL_QNUM       = 14

client = OpenAI(api_key=OPENAI_API_KEY)


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


REWRITE_SYSTEM_PROMPT = """You are an expert GCSE question writer. Your task is to paraphrase GCSE exam questions into a cleaner format while strictly preserving all original content.

CRITICAL RULES - these must NEVER be broken:
- For tick box / multiple choice questions, copy the answer options EXACTLY word-for-word from the original. Do NOT change, add, remove, or reorder any options under any circumstances.
- Do NOT invent new answer options. Only use what is in the original question.
- Do NOT change what the question is testing.
- Do NOT add or remove marks.

Other rules:
- Keep the same number of marks
- Preserve all scientific/subject-specific terminology exactly
- Add a short subject-appropriate scenario before the question to give real-world context
- Keep the question type identical (tick box stays tick box, explain stays explain, calculate stays calculate)
- Include the mark count at the end e.g. [1 mark] or [2 marks]
- Format the marking scheme clearly as:
  Marking scheme [X marks]:
  - point 1 [1]
  - point 2 [1]
- Write in clear, accessible English suitable for GCSE students
- Return ONLY the rewritten question + marking scheme, nothing else"""


def rewrite_question(question: str, marks: str, answer: str, extra_info: str, paper: str, qnum: str) -> str:
    subject = paper.split(" Paper")[0] if " Paper" in paper else paper
    answer_section = answer if answer else "NOT PROVIDED - do not guess, leave marking points blank"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": f"""Subject: {subject}
Question Number: {qnum}
Original Question: {question}
Marks: {marks}
Correct Answer / Mark Scheme: {answer_section}
Extra Information (allow/accept notes): {extra_info}

IMPORTANT: Only include marking scheme points from the Correct Answer provided above. If the answer says "NOT PROVIDED", write "Marking scheme [X marks]: - [answer not available]" and do not guess.

Please rewrite this question with a subject-appropriate scenario and format the marking scheme clearly."""}
        ],
        temperature=0.7,
        max_tokens=1000
    )
    return response.choices[0].message.content.strip()


def run_rewriter(overwrite: bool = False):
    print("\n✏️  Stage 2 — Rewriting questions...")
    sheet = get_sheet()
    all_rows = sheet.get_all_values()
    data_rows = all_rows[1:]

    rewritten = skipped = failed = 0

    for i, row in enumerate(data_rows):
        row_num = i + 2
        while len(row) < 14:
            row.append("")

        question   = row[COL_QUESTION - 1].strip()
        marks      = row[COL_MARKS - 1].strip()
        answer     = row[COL_ANSWER - 1].strip()
        extra_info = row[COL_EXTRA_INFO - 1].strip()
        rewriting  = row[COL_REWRITING - 1].strip()
        paper      = row[COL_PAPER - 1].strip()
        qnum       = row[COL_QNUM - 1].strip()

        if not question:
            skipped += 1
            continue

        if rewriting and not overwrite:
            print(f"   Row {row_num} ({qnum}): ⏭️  Already rewritten, skipping")
            skipped += 1
            continue

        print(f"   Row {row_num} ({qnum}): ✏️  Rewriting...")
        try:
            text = rewrite_question(question, marks, answer, extra_info, paper, qnum)
            sheet.update_cell(row_num, COL_REWRITING, text)
            rewritten += 1
            print(f"   Row {row_num} ({qnum}): ✅ Done")
            time.sleep(0.5)
        except Exception as e:
            print(f"   Row {row_num} ({qnum}): ❌ Error — {e}")
            failed += 1
            time.sleep(2)

    print(f"\n── Rewriting complete ──")
    print(f"   ✅ Rewritten: {rewritten} | ⏭️  Skipped: {skipped} | ❌ Failed: {failed}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--row", type=int, default=None)
    args = parser.parse_args()

    if args.row:
        sheet = get_sheet()
        row = sheet.get_all_values()[args.row - 1]
        while len(row) < 14:
            row.append("")
        result = rewrite_question(
            row[COL_QUESTION-1], row[COL_MARKS-1], row[COL_ANSWER-1],
            row[COL_EXTRA_INFO-1], row[COL_PAPER-1], row[COL_QNUM-1]
        )
        print(f"\nRewritten:\n{result}")
    else:
        run_rewriter(overwrite=args.overwrite)