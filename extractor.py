"""
GCSE Exam PDF Extraction Engine
Extracts questions, sub-parts, marks, answers, and figures from GCSE exam PDFs.
Uploads figures to Google Drive and writes structured data to Google Sheets.
"""

import os
import re
import json
import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

GDRIVE_FOLDER_ID = os.getenv("GDRIVE_FOLDER_ID")       # Google Drive folder for figures
GSHEET_ID        = os.getenv("GSHEET_ID")               # Google Sheet ID
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")

# Column order matching your existing sheet
SHEET_HEADERS = [
    "Row",
    "Subtopic Name",
    "Question",
    "Marks",
    "Answer",
    "Extra Information",
    "Table",          # Google Drive link to figure image (if any)
    "Rewriting",
    "Prompts",
    "Identity+ Prompt+ Query",
    "Scoring",
    "Paper",
    "Question Number",
    "Sub Part",
]


# ─────────────────────────────────────────────
# GOOGLE DRIVE UPLOAD
# ─────────────────────────────────────────────

def get_drive_service():
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    creds = service_account.Credentials.from_service_account_file(
        os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "service_account.json"),
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds)


def upload_image_to_drive(image_path: str, filename: str, drive_service) -> str:
    """Upload an image to Google Drive and return a viewable URL.
    Uses supportsAllDrives so it works with both shared drives and regular folders
    where the service account has been granted access via folder sharing.
    """
    from googleapiclient.http import MediaFileUpload
    # For Shared Drives, driveId must match the parent
    file_metadata = {
        "name": filename,
        "parents": [GDRIVE_FOLDER_ID],
        "driveId": GDRIVE_FOLDER_ID
    }
    media = MediaFileUpload(image_path, mimetype="image/png")
    uploaded = drive_service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id",
        supportsAllDrives=True
    ).execute()
    file_id = uploaded.get("id")
    # Make it viewable by anyone with the link
    try:
        drive_service.permissions().create(
            fileId=file_id,
            body={"role": "reader", "type": "anyone"},
            supportsAllDrives=True
        ).execute()
    except Exception:
        pass  # Permission may already be inherited from shared drive
    return f"https://drive.google.com/file/d/{file_id}/view"


# ─────────────────────────────────────────────
# GOOGLE SHEETS WRITE
# ─────────────────────────────────────────────

def get_sheet(sheet_id: str):
    import gspread
    from google.oauth2 import service_account
    creds = service_account.Credentials.from_service_account_file(
        os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "service_account.json"),
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
    )
    gc = gspread.authorize(creds)
    return gc.open_by_key(sheet_id).sheet1


def ensure_headers(sheet):
    """Write headers if the sheet is empty."""
    existing = sheet.row_values(1)
    if not existing or existing[0] != SHEET_HEADERS[0]:
        sheet.insert_row(SHEET_HEADERS, index=1)


def append_rows_to_sheet(sheet, rows: list[dict], paper_label: str = ""):
    """Write extracted question rows to the Google Sheet, replacing existing rows for this paper."""
    ensure_headers(sheet)

    # If paper_label provided, delete existing rows for this paper first
    if paper_label:
        all_rows = sheet.get_all_values()
        rows_to_delete = []
        for i, row in enumerate(all_rows[1:], start=2):  # skip header
            if len(row) >= 13 and row[12] == paper_label:
                rows_to_delete.append(i)
        # Delete in reverse order to preserve row numbers
        for row_num in reversed(rows_to_delete):
            sheet.delete_rows(row_num)
        if rows_to_delete:
            print(f"🗑️  Deleted {len(rows_to_delete)} existing rows for '{paper_label}'")

    values = []
    for i, row in enumerate(rows, start=1):
        values.append([
            i,
            row.get("subtopic", ""),
            row.get("question", ""),
            row.get("marks", ""),
            row.get("answer", ""),
            row.get("extra_info", ""),
            row.get("figure_url", ""),
            "",  # Rewriting — filled in Stage 2
            "",  # Prompts — filled in Stage 3
            "",  # Identity+Prompt+Query — filled in Stage 3
            "",  # Scoring — filled in Stage 4
            row.get("paper", ""),
            row.get("question_number", ""),
            row.get("sub_part", ""),
        ])
    sheet.append_rows(values, value_input_option="RAW")
    print(f"✅ Written {len(values)} rows to Google Sheet.")


# ─────────────────────────────────────────────
# GPT-4o MINI — STRUCTURED EXTRACTION
# ─────────────────────────────────────────────

def extract_questions_with_gpt(page_text: str, page_images_b64: list[str], paper_label: str) -> list[dict]:
    """
    Send page text (and optionally page images) to GPT-4o mini.
    Returns a list of structured question dicts.
    """
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    system_prompt = """You are an expert GCSE exam paper parser.
You will be given text (and possibly images) from one page of a GCSE exam paper.
Your job is to extract every REAL exam question and sub-part from this page.

SKIP all of the following — return an empty array [] for pages that contain only:
- Cover page fields (Surname, Forename, Centre number, Candidate number, signature)
- General instructions (e.g. "Answer all questions", "Write in black ink")
- Page headers, footers, logos, exam board branding
- Blank pages or filler pages
- "Do not write outside the box" or similar instructions
- "Turn over" instructions

Only extract content that is an actual exam question a student must answer.

Return a JSON array. Each element must have these exact keys:
- question_number: string — use AQA format exactly as printed e.g. "01.1", "01.2", "02.3", "03.4"
  - Remove any spaces from the number e.g. "0 1 . 1" -> "01.1"
  - Always use two digits for the question e.g. "01", "02", "10"
  - If there is no sub-part, just use the question number e.g. "01", "02"
- sub_part: always set to "" — the sub-part is already embedded in the question_number
- question_text: the full question text exactly as written, including any bullet points, tables, or context given. IMPORTANT: For tick box or multiple choice questions, you MUST include ALL the answer options exactly as printed below the question (e.g. "Organ\nOrganism\nOrgan system"). Look carefully at the page image to find these options. Do NOT include the mark count in the question text.
- marks: integer — extract from [X marks] or [X mark] shown in the question. Must be accurate.
- figure_ref: string — if the question says "Figure 1", "Table 2", "the diagram" etc., write that label here, else ""
- extra_info: string — any extra instructions specific to this question, else ""

Rules:
- Preserve the exact question text including bullet points and formatting
- If a figure applies to multiple sub-parts, include the same figure_ref in each sub-part
- Do NOT invent answers
- Return ONLY valid JSON array, no markdown, no explanation
- If this page has no real exam questions, return []

Paper: """ + paper_label

    content = [{"type": "text", "text": f"Page content:\n\n{page_text}"}]
    for b64 in page_images_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}
        })

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ],
        temperature=0,
        max_tokens=4000
    )

    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if present
    raw = re.sub(r"^```json|^```|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"⚠️  JSON parse error on page: {e}\nRaw response: {raw[:300]}")
        return []


def extract_answers_with_gpt(page_text: str, paper_label: str) -> list[dict]:
    """
    Extract answers from the marking scheme pages.
    Returns list of {question_number, sub_part, answer, extra_info, marks}
    """
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    system_prompt = """You are an expert GCSE marking scheme parser.
You will be given text from a GCSE marking scheme page.
Extract every answer entry.

Return a JSON array. Each element must have:
- question_number: string e.g. "1", "10.4"
- sub_part: string e.g. "a", "b", or ""
- answer: the correct answer(s), exactly as written
- extra_info: any "allow", "accept", "do not accept" notes
- marks: integer marks for this answer point

Return ONLY valid JSON, no markdown, no explanation."""

    response_obj = __import__("openai").OpenAI(api_key=OPENAI_API_KEY)
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Marking scheme page:\n\n{page_text}"}
        ],
        temperature=0,
        max_tokens=4000
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```json|^```|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []


# ─────────────────────────────────────────────
# FIGURE EXTRACTION
# ─────────────────────────────────────────────

def extract_figures_from_page(pdf_path: str, page_num: int, output_dir: str) -> dict[str, str]:
    """
    Extract figures from a page using PyMuPDF.
    Returns dict: {figure_label -> local_image_path}
    Also rasterizes the whole page to detect vector figures.
    """
    import base64
    import subprocess

    figures = {}
    os.makedirs(output_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    page = doc[page_num]

    # Extract raster images embedded in the page
    image_list = page.get_images(full=True)
    for img_index, img in enumerate(image_list):
        xref = img[0]
        pix = fitz.Pixmap(doc, xref)
        if pix.n - pix.alpha > 3:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        # Only save images that are large enough to be real figures (not decorative)
        if pix.width > 80 and pix.height > 80:
            img_path = os.path.join(output_dir, f"page{page_num+1}_img{img_index}.png")
            pix.save(img_path)
            figures[f"page{page_num+1}_img{img_index}"] = img_path

    # Rasterize the whole page to capture vector graphics too
    page_img_path = os.path.join(output_dir, f"page{page_num+1}_full.png")
    mat = fitz.Matrix(2, 2)  # 2x zoom = ~144 DPI
    clip = page.get_pixmap(matrix=mat)
    clip.save(page_img_path)
    figures[f"page{page_num+1}_full"] = page_img_path

    doc.close()
    return figures


def page_to_base64(image_path: str) -> str:
    import base64
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ─────────────────────────────────────────────
# MAIN EXTRACTION PIPELINE
# ─────────────────────────────────────────────

def detect_paper_label(pdf_path: str) -> str:
    """Try to auto-detect paper label from filename or first page."""
    filename = Path(pdf_path).stem
    # e.g. "biology-paper-1-2023" -> "Biology Paper 1 2023"
    label = filename.replace("-", " ").replace("_", " ").title()
    return label


def is_marking_scheme_page(text: str) -> bool:
    """Heuristic: marking scheme pages contain these patterns."""
    markers = ["mark scheme", "marking scheme", "marking guidelines",
               "award", "allow", "accept", "do not accept", "ignore"]
    text_lower = text.lower()
    return sum(1 for m in markers if m in text_lower) >= 2


def extract_pdf(
    question_pdf_path: str,
    marking_pdf_path: str,
    paper_label: str = None,
    output_dir: str = "/tmp/gcse_figures",
    upload_to_drive: bool = True,
    write_to_sheet: bool = True,
) -> list[dict]:
    """
    Main entry point. Extracts questions from question_pdf and answers from marking_pdf.
    Returns list of merged row dicts ready for Google Sheets.
    """

    if not paper_label:
        paper_label = detect_paper_label(question_pdf_path)

    print(f"\n📄 Processing: {paper_label}")
    print(f"   Questions PDF: {question_pdf_path}")
    print(f"   Mark Scheme:   {marking_pdf_path}")

    os.makedirs(output_dir, exist_ok=True)

    # ── Step A: Extract questions page by page ──
    all_questions = []
    figure_map = {}  # question_number+sub_part -> drive_url

    drive_service = None
    if upload_to_drive and GDRIVE_FOLDER_ID:
        drive_service = get_drive_service()

    with pdfplumber.open(question_pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"   Pages in question paper: {total_pages}")

        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if not text.strip():
                print(f"   ⚠️  Page {page_num+1}: no text extracted, skipping")
                continue

            print(f"   Extracting page {page_num+1}/{total_pages}...")

            # Rasterize page for GPT vision (always needed so GPT can see the layout)
            figs = extract_figures_from_page(question_pdf_path, page_num, output_dir)
            full_page_img = figs.get(f"page{page_num+1}_full")

            # Send to GPT-4o mini for structured extraction
            b64_images = []
            if full_page_img and os.path.exists(full_page_img):
                b64_images = [page_to_base64(full_page_img)]

            page_questions = extract_questions_with_gpt(text, b64_images, paper_label)

            # Only upload to Drive if at least one question references a figure
            page_has_figure = any(q.get("figure_ref") for q in page_questions)
            page_fig_url = ""

            if page_has_figure and full_page_img and drive_service:
                fname = f"{paper_label.replace(' ', '_')}_page{page_num+1}.png"
                print(f"   📎 Uploading figure for page {page_num+1} to Drive...")
                page_fig_url = upload_image_to_drive(full_page_img, fname, drive_service)

            for q in page_questions:
                q["paper"] = paper_label
                q["page"] = page_num + 1
                if q.get("figure_ref") and page_fig_url:
                    q["figure_url"] = page_fig_url
                else:
                    q["figure_url"] = ""
                all_questions.append(q)

    print(f"   ✅ Extracted {len(all_questions)} questions from question paper")

    # ── Step B: Extract answers from marking scheme ──
    all_answers = []
    with pdfplumber.open(marking_pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"   Pages in mark scheme: {total_pages}")
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            page_answers = extract_answers_with_gpt(text, paper_label)
            all_answers.extend(page_answers)

    print(f"   ✅ Extracted {len(all_answers)} answer entries from mark scheme")

    # ── Step C: Merge questions with answers ──

    def normalise_qnum(raw: str) -> str:
        """Normalise AQA question number: remove spaces e.g. '0 1 . 1' -> '01.1'"""
        return re.sub(r"\s+", "", str(raw)).strip()

    # Build answer lookup keyed by normalised question_number
    answer_lookup = {}
    for a in all_answers:
        key = normalise_qnum(a.get("question_number", ""))
        if key not in answer_lookup:
            answer_lookup[key] = a
        else:
            existing = answer_lookup[key]
            existing["answer"] = existing.get("answer", "") + "\n" + a.get("answer", "")
            existing["extra_info"] = existing.get("extra_info", "") + " " + a.get("extra_info", "")

    merged_rows = []
    for q in all_questions:
        qnum = normalise_qnum(q.get("question_number", ""))
        answer_data = answer_lookup.get(qnum, {})
        # Use marks from question text if mark scheme doesn't have it
        q_marks = q.get("marks", 0)
        ms_marks = answer_data.get("marks", 0)
        final_marks = ms_marks if ms_marks and ms_marks > 0 else q_marks

        row = {
            "paper":           q.get("paper", paper_label),
            "question_number": qnum,
            "sub_part":        "",  # Always blank — sub-part embedded in AQA question number
            "subtopic":        "",
            "question":        q.get("question_text", ""),
            "marks":           final_marks,
            "answer":          answer_data.get("answer", ""),
            "extra_info":      answer_data.get("extra_info", q.get("extra_info", "")),
            "figure_url":      q.get("figure_url", ""),
        }
        merged_rows.append(row)

    print(f"   ✅ Merged into {len(merged_rows)} rows")

    # ── Step D: Write to Google Sheets ──
    if write_to_sheet and GSHEET_ID:
        sheet = get_sheet(GSHEET_ID)
        append_rows_to_sheet(sheet, merged_rows, paper_label=paper_label)

    return merged_rows


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract GCSE exam questions into Google Sheets")
    parser.add_argument("question_pdf",  help="Path to the GCSE question paper PDF")
    parser.add_argument("marking_pdf",   help="Path to the marking scheme PDF")
    parser.add_argument("--paper",       help="Paper label e.g. 'Biology Paper 1 2023'", default=None)
    parser.add_argument("--output-dir",  help="Directory for extracted figures", default="/tmp/gcse_figures")
    parser.add_argument("--no-drive",    action="store_true", help="Skip Google Drive upload")
    parser.add_argument("--no-sheet",    action="store_true", help="Skip Google Sheets write")
    parser.add_argument("--dry-run",     action="store_true", help="Print results only, no uploads")

    args = parser.parse_args()

    rows = extract_pdf(
        question_pdf_path=args.question_pdf,
        marking_pdf_path=args.marking_pdf,
        paper_label=args.paper,
        output_dir=args.output_dir,
        upload_to_drive=not args.no_drive and not args.dry_run,
        write_to_sheet=not args.no_sheet and not args.dry_run,
    )

    if args.dry_run:
        print("\n── DRY RUN OUTPUT ──")
        for r in rows[:5]:
            print(json.dumps(r, indent=2))
        print(f"\n... ({len(rows)} total rows)")