# GCSE Evaluation Automation — Setup Guide

## Prerequisites
- Python 3.10+
- A Google Cloud project with Drive & Sheets APIs enabled
- A Google Service Account with editor access to your Sheet and Drive folder
- An OpenAI API key

---

## Step 1: Install dependencies

```bash
pip install pymupdf pdfplumber gspread google-auth google-auth-oauthlib google-api-python-client openai python-dotenv
```

---

## Step 2: Set up Google Cloud (one-time)

1. Go to https://console.cloud.google.com
2. Create a new project (or use existing)
3. Enable these APIs:
   - Google Drive API
   - Google Sheets API
4. Go to IAM & Admin → Service Accounts → Create Service Account
5. Give it a name e.g. "gcse-automation"
6. Download the JSON key file → save as `service_account.json` in this folder
7. Share your Google Sheet with the service account email (Editor access)
8. Share your Google Drive folder with the service account email (Editor access)

---

## Step 3: Configure environment

```bash
cp .env.example .env
# Edit .env and fill in your values
```

---

## Step 4: Run extraction

```bash
# Basic usage
python extractor.py path/to/questions.pdf path/to/markscheme.pdf --paper "Biology Paper 1 2023"

# Dry run (no uploads, just print results)
python extractor.py questions.pdf markscheme.pdf --dry-run

# Skip Google Drive upload (local only)
python extractor.py questions.pdf markscheme.pdf --no-drive

# Custom output directory for figures
python extractor.py questions.pdf markscheme.pdf --output-dir /tmp/my_figures
```

---

## How it works

1. **Question extraction**: Each page is sent to GPT-4o mini with its text AND a rendered image of the page. This means vector diagrams, tables, and figures are all visible to the AI.

2. **Figure handling**: Every page is rasterized at 144 DPI. If a question references a figure, the full page image is uploaded to Google Drive and linked in the "Table" column.

3. **Shared figures**: If Figure 1 applies to Q3a, Q3b, and Q3c — GPT detects the `figure_ref` field for each sub-part and the same Drive URL is stored in each row.

4. **Answer merging**: The marking scheme is processed separately and merged by (question_number, sub_part) key.

5. **Google Sheets**: Results are appended to your sheet with all columns matching your existing template.

---

## Column mapping

| Sheet Column | Source |
|---|---|
| Subtopic Name | Left blank (fill manually or via subject mapping) |
| Question | Extracted from question paper |
| Marks | From marking scheme |
| Answer | From marking scheme |
| Extra Information | "allow/accept" notes from marking scheme |
| Table | Google Drive link to figure image |
| Rewriting | Filled in Stage 2 |
| Prompts | Filled in Stage 3 |
| Identity+Prompt+Query | Filled in Stage 3 |
| Scoring | Filled in Stage 4 |
| Paper | Auto-detected from filename or --paper flag |
| Question Number | e.g. "1", "10.4" |
| Sub Part | e.g. "a", "b", "i" |

---

## Troubleshooting

**"No text extracted" on a page**: The page may be image-only. The system still sends the rasterized image to GPT which can read it visually.

**Figures not linking correctly**: Check that the question PDF has figures embedded as images (use `pdfimages -list your.pdf` to verify).

**Google auth errors**: Make sure the service account JSON is in the right path and the service account has been shared on both the Drive folder and the Sheet.