from mistralai import Mistral
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)

# folder path(containing pdfs)
base_dir = "task_2/test_dir/img_table_blanks_brail_test.pdf"

# output file txt containg markdown
output_path = "./ocr_output/mistral_batch_ocr_output.txt"

# Helper: get all PDF file paths recursively
def get_all_pdfs(folder):
    pdf_files = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, f))
    return pdf_files


pdf_files = get_all_pdfs(base_dir)
print(f"Found {len(pdf_files)} PDF files in '{base_dir}'\n")

if not pdf_files:
    print("⚠️ No PDF files found. Exiting.")
    exit()

with open(output_path, "w", encoding="utf-8") as output_file:
    for idx, pdf_path in enumerate(pdf_files, start=1):
        print(f"📄 [{idx}/{len(pdf_files)}] Processing: {pdf_path}")

        try:
            # Upload file to Mistral
            uploaded_file = client.files.upload(
                file={
                    "file_name": os.path.basename(pdf_path),
                    "content": open(pdf_path, "rb")
                },
                purpose="ocr"
            )

            # Get signed URL
            file_url = client.files.get_signed_url(file_id=uploaded_file.id)

            # Run OCR
            response = client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": file_url.url
                },
                include_image_base64=False
            )

            # Write OCR output to text file
            output_file.write(f"\n\n==============================\n")
            output_file.write(f"📘 FILE: {pdf_path}\n")
            output_file.write("==============================\n\n")

            for page_num, page in enumerate(response.pages, start=1):
                page_text = getattr(page, "markdown", "") or getattr(page, "text", "")
                output_file.write(f"\n--- Page {page_num} ---\n\n{page_text.strip()}\n")

            print(f"✅ Completed OCR for {pdf_path}\n")

        except Exception as e:
            print(f"❌ Error processing {pdf_path}: {e}")
            output_file.write(f"\n\n[ERROR processing {pdf_path}: {e}]\n")

print(f"\n🎉 All OCR text saved to '{output_path}'")