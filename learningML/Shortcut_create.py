import os
import csv
import json
import shutil
from fpdf import FPDF
from PIL import Image
import markdown
import pdfkit

# 1. Create a Text File
def create_text_file(file_name, content):
    with open(file_name, 'w') as file:
        file.write(content)
    print(f"Text file '{file_name}' created.")

# 2. Create a PDF from Text
def create_pdf(file_name, content):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, content)
    pdf.output(file_name)
    print(f"PDF file '{file_name}' created.")

# 3. Convert an Image to PDF
def image_to_pdf(image_path, pdf_path):
    image = Image.open(image_path)
    image.convert("RGB").save(pdf_path)
    print(f"Image '{image_path}' converted to PDF '{pdf_path}'.")

# 4. Convert Markdown to PDF
def markdown_to_pdf(md_file, pdf_file):
    with open(md_file, 'r') as file:
        html_content = markdown.markdown(file.read())
    create_pdf(pdf_file, html_content)
    print(f"Markdown file '{md_file}' converted to PDF '{pdf_file}'.")

# 5. Create a URL Shortcut
def create_url_shortcut(file_name, url):
    with open(file_name, 'w') as file:
        file.write(f"[InternetShortcut]\nURL={url}\n")
    print(f"URL shortcut '{file_name}' created.")

# 6. Create a CSV File
def create_csv_file(file_name, data, headers=None):
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        if headers:
            writer.writerow(headers)
        writer.writerows(data)
    print(f"CSV file '{file_name}' created.")

# 7. Create a JSON File
def create_json_file(file_name, data):
    with open(file_name, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"JSON file '{file_name}' created.")

# 8. Convert JSON to CSV
def json_to_csv(json_file, csv_file):
    with open(json_file, 'r') as jfile:
        data = json.load(jfile)
    headers = data[0].keys() if isinstance(data, list) else data.keys()
    with open(csv_file, 'w', newline='') as cfile:
        writer = csv.DictWriter(cfile, fieldnames=headers)
        writer.writeheader()
        if isinstance(data, list):
            writer.writerows(data)
        else:
            writer.writerow(data)
    print(f"JSON file '{json_file}' converted to CSV '{csv_file}'.")

# 9. Convert CSV to JSON
def csv_to_json(csv_file, json_file):
    with open(csv_file, 'r') as cfile:
        reader = csv.DictReader(cfile)
        data = [row for row in reader]
    with open(json_file, 'w') as jfile:
        json.dump(data, jfile, indent=4)
    print(f"CSV file '{csv_file}' converted to JSON '{json_file}'.")

# 10. Copy a File
def copy_file(src, dest):
    shutil.copy(src, dest)
    print(f"File '{src}' copied to '{dest}'.")

# 11. Move a File
def move_file(src, dest):
    shutil.move(src, dest)
    print(f"File '{src}' moved to '{dest}'.")

# 12. Delete a File
def delete_file(file_name):
    os.remove(file_name)
    print(f"File '{file_name}' deleted.")

# 13. Read a Text File
def read_text_file(file_name):
    with open(file_name, 'r') as file:
        content = file.read()
    print(f"Contents of '{file_name}':\n{content}")

# 14. Convert Markdown to HTML
def markdown_to_html(md_file, html_file):
    with open(md_file, 'r') as file:
        html_content = markdown.markdown(file.read())
    with open(html_file, 'w') as file:
        file.write(html_content)
    print(f"Markdown file '{md_file}' converted to HTML '{html_file}'.")

# 15. Convert HTML to PDF (requires wkhtmltopdf installed)
def html_to_pdf(html_file, pdf_file):

    pdfkit.from_file(html_file, pdf_file)
    print(f"HTML file '{html_file}' converted to PDF '{pdf_file}'.")

# Example usage
if __name__ == "__main__":
    # Create a text file
    create_text_file("example.txt", "This is an example text file.")

    # Create a PDF from text
    create_pdf("example.pdf", "This is an example PDF with multiple lines.\nSecond paragraph.")

    # Convert an image to PDF
    image_to_pdf("example.jpg", "example_image.pdf")

    # Convert markdown to PDF
    create_text_file("example.md", "# Hello World\nThis is a markdown file.")
    markdown_to_pdf("example.md", "example_markdown.pdf")

    # Create a URL shortcut
    create_url_shortcut("NLTK_Chapter8.url", "https://www.nltk.org/book/ch08.html")

    # Create a CSV file
    csv_data = [["Name", "Age", "City"], ["Alice", 30, "New York"], ["Bob", 25, "Los Angeles"]]
    create_csv_file("example.csv", csv_data[1:], headers=csv_data[0])

    # Create a JSON file
    json_data = {"Name": "Alice", "Age": 30, "City": "New York", "Interests": ["reading", "coding", "hiking"]}
    create_json_file("example.json", json_data)

    # Convert JSON to CSV
    json_to_csv("example.json", "converted_example.csv")

    # Convert CSV to JSON
    csv_to_json("example.csv", "converted_example.json")

    # Copy a file
    copy_file("example.txt", "example_copy.txt")

    # Move a file
    move_file("example_copy.txt", "moved_example.txt")

    # Delete a file
    delete_file("moved_example.txt")

    # Read a text file
    read_text_file("example.txt")

    # Convert Markdown to HTML
    markdown_to_html("example.md", "example.html")

    # Convert HTML to PDF (requires wkhtmltopdf installed)
    html_to_pdf("example.html", "example_html.pdf")
