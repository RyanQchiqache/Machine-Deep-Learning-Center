import os
from fpdf import FPDF
import csv
import json

def create_text_file(file_name, content):
    with open(file_name, 'w') as file:
        file.write(content)
    print(f"Text file '{file_name}' created.")

def create_pdf(file_name, content):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, content)
    pdf.output(file_name)
    print(f"PDF file '{file_name}' created.")

def create_url_shortcut(file_name, url):
    with open(file_name, 'w') as file:
        file.write(f"[InternetShortcut]\nURL={url}\n")
    print(f"URL shortcut '{file_name}' created.")

def create_csv_file(file_name, data, headers=None):
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        if headers:
            writer.writerow(headers)
        writer.writerows(data)
    print(f"CSV file '{file_name}' created.")

def create_json_file(file_name, data):
    with open(file_name, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"JSON file '{file_name}' created.")

# Example usage:
if __name__ == "__main__":
    # Create a text file
    create_text_file("example.txt", "This is an example text file.")

    # Create a PDF
    create_pdf("example.pdf", "This is an example PDF with multiple lines.\n\nSecond paragraph.")

    # Create a URL shortcut
    create_url_shortcut("NLTK_Chapter8.url", "https://www.nltk.org/book/ch08.html")

    # Create a CSV file
    csv_data = [
        ["Name", "Age", "City"],
        ["Alice", 30, "New York"],
        ["Bob", 25, "Los Angeles"]
    ]
    create_csv_file("example.csv", csv_data[1:], headers=csv_data[0])

    # Create a JSON file
    json_data = {
        "Name": "Alice",
        "Age": 30,
        "City": "New York",
        "Interests": ["reading", "coding", "hiking"]
    }
    create_json_file("example.json", json_data)
