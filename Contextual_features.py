import os
import json
import logging
import numpy as np
import cv2
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
from paddleocr import PaddleOCR, draw_ocr, PPStructure
from tqdm import tqdm

# Configure PaddleOCR logger
logger = logging.getLogger('ppocr')
logger.setLevel(logging.WARNING)

class PDFProcessor:
    def __init__(self, folder_path, save_path, font_path):
        self.folder_path = folder_path
        self.save_path = save_path
        self.font_path = font_path
        self.ocr_engine = PaddleOCR(use_gpu=True, use_angle_cls=True, lang="en")
        self.table_engine = PPStructure(use_gpu=True, show_log=True, return_ocr_result_in_table=True, lang='en', structure_version='PP-StructureV2')

    def process_folder(self, search_keywords):
        all_tables = {}
        processed_count = 0  # Counter to track processed files
        files = [f for f in os.listdir(self.folder_path) if f.endswith('.pdf')]  # Get all PDF files

        # Create progress bar with tqdm
        for filename in tqdm(files, desc="Processing files"):
            pdf_path = os.path.join(self.folder_path, filename)
            try:
                all_tables[filename] = self.process_pdf(pdf_path, search_keywords)
                processed_count += 1  # Increment counter after processing a file

                # Save progress every 10 files
                if processed_count % 10 == 0:
                    with open(self.save_path + "all_pdf_table_texts_partial.json", "w") as f:
                        json.dump(all_tables, f)
                    print(f"Saved progress after processing {processed_count} files.")

            except Exception as e:
                print(f"Error processing file {filename}: {e}")  # Print error message

        # Save all processed files at the end
        with open(self.save_path + "all_pdf_table_texts_final.json", "w") as f:
            json.dump(all_tables, f)
        print("Saved all processed files.")

        return all_tables

    def process_pdf(self, pdf_path, search_keywords):
        imgs = self._convert_pdf_to_images(pdf_path)
        ocr_result = self.ocr_engine.ocr(pdf_path, cls=True)
        pdf_pages_to_save = self._find_relevant_pages(ocr_result, imgs, search_keywords, pdf_path)
        self._save_relevant_pages(imgs, pdf_pages_to_save, pdf_path)
        table_index = self._detect_tables(pdf_pages_to_save, pdf_path)
        new_table_coords = self._calculate_new_coordinates(table_index, pdf_path)
        table_text = self._extract_table_text(new_table_coords, pdf_path)
        return table_text if table_text else {}

    def _convert_pdf_to_images(self, pdf_path):
        imgs = []
        with fitz.open(pdf_path) as pdf:
            for pg in range(pdf.page_count):
                page = pdf[pg]
                mat = fitz.Matrix(2, 2)
                pm = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                imgs.append(img)
        return imgs

    def _find_relevant_pages(self, ocr_result, imgs, search_keywords, pdf_path):
        pdf_pages_to_save = []
        for idx, page_result in enumerate(ocr_result):
            image = imgs[idx]
            for line in page_result:
                text = line[1][0]
                if any(keyword.lower() in text.lower() for keyword in search_keywords):
                    print(text)
                    pdf_pages_to_save.append(idx)
                    self._save_ocr_results(image, idx, page_result, pdf_path)
                    break
        return list(set(pdf_pages_to_save))

    def _save_ocr_results(self, image, idx, page_result, pdf_path):
        boxes = [line[0] for line in page_result]
        txts = [line[1][0] for line in page_result]
        scores = [line[1][1] for line in page_result]
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        im_show = draw_ocr(image, boxes, txts, scores, font_path=self.font_path)
        im_show = Image.fromarray(im_show)
        im_show.save(f'{self.save_path}/{base_name}_ocr_page_{idx}.jpg')

    def _save_relevant_pages(self, imgs, pdf_pages_to_save, pdf_path):
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        with fitz.open(pdf_path) as pdf:
            for pg in pdf_pages_to_save:
                page = pdf[pg]
                mat = fitz.Matrix(4, 4)
                pm = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
                img.save(f'{self.save_path}/{base_name}_saved_page_{pg}.png')

    def _detect_tables(self, pdf_pages_to_save, pdf_path):
        table_index = {}
        for j in pdf_pages_to_save:
            img_path = f'{self.save_path}/{os.path.splitext(os.path.basename(pdf_path))[0]}_saved_page_{j}.png'
            img = cv2.imread(img_path)
            result = self.table_engine(img)
            final_table = [res['bbox'] for res in result if res['type'] == 'table']
            if final_table:
                table_index[str(j)] = final_table
        return table_index

    def _calculate_new_coordinates(self, table_index, pdf_path):
        new_table_coords = {}
        pdf = pdfplumber.open(pdf_path)
        for i, tables in table_index.items():
            old_image_path = f'{self.save_path}/{os.path.splitext(os.path.basename(pdf_path))[0]}_saved_page_{i}.png'
            old_img = Image.open(old_image_path)
            old_size = old_img.size
            im = pdf.pages[int(i)].to_image()
            new_image_path = f'{self.save_path}/im.jpg'
            im.save(new_image_path)
            new_img = Image.open(new_image_path)
            new_size = new_img.size
            table_coords = [self._calculate_coordinates(j, old_size, new_size) for j in tables]
            new_table_coords[str(i)] = table_coords
        return new_table_coords

    def _calculate_coordinates(self, old_coords, old_size, new_size):
        old_width, old_height = old_size
        new_width, new_height = new_size
        x1, y1, x2, y2 = old_coords
        new_x1 = x1 * new_width / old_width
        new_y1 = y1 * new_height / old_height
        new_x2 = x2 * new_width / old_width
        new_y2 = y2 * new_height / old_height
        return (new_x1, new_y1, new_x2, new_y2)

    def _extract_table_text(self, new_table_coords, pdf_path):
        table_text = {}
        pdf = pdfplumber.open(pdf_path)
        for i, tables in new_table_coords.items():
            texts = []
            for j in tables:
                cropped_table = pdf.pages[int(i)].crop(j)
                text = cropped_table.extract_text()
                cropped_table.to_image()
                if text:
                    texts.append(text.strip())
            if texts:
                table_text[str(i)] = texts
        return table_text

# Paths to folder, save location, and font file
folder_path = 'your/folder/path/'  # Set your folder path here
save_path = 'your/save/path/'  # Set your save path here
font_path = 'your/font/path/simfang.ttf'  # Set your font path here (e.g., simfang.ttf)

# Create a PDF processor instance and process the folder
processor = PDFProcessor(folder_path, save_path, font_path)
all_table_texts = processor.process_folder(search_keywords=['Characteristics', 'characteristics', 'Characteristic', 'characteristic', 'demographic', 'Demographic', 'demographics', 'Demographics', 'sociodemographic', 'Sociodemographic', 'sociodemographics', 'Sociodemographics'])

# Save all processed table texts to a JSON file
with open(save_path + "all_pdf_table_texts3.json", "w") as f:
    json.dump(all_table_texts, f)

# Function to query OpenAI GPT in JSON mode (API key and base URL should be set in your environment)
def query_openai_gpt_json_mode(prompt, text):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",  # Specify the model
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": prompt + text}
        ],
        response_format={"type": "json_object"}  # Enable JSON mode
    )
    return response.choices[0].message.content

prompt = """
I have accessed a table from a medical research study, and I need to confirm its content before proceeding with data extraction. Could you first verify if this table describes the demographics or characteristics of the study participants? If it does, I would then like you to extract and answer the following queries based on the table. However, if the table does not describe baseline characteristics, please respond with the statement: 'This table does not detail the baseline characteristics of a study population.' If any information is not available in the table, please respond with 'unknown'. In case the table details multiple study groups, kindly address each group separately.
I need the following details, please with units included where applicable:
1. Total sample size of the participants.
2. Age range or average age of the participants.
3. Gender distribution among the participants.
4. Racial or ethnic distribution of the participants.
5. Educational background of the participants.
6. Employment status of the participants.
Please present your findings in a JSON format, with the following keys: 'Sample Size', 'Age', 'Gender', 'Racial Distribution', 'Education', and 'Employment'. This will ensure clarity and consistency in understanding the demographic composition of the study participants. Here is the text from the table:
"""

# Use tqdm to create a progress bar
answer = {}  # Initialize as a dictionary
for pmid in tqdm(list(filtered_table_text.keys()), "Chatgpt answer"):
    answer[pmid] = []
    pmid_key = list(filtered_table_text[pmid].keys())  # Get all pages for a specific PMID
    for page in pmid_key:
        text = table_text[pmid][page][0]  # Assume the first element is the text
        response_str = query_openai_gpt_json_mode(prompt, text)  # Get response from the model

        try:
            # Try to parse the response string as a dictionary
            response_json = json.loads(response_str)
        except json.JSONDecodeError:
            # If parsing fails, skip this response
            print("Failed to decode JSON from response:", response_str)
            continue  # Skip the rest of the current loop

        # Check if response_json meets the conditions to be added
        if response_json and len(response_json) > 1:  # Ensure response_json contains more than one key-value pair
            all_unknown = all(value == "unknown" for value in response_json.values())  # Check if all values are "unknown"
            contains_digit = any(re.search(r'\d', str(value)) for value in response_json.values())  # Check if any value contains a digit
            if not all_unknown and contains_digit:  # If not all values are "unknown" and at least one value contains a digit
                print(response_json)
                filtered_table_text[pmid][page].append(response_json)
                contextual_features = {"source": "table", "location": str(page), "info": response_json}
                answer[pmid].append(contextual_features)

df = pd.read_csv(target_pmid_file_path)

# Remove rows where 'PMCID' column is empty
df_clean = df.dropna(subset=['PMCID'])

# Show the number of remaining rows
remaining_rows = df_clean.shape[0]
print(f"Remaining rows: {remaining_rows}")

"""1. Download XML files"""

from Bio import Entrez
import time
import pandas as pd

# Set your email
Entrez.email = "your_email@example.com"  # Replace with your email

# Path to the CSV file
target_pmid_file_path = 'your/csv/file/path.csv'  # Set your CSV file path here
df = pd.read_csv(target_pmid_file_path)

# Remove rows where 'PMCID' column is empty
df_clean = df.dropna(subset=['PMCID'])

# Get the list of PMCIDs from the CSV file
pmcids = df_clean['PMCID'].tolist()  # Assume PMCID is in a column named 'PMCID'

# Ensure the download directory exists
download_dir = "your/download/dir/path/"  # Set your download directory path here
os.makedirs(download_dir, exist_ok=True)

# Loop through each PMCID and download the full-text XML
for pmcid in tqdm(pmcids, "Downloading PMC XML"):
    try:
        # Use efetch to retrieve the full-text XML
        handle = Entrez.efetch(db="pmc", id=pmcid, rettype="xml")
        xml_data = handle.read()
        handle.close()

        # Write the XML content to a file named PMCID.xml
        filename = os.path.join(download_dir, f"{pmcid}.xml")
        with open(filename, "w", encoding='utf-8') as xml_file:
            xml_file.write(xml_data.decode('utf-8'))

        print(f"Downloaded {pmcid}")

    except Exception as e:
        print(f"Error downloading {pmcid}: {e}")

    # Pause for 2 seconds between requests to avoid overwhelming the server
    time.sleep(2)

"""2. Extract tables from XML files"""

import xml.etree.ElementTree as ET

# Path to the directory containing the XML files
xml_directory_path = "your/xml/directory/path/"  # Set your XML directory path here

# Initialize a list to store all table data
tables_list = []

# Get all XML files in the directory
xml_files = [f for f in os.listdir(xml_directory_path) if f.endswith('.xml')]

# Use tqdm to create a progress bar
for filename in tqdm(xml_files, desc='Extracting tables from XML files'):
    file_path = os.path.join(xml_directory_path, filename)

    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Initialize a dictionary to store table data for the current XML file
    tables_dict = {}
    tables_count = 0  # Initialize table counter

    # Find all tables
    for table in root.findall('.//table-wrap'):
        tables_count += 1  # Update table count
        table_title = table.find('.//title')
        if table_title is not None:
            table_title_text = table_title.text
        else:
            table_title_text = "No Title"

        # Initialize a list to store all rows of the current table
        table_content = []

        # Get table rows
        for row in table.findall('.//tr'):
            row_content = []
            # Get cells
            for cell in row.findall('.//td'):
                if cell.text:
                    row_content.append(cell.text.strip())
                else:
                    row_content.append('')
            table_content.append(row_content)

        # Store table title and content in the dictionary
        tables_dict[table_title_text] = table_content

    # Add the current file's table data to the total list
    tables_list.append({filename: tables_dict})

    # Print the number of tables extracted from the current file
    print(f'{filename}: Extracted {tables_count} tables')

# Finally, print the total number of tables extracted
total_tables = sum(len(item[next(iter(item))]) for item in tables_list)
print(f'Total tables extracted: {total_tables}')

# Replace with your OpenAI API client setup
client = OpenAI(
    base_url="your_openai_api_base_url",  # Replace with your OpenAI API base URL
    api_key="your_openai_api_key",  # Replace with your OpenAI API key
)

def query_openai_gpt_json_mode(prompt, table_text):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",  # Specify the model
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": prompt + table_text}
        ],
        response_format={"type": "json_object"}  # Enable JSON mode
    )
    return response.choices[0].message.content

prompt = """
I have accessed a html format table from a medical research study, and I need to confirm its content before proceeding with data extraction. Could you first verify if this table describes the demographics or characteristics of the study participants? If it does, I would then like you to extract and answers to the following queries based on the table. However, if the table does not describe baseline characteristics, please respond with the statement: 'This table does not detail the baseline characteristics of a study population.'If any information is not available in the table, please respond with 'unknown'. In case the table details multiple study groups, kindly address each group separately.
I need the following details, please with units included where applicable:
1. Total sample size of the participants.
2. Age range or average age of the participants.
3. Gender distribution among the participants.
4. Racial or ethnic distribution of the participants.
5. Educational background of the participants.
6. Employment status of the participants.
Please present your findings in a JSON format, with the following keys: 'Sample Size', 'Age', 'Gender', 'Racial Distribution', 'Education', and 'Employment'. This will ensure clarity and consistency in understanding the demographic composition of the study participants. Here is the text from the table:
"""

# Use tqdm to create a progress bar
answer = {}  # Initialize as a dictionary
for table in tqdm(xml_table, "ChatGPT answer"):
    for xml, content in table.items():
        # Check if the current XML key exists in the answer dictionary, if not, skip the current loop iteration
        if xml in answer:
            continue  # Skip to the next iteration

        answer[xml] = []
        for table_1 in content.values():
            text = table_1
            response_str = query_openai_gpt_json_mode(prompt, text)  # Get response from the model
            try:
                # Try to parse the response string as a dictionary
                response_json = json.loads(response_str)
            except json.JSONDecodeError:
                # If parsing fails, print error message and skip this response
                print("Failed to decode JSON from response:", response_str)
                continue  # Skip the rest of the current loop

            if response_json and len(response_json) > 1:  # Ensure response_json contains more than one key-value pair
                all_unknown = all(value == "unknown" for value in response_json.values())  # Check if all values are "unknown"
                contains_digit = any(re.search(r'\d', str(value)) for value in response_json.values())  # Check if any value contains a digit
                if not all_unknown and contains_digit:  # If not all values are "unknown" and at least one value contains a digit
                    print(response_json)
                    contextual_features = {"source": "table", "location": "xml_unknown", "info": response_json}
                    answer[xml].append(contextual_features)
