import os
import cv2
import fitz
from tqdm import tqdm
import numpy as np
import pdfplumber
from PIL import Image
from paddleocr import PaddleOCR, draw_ocr, PPStructure
import logging
from paddleocr import PaddleOCR
import json

# Get PaddleOCR logger
logger = logging.getLogger('ppocr')
# Set log level to WARNING or higher
logger.setLevel(logging.WARNING)

class PDFProcessor:
    def __init__(self, folder_path, save_path, font_path):
        self.folder_path = folder_path
        self.save_path = save_path
        self.font_path = font_path
        # Initialize OCR engine with GPU support
        self.ocr_engine = PaddleOCR(use_gpu=True, use_angle_cls=True, lang="en")
        self.table_engine = PPStructure(use_gpu=True, show_log=True, return_ocr_result_in_table=True, lang='en', structure_version='PP-StructureV2')

    def process_folder(self, search_keywords):
        all_tables = {}
        processed_count = 0  # Counter for processed files
        files = [f for f in os.listdir(self.folder_path) if f.endswith('.pdf')]  # Get all PDF files

        # Process files with progress bar
        for filename in tqdm(files, desc="Processing files"):
            pdf_path = os.path.join(self.folder_path, filename)
            try:
                all_tables[filename] = self.process_pdf(pdf_path, search_keywords)
                processed_count += 1

                # Save progress every 10 files
                if processed_count % 10 == 0:
                    with open(self.save_path + "all_pdf_table_texts_partial.json", "w") as f:
                        json.dump(all_tables, f)
                    print(f"Saved progress after processing {processed_count} files.")

            except Exception as e:
                print(f"Error processing file {filename}: {e}")

        # Save all processed files after the loop
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

# Example usage with generic paths
folder_path = "./input_pdfs/"
save_path = './output/'
font_path = './fonts/font.ttf'

processor = PDFProcessor(folder_path, save_path, font_path)
search_keywords = [
    'Characteristics', 'characteristics', 
    'Demographic', 'demographic',
    'Sociodemographic', 'sociodemographic',
    'TABLE', 'Table', 'table'
]
all_table_texts = processor.process_folder(search_keywords)

# Save results to JSON
with open(save_path + "all_pdf_table_texts.json", "w") as f:
    json.dump(all_table_texts, f)
