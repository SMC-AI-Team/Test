import os
import shutil

import fitz  # PyMuPDF


# =========================================================
# 경로 설정
# =========================================================
BASE = "/Volumes/Expansion/[이식외과]/SMC_Annotation/"

#(6:8)48case
PDF_ROOT = os.path.join(BASE, "Annotation_pdf/(6:8)48case")
PNG_ROOT = os.path.join(BASE, "Annotation_png/(6:8)48case")

os.makedirs(PNG_ROOT, exist_ok=True)


# =========================================================
# 표시(window) 설정
# =========================================================
PDF_DPI = 600
OVERWRITE_PNG = True


# =========================================================
# 1단계: PDF -> Annotation_png
# =========================================================
def pdf_to_images(pdf_path, output_dir, dpi=300):
    pdf = fitz.open(pdf_path)
    os.makedirs(output_dir, exist_ok=True)

    for page_index in range(1, len(pdf)):
        page = pdf[page_index]
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_path = os.path.join(output_dir, f"slide_{page_index}.png")
        pix.save(out_path)

    pdf.close()



def rename_slide_images(folder_path, pdf_filename):
    base_name = os.path.splitext(pdf_filename)[0]

    slide_images = sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith(".png")],
        key=lambda x: int(os.path.splitext(x)[0].split("_")[-1])
    )

    for idx, filename in enumerate(slide_images, start=1):
        ext = os.path.splitext(filename)[1]
        new_name = f"{base_name}_{idx}{ext}"

        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, new_name)
        if src != dst:
            os.rename(src, dst)



def process_pdf_file(pdf_path, output_root, dpi=300, overwrite=True):
    pdf_filename = os.path.basename(pdf_path)
    base_name = os.path.splitext(pdf_filename)[0]
    output_folder = os.path.join(output_root, base_name)

    if overwrite and os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    os.makedirs(output_folder, exist_ok=True)

    pdf_to_images(pdf_path, output_folder, dpi=dpi)
    rename_slide_images(output_folder, pdf_filename)

    print(f"[DONE PDF->PNG] {pdf_filename} -> {output_folder}")



def step1_pdf_to_png():
    print("\n================ STEP 1: PDF -> PNG ================")
    for file in os.listdir(PDF_ROOT):
        if file.startswith("."):
            continue
        if file.lower().endswith(".pdf"):
            full_path = os.path.abspath(os.path.join(PDF_ROOT, file))
            process_pdf_file(full_path, PNG_ROOT, dpi=PDF_DPI, overwrite=OVERWRITE_PNG)



def main():
    step1_pdf_to_png()
    print("\n================ STEP 1 DONE ================")


if __name__ == "__main__":
    main()
