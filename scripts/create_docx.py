"""
create_docx.py -- Generate a highly professional Microsoft Word Document (.docx)
for the Week 1 deliverable, containing embedded tables and images.
"""
import os
import sys
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import CFG

def set_cell_background(cell, fill_hex):
    """Set cell background color."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Set internal cell margins (padding) in twentieths of a point (dxa)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def add_styled_table(doc, headers, data, column_widths=None):
    """Add a stylized table with header highlighting and light gray gridlines."""
    table = doc.add_table(rows=len(data) + 1, cols=len(headers))
    table.alignment = WD_ALIGN_PARAGRAPH.CENTER
    table.autofit = False

    # Header
    hdr_cells = table.rows[0].cells
    for i, header in enumerate(headers):
        hdr_cells[i].text = header
        set_cell_background(hdr_cells[i], "1A1A2E")
        set_cell_margins(hdr_cells[i], top=120, bottom=120, left=150, right=150)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        for run in p.runs:
            run.font.bold = True
            run.font.color.rgb = RGBColor(255, 255, 255)
            run.font.name = "Arial"
            run.font.size = Pt(9.5)

    # Data rows
    for row_idx, row_data in enumerate(data):
        row_cells = table.rows[row_idx + 1].cells
        bg_color = "F9F9F9" if row_idx % 2 == 1 else "FFFFFF"
        # Highlight total row
        if "Total" in str(row_data[0]):
            bg_color = "EAEAEA"
            
        for col_idx, text in enumerate(row_data):
            row_cells[col_idx].text = str(text)
            set_cell_background(row_cells[col_idx], bg_color)
            set_cell_margins(row_cells[col_idx], top=100, bottom=100, left=150, right=150)
            p = row_cells[col_idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            for run in p.runs:
                run.font.name = "Arial"
                run.font.size = Pt(9.5)
                if "Total" in str(row_data[0]):
                    run.font.bold = True

    # Adjust widths
    if column_widths:
        for row in table.rows:
            for idx, width in enumerate(column_widths):
                row.cells[idx].width = Inches(width)
                
    # Add a blank spacing paragraph after table
    p_space = doc.add_paragraph()
    p_space.paragraph_format.space_before = Pt(0)
    p_space.paragraph_format.space_after = Pt(6)

def main():
    doc = Document()
    
    # Configure document margins (1 inch / 2.54 cm all sides)
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    # Base style configurations
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(11.5)
    font.color.rgb = RGBColor(0, 0, 0)

    # Title Block
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    title_run = title.add_run("Cardiotox-Fusion: Phase 1 Usability Audit and Dataset Splits")
    title_run.font.bold = True
    title_run.font.size = Pt(16)
    title_run.font.name = "Times New Roman"
    title.paragraph_format.space_after = Pt(6)

    # Metadata Info
    meta = doc.add_paragraph()
    meta.paragraph_format.line_spacing = 1.3
    meta.paragraph_format.space_after = Pt(18)
    
    runs_data = [
        ("Milestone: ", True), ("Week 1 Deliverable   |   ", False),
        ("Date: ", True), ("July 27, 2026\n", False),
        ("Team Members: ", True), ("Jimit Patel, Vineela, Aditya, Nehal, Pranjal\n", False),
        ("Repository: ", True), ("github.com/JP-Bro/Cardiotox-Fusion-project", False)
    ]
    for text, is_bold in runs_data:
        r = meta.add_run(text)
        r.font.name = "Arial"
        r.font.size = Pt(10)
        r.font.bold = is_bold
        r.font.color.rgb = RGBColor(50, 50, 50)

    # 1. Dataset Usability Audit
    h1 = doc.add_paragraph()
    h1.paragraph_format.space_before = Pt(18)
    h1.paragraph_format.space_after = Pt(6)
    r = h1.add_run("1. Dataset Usability Audit")
    r.font.bold = True
    r.font.size = Pt(13)
    r.font.color.rgb = RGBColor(0, 0, 0)

    p1 = doc.add_paragraph(
        "To establish the modeling boundaries for the GNN-Transformer fusion model, "
        "we performed a joint audit of the FDA DICTrank labeling set and the LINCS L1000 GSE70138 "
        "level-5 expression matrix. Compounds were retained in the usable cohort only if they met "
        "the following criteria:"
    )
    p1.paragraph_format.space_after = Pt(6)

    bullets = [
        "Possessed a valid, parseable chemical structure (verified via RDKit).",
        "Matched a unique Level-5 perturbation signature (COMPZ) in the HA1E cell line (10.0 µM dose, 24-hour time point).",
        "Carried a non-ambiguous DICTrank label mapping to binary classification (No concern = 0; Less/Most concern = 1)."
    ]
    for b in bullets:
        bp = doc.add_paragraph(style='List Bullet')
        bp.paragraph_format.space_after = Pt(3)
        br = bp.add_run(b)
        br.font.size = Pt(11)

    p2 = doc.add_paragraph(
        "Out of the raw 1,211 labeled DICTrank compounds, a total of 562 compounds met all three requirements, "
        "forming our core dataset. The breakdown across the cardiotoxicity label categories is detailed below:"
    )
    p2.paragraph_format.space_before = Pt(6)
    p2.paragraph_format.space_after = Pt(10)

    # Table 1: Usability Audit Counts
    t1_headers = ["DICTrank Category", "Model Label", "Usable Count", "Percentage"]
    t1_data = [
        ["Most Concern (High Risk)", "1 (Positive)", "199", "35.4%"],
        ["Less Concern (Medium Risk)", "1 (Positive)", "246", "43.8%"],
        ["No Concern (Safe)", "0 (Negative)", "117", "20.8%"],
        ["Total Usable Set", "—", "562", "100.0%"]
    ]
    add_styled_table(doc, t1_headers, t1_data, column_widths=[2.8, 1.2, 1.2, 1.3])

    # Insert Image 1: Usable Cohort Plot
    img1_path = os.path.join(CFG.RESULTS_DIR, "phase1_usable_cohort.png")
    if os.path.isfile(img1_path):
        p_img1 = doc.add_paragraph()
        p_img1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img1.paragraph_format.space_after = Pt(18)
        p_img1.add_run().add_picture(img1_path, width=Inches(5.0))
        caption1 = doc.add_paragraph()
        caption1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption1.paragraph_format.space_after = Pt(12)
        cr1 = caption1.add_run("Figure 1: Usable Dataset Cohort distribution by FDA DICTrank concern class.")
        cr1.font.italic = True
        cr1.font.size = Pt(9.5)

    # 2. Leakage-Free Dataset Splits
    h2 = doc.add_paragraph()
    h2.paragraph_format.space_before = Pt(18)
    h2.paragraph_format.space_after = Pt(6)
    r2 = h2.add_run("2. Leakage-Free Dataset Splits")
    r2.font.bold = True
    r2.font.size = Pt(13)

    p3 = doc.add_paragraph(
        "To prevent target leakage and ensure realistic validation generalizability, "
        "we implemented two separate stratified splitting schemes. In both schemes, splits are frozen at a 70 / 15 / 15 ratio."
    )
    p3.paragraph_format.space_after = Pt(12)

    # Heading 2.1: Drug-Level Splits
    h2_1 = doc.add_paragraph()
    h2_1.paragraph_format.space_before = Pt(8)
    h2_1.paragraph_format.space_after = Pt(6)
    r2_1 = h2_1.add_run("A. Drug-Level Splits (Stratified)")
    r2_1.font.bold = True
    r2_1.font.italic = True
    r2_1.font.size = Pt(11.5)

    # Table 2: Drug Splits
    t2_headers = ["Partition", "Compound Count", "Positive (Tox) Count", "Negative (Safe) Count", "Positive Rate"]
    t2_data = [
        ["Train", "393", "311", "82", "79.1%"],
        ["Validation", "84", "67", "17", "79.8%"],
        ["Test", "85", "67", "18", "78.8%"],
        ["Total Set", "562", "445", "117", "79.2%"]
    ]
    add_styled_table(doc, t2_headers, t2_data, column_widths=[1.5, 1.3, 1.3, 1.3, 1.1])

    # Heading 2.2: Scaffold-Level Splits
    h2_2 = doc.add_paragraph()
    h2_2.paragraph_format.space_before = Pt(8)
    h2_2.paragraph_format.space_after = Pt(6)
    r2_2 = h2_2.add_run("B. Bemis-Murcko Scaffold Splits")
    r2_2.font.bold = True
    r2_2.font.italic = True
    r2_2.font.size = Pt(11.5)

    # Table 3: Scaffold Splits
    t3_data = [
        ["Train", "393", "312", "81", "79.4%"],
        ["Validation", "84", "63", "21", "75.0%"],
        ["Test", "85", "70", "15", "82.4%"],
        ["Total Set", "562", "445", "117", "79.2%"]
    ]
    add_styled_table(doc, t2_headers, t3_data, column_widths=[1.5, 1.3, 1.3, 1.3, 1.1])

    # Insert Image 2: Splits Distribution Plot
    img2_path = os.path.join(CFG.RESULTS_DIR, "phase1_splits_distribution.png")
    if os.path.isfile(img2_path):
        p_img2 = doc.add_paragraph()
        p_img2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img2.paragraph_format.space_after = Pt(18)
        p_img2.add_run().add_picture(img2_path, width=Inches(5.0))
        caption2 = doc.add_paragraph()
        caption2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption2.paragraph_format.space_after = Pt(12)
        cr2 = caption2.add_run("Figure 2: Target partition sizes and positive rates across drug-level and scaffold splits.")
        cr2.font.italic = True
        cr2.font.size = Pt(9.5)

    # 3. Verification & Reproducibility
    h3 = doc.add_paragraph()
    h3.paragraph_format.space_before = Pt(18)
    h3.paragraph_format.space_after = Pt(6)
    r3 = h3.add_run("3. Verification and Reproducibility")
    r3.font.bold = True
    r3.font.size = Pt(13)

    p4 = doc.add_paragraph(
        "All data processing steps are fully automated. The random seed is frozen at 42 in config.py, "
        "and both split files are frozen and saved in the project repository at data/splits/drug_split.csv "
        "and data/splits/scaffold_split.csv. "
        "These splits will remain frozen throughout the modeling pipeline."
    )
    p4.paragraph_format.space_after = Pt(12)

    # Save document
    out_docx_path = os.path.join(CFG.RESULTS_DIR, "Cardiotox_Fusion_Phase1_Report.docx")
    doc.save(out_docx_path)
    print(f"Word document saved successfully to: {out_docx_path}")

if __name__ == "__main__":
    main()
