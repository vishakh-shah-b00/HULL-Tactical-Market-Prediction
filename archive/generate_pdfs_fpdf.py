import markdown
from fpdf import FPDF
import os
import re

# Configuration
ARTIFACT_DIR = '/Users/sanchaybhutani/.gemini/antigravity/brain/0b2a736f-a41a-47ad-87b5-41277f07fdcc'
REPORTS = [
    {
        'md_file': os.path.join(ARTIFACT_DIR, 'complete_analysis_report.md'),
        'pdf_file': 'Complete_Analysis_Report.pdf',
        'title': 'Hull Tactical - Complete Analysis Report'
    },
    {
        'md_file': os.path.join(ARTIFACT_DIR, 'model_development_report.md'),
        'pdf_file': 'Model_Development_Report.pdf',
        'title': 'Hull Tactical - Model Development Report'
    }
]

class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        self.cell(0, 10, self.title_text, new_x="LMARGIN", new_y="NEXT", align='C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def clean_text_for_pdf(text):
    # 1. Replace specific symbols
    replacements = {
        '✅': '[OK]',
        '❌': '[X]',
        '🚨': '[ALERT]',
        '🎯': '[TARGET]',
        '⚠️': '[WARNING]',
        '→': '->',
        '≈': '~',
        '≥': '>=',
        '×': 'x',
        '—': '-',
        '–': '-',
        '“': '"',
        '”': '"',
        '‘': "'",
        '’': "'",
        '…': '...',
        '↑': '^',
        '↓': 'v'
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
        
    # 2. Encode to latin-1 to strip any other unsupported chars
    # This prevents the UnicodeEncodeError
    return text.encode('latin-1', 'replace').decode('latin-1')

def generate_pdf(report_config):
    print(f"Processing {report_config['md_file']}...")
    
    # Read markdown content
    with open(report_config['md_file'], 'r') as f:
        md_content = f.read()
    
    # Clean text
    cleaned_md = clean_text_for_pdf(md_content)
    
    # Convert to HTML
    html_content = markdown.markdown(
        cleaned_md,
        extensions=['tables', 'fenced_code']
    )
    
    # Create PDF
    pdf = PDF()
    pdf.title_text = report_config['title']
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    
    # Try to write HTML
    try:
        # fpdf2 write_html
        pdf.write_html(html_content)
        output_path = report_config['pdf_file']
        pdf.output(output_path)
        print(f"✅ Generated {output_path}")
    except Exception as e:
        print(f"❌ Error rendering HTML for {report_config['pdf_file']}: {str(e)}")
        
        # Fallback: Write plain text (cleaned)
        pdf = PDF()
        pdf.title_text = report_config['title']
        pdf.add_page()
        pdf.set_font("Courier", size=10)
        
        # Split lines to avoid huge blocks
        for line in cleaned_md.split('\n'):
            pdf.multi_cell(0, 5, line)
            
        output_path = report_config['pdf_file']
        pdf.output(output_path)
        print(f"⚠️ Generated fallback text PDF for {report_config['pdf_file']}")

if __name__ == "__main__":
    for report in REPORTS:
        generate_pdf(report)
