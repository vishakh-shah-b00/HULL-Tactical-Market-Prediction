import markdown
from weasyprint import HTML, CSS
import os

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

# CSS for PDF styling
PDF_CSS = CSS(string='''
    @page {
        size: A4;
        margin: 2cm;
        @bottom-center {
            content: "Page " counter(page);
            font-family: sans-serif;
            font-size: 9pt;
            color: #7f8c8d;
        }
    }
    body {
        font-family: "Helvetica", "Arial", sans-serif;
        font-size: 10pt;
        line-height: 1.5;
        color: #2c3e50;
    }
    h1 {
        color: #2980b9;
        border-bottom: 2px solid #2980b9;
        padding-bottom: 10px;
        margin-top: 0;
    }
    h2 {
        color: #2c3e50;
        border-bottom: 1px solid #bdc3c7;
        padding-bottom: 5px;
        margin-top: 20px;
        page-break-after: avoid;
    }
    h3 {
        color: #34495e;
        margin-top: 15px;
        page-break-after: avoid;
    }
    img {
        max-width: 100%;
        height: auto;
        margin: 15px auto;
        display: block;
        border: 1px solid #bdc3c7;
        border-radius: 4px;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
        font-size: 9pt;
    }
    th, td {
        border: 1px solid #bdc3c7;
        padding: 8px;
        text-align: left;
    }
    th {
        background-color: #ecf0f1;
        font-weight: bold;
    }
    tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    code {
        font-family: "Courier New", monospace;
        background-color: #f4f6f7;
        padding: 2px 4px;
        border-radius: 3px;
        font-size: 0.9em;
    }
    pre {
        background-color: #2c3e50;
        color: #ecf0f1;
        padding: 10px;
        border-radius: 5px;
        overflow-x: auto;
        font-family: "Courier New", monospace;
        font-size: 0.85em;
        white-space: pre-wrap;
    }
    pre code {
        background-color: transparent;
        color: inherit;
        padding: 0;
    }
    blockquote {
        border-left: 4px solid #3498db;
        padding-left: 15px;
        color: #7f8c8d;
        margin: 15px 0;
        font-style: italic;
    }
    hr {
        border: 0;
        border-top: 1px solid #bdc3c7;
        margin: 20px 0;
    }
''')

def generate_pdf(report_config):
    print(f"Processing {report_config['md_file']}...")
    
    # Read markdown content
    with open(report_config['md_file'], 'r') as f:
        md_content = f.read()
    
    # Convert to HTML
    html_content = markdown.markdown(
        md_content,
        extensions=['tables', 'fenced_code', 'codehilite']
    )
    
    # Wrap in HTML structure
    full_html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{report_config['title']}</title>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    '''
    
    # Generate PDF
    output_path = report_config['pdf_file']
    HTML(string=full_html, base_url=ARTIFACT_DIR).write_pdf(
        output_path,
        stylesheets=[PDF_CSS]
    )
    print(f"✅ Generated {output_path}")

if __name__ == "__main__":
    for report in REPORTS:
        try:
            generate_pdf(report)
        except Exception as e:
            print(f"❌ Error generating {report['pdf_file']}: {str(e)}")
