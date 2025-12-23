#!/usr/bin/env python3
"""
Generate HTML reports with all visualizations embedded
Then convert to PDF using weasyprint
"""
import markdown
import os

def create_html_from_markdown(md_file, html_file, title):
    """Convert markdown to standalone HTML"""
    
    # Read markdown
    with open(md_file, 'r') as f:
        md_content = f.read()
    
    # Convert to HTML
    html_content = markdown.markdown(
        md_content,
        extensions=['tables', 'fenced_code', 'codehilite']
    )
    
    # Create full HTML with styling
    full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 40px auto;
            padding: 0 20px;
            color: #333;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-top: 40px;
        }}
        h2 {{
            color: #34495e;
            border-bottom: 2px solid #bdc3c7;
            padding-bottom: 8px;
            margin-top: 30px;
        }}
        h3 {{
            color: #7f8c8d;
            margin-top: 25px;
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-radius: 8px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px 15px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Monaco', 'Courier New', monospace;
            font-size: 0.9em;
        }}
        pre {{
            background-color: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        pre code {{
            background-color: transparent;
            color: #ecf0f1;
            padding: 0;
        }}
        blockquote {{
            border-left: 4px solid #3498db;
            padding-left: 20px;
            color: #7f8c8d;
            margin: 20px 0;
        }}
        hr {{
            border: none;
            border-top: 2px solid #ecf0f1;
            margin: 30px 0;
        }}
        .page-break {{
            page-break-after: always;
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>
"""
    
    # Write HTML
    with open(html_file, 'w') as f:
        f.write(full_html)
    
    print(f"✓ Created {html_file}")

if __name__ == "__main__":
    print("Generating HTML reports...")
    
    # Create HTML versions
    create_html_from_markdown(
        'COMPLETE_ANALYSIS_REPORT.md',
        'Complete_Data_Analysis_Report.html',
        'Hull Tactical - Complete Data Analysis'
    )
    
    create_html_from_markdown(
        'MODEL_DEVELOPMENT_REPORT.md',
        'Model_Development_Report.html',
        'Hull Tactical - Model Development'
    )
    
    print("\n✅ HTML reports generated!")
    print("   - Complete_Data_Analysis_Report.html")
    print("   - Model_Development_Report.html")
    print("\n💡 Open in browser → Print → Save as PDF")
    
    # Try to open in browser
    import webbrowser
    webbrowser.open('file://' + os.path.abspath('Complete_Data_Analysis_Report.html'))
    webbrowser.open('file://' + os.path.abspath('Model_Development_Report.html'))
