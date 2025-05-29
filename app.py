import os
import torch
import open_clip
import pytesseract
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_file
from pdf2image import convert_from_bytes
import google.generativeai as genai
from torchvision import transforms
from dotenv import load_dotenv
import tempfile
import uuid
import pdfkit
from werkzeug.utils import secure_filename
import datetime
import time
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# ============================
# Load API Key and Configure
# ============================
GOOGLE_API_KEY = "AIzaSyD7JPJ67ZVmMwIwrgYz_G1iYoKhJ1ZHZqY"
genai.configure(api_key=GOOGLE_API_KEY)
model_gemini = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Rate limiting configuration
RATE_LIMIT_DELAY = 2  # seconds between API calls
last_api_call_time = 0

def summarize_text(text, max_sentences=10):
    """Summarize text by extracting key sentences."""
    try:
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        # If text is short enough, return as is
        if len(sentences) <= max_sentences:
            return text
            
        # Remove stopwords and punctuation
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text.lower())
        filtered_text = [word for word in word_tokens if word not in stop_words and word not in string.punctuation]
        
        # Score sentences based on word frequency
        word_frequencies = {}
        for word in filtered_text:
            if word not in word_frequencies:
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
                
        # Normalize word frequencies
        max_frequency = max(word_frequencies.values())
        for word in word_frequencies:
            word_frequencies[word] = word_frequencies[word] / max_frequency
            
        # Score sentences
        sentence_scores = {}
        for sentence in sentences:
            for word in word_tokenize(sentence.lower()):
                if word in word_frequencies:
                    if sentence not in sentence_scores:
                        sentence_scores[sentence] = word_frequencies[word]
                    else:
                        sentence_scores[sentence] += word_frequencies[word]
                        
        # Get top sentences
        summary_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:max_sentences]
        summary_sentences = [s[0] for s in sorted(summary_sentences, key=lambda x: sentences.index(x[0]))]
        
        return ' '.join(summary_sentences)
    except Exception as e:
        print(f"Error in summarization: {str(e)}")
        return text

def ask_gemini(text, image_mode=False):
    """Ask Gemini with rate limiting and text summarization."""
    global last_api_call_time
    
    # Add delay between API calls
    current_time = time.time()
    time_since_last_call = current_time - last_api_call_time
    if time_since_last_call < RATE_LIMIT_DELAY:
        time.sleep(RATE_LIMIT_DELAY - time_since_last_call)
    
    # Summarize text if it's too long
    if len(text.split()) > 1000:  # If more than 1000 words
        text = summarize_text(text)
        print("Text was summarized before sending to Gemini")
    
    prompt = f'''
You are a professional medical report analyst and claims assistant working for an Indian health insurance company. Your task is to assist in reviewing and summarizing medical reports and scans to help assess claim eligibility, risk, and underwriting clarity.

The following is a {'combined analysis of a medical report and scan' if 'Report Analysis:' in text else 'text extracted from a scanned medical report'}:
\"\"\"{text}\"\"\"

Please provide your response in a clear, professional format that can assist the insurance team and claims officer. Follow these formatting rules strictly:
1. Use only plain text - no markdown, no asterisks, no emojis, no special characters
2. Use clear section headers in CAPITAL LETTERS
3. Use bullet points (dashes) for lists
4. Keep paragraphs concise and well-spaced
5. Use proper indentation for better readability

Organize your response under these sections:

PATIENT SUMMARY
- Summarize the medical condition(s) in 5 clear, simple lines
- Highlight the primary diagnosis and any comorbidities
- Include the age, gender, and hospital if available

MEDICAL CONDITION ANALYSIS
- Identify and briefly explain the medical condition(s)
- List common symptoms and probable causes
- Mention if it appears acute, chronic, critical, or elective in nature

RECOMMENDED CLINICAL ACTIONS
- Mention diagnostic tests, medications, surgery, or rest as advised
- Include estimated recovery time and risks if possible

INSURANCE CLAIM ASSESSMENT
- Type of treatment: OPD / Inpatient / Surgery / Diagnostic / Emergency
- Evidence of hospitalization or ongoing treatment
- Does this fall under pre-existing conditions? If yes, mention duration
- Does this relate to any lifestyle-induced or hereditary/family-induced disease
- Are there any ICD-10 codes or known insurance exclusions inferred

RISK AND RED FLAG CHECK
- Do the symptoms, treatment, and doctor notes logically align
- Any signs of inconsistency or suspicious content
- Check if documents could be forged or reused from older claims

HOSPITAL & DOCUMENT METADATA
- Name and address of the hospital (if mentioned)
- Date of report, doctor name, and signature presence (if available)
- Does the document appear scanned from original or digitally generated

{'SCAN FINDINGS COMPARISON' if 'Report Analysis:' in text else ''}
- {'Compare the scan findings with the report content' if 'Scan Analysis:' in text else 'Note: Scan details are mentioned in the report but no scan image was uploaded. Please verify the scan details with the original scan images.'}
- Verify if the scan type matches what's mentioned in the report
- Check for any discrepancies between scan results and report findings
- Assess if the scan supports the diagnosis mentioned in the report
- Evaluate if additional scans or tests are needed based on the report
- Determine if the scan and report together provide sufficient evidence for the claim

Focus on clarity, professional tone, and practical insurance insights. Ensure each section is clearly separated and properly formatted.
    '''
    
    try:
        last_api_call_time = time.time()
        response = model_gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {str(e)}")
        return f"Error: Unable to process the request. Please try again in a few minutes. Error details: {str(e)}"

# ============================
# BiomedCLIP Initialization
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_clip, _, preprocess = open_clip.create_model_and_transforms(
    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
)
tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
model_clip.eval()
model_clip = model_clip.to(device)

# ============================
# Flask App
# ============================
app = Flask(__name__)

# ============================
# Helper Functions
# ============================
def analyze_image_with_biomedclip(img_pil, modality="CT scan"):
    # List of possible medical scan types to check against
    medical_prompts = [
        "chest x-ray",
        "brain CT scan",
        "abdominal CT scan",
        "lung CT scan",
        "spine CT scan",
        "pelvis CT scan",
        "cardiac CT scan",
        "COVID-19 CT scan",
        "tumor in CT scan",
        "brain tumor",
        "brain hemorrhage",
        "stroke in brain",
        "pneumonia in chest x-ray",
        "lung cancer",
        "tuberculosis",
        "fracture in spine",
        "kidney stone in CT scan",
        "liver lesion",
        "aortic aneurysm",
        "pleural effusion",
        "fluid in lungs",
        "blood clot in brain",
        "cyst in brain",
        "multiple sclerosis",
        "abdominal infection"
    ]

    # Prepare image input
    image_input = preprocess(img_pil).unsqueeze(0).to(device)
    
    # Get the best matching scan type
    best_match = None
    best_score = -1
    
    for prompt in medical_prompts:
        text_input = tokenizer([prompt]).to(device)
        
        with torch.no_grad():
            image_features = model_clip.encode_image(image_input)
            text_features = model_clip.encode_text(text_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).squeeze().item()
            
            if similarity > best_score:
                best_score = similarity
                best_match = prompt

    return f"The uploaded scan is most likely a **{best_match.upper()}** with a confidence score of **{best_score:.2f}**."

# Set Poppler path explicitly
POPPLER_PATH = r"C:\Program Files\poppler-24.08.0\Library\bin"

def extract_text_from_file(file):
    all_text = ""
    images = []
    try:
        if file.filename.endswith(".pdf"):
            # Save the file temporarily
            temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            file.save(temp_pdf.name)
            temp_pdf.close()
            
            # Convert PDF to images using explicit Poppler path
            images = convert_from_bytes(
                open(temp_pdf.name, 'rb').read(),
                poppler_path=POPPLER_PATH
            )
            
            # Clean up temporary file
            os.unlink(temp_pdf.name)
        else:
            img = Image.open(file)
            images = [img]

        for img in images:
            text = pytesseract.image_to_string(img)
            all_text += text + "\n"

        return all_text.strip(), images
    except Exception as e:
        print(f"Error in extract_text_from_file: {str(e)}")
        return "", []

# ============================
# PDF Generation Helper
# ============================
app.config['WKHTMLTOPDF_PATH'] = "C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe"

def generate_pdf_from_html(html_content):
    pdfkit_config = pdfkit.configuration(wkhtmltopdf=app.config['WKHTMLTOPDF_PATH'])
    options = {
        'page-size': 'A4',
        'margin-top': '25mm',
        'margin-right': '25mm',
        'margin-bottom': '25mm',
        'margin-left': '25mm',
        'encoding': 'UTF-8',
        'no-outline': None,
        'enable-local-file-access': None
    }
    pdf = pdfkit.from_string(html_content, False, configuration=pdfkit_config, options=options)
    return pdf

# ============================
# Routes
# ============================
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files and 'scan' not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    result_id = str(uuid.uuid4())[:8]
    gemini_input = ""
    biomed_summary = ""
    report_text = ""
    scan_text = ""

    try:
        # Handle PDF/Report file
        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            if file.filename.endswith(".pdf"):
                report_text, _ = extract_text_from_file(file)
                gemini_input = report_text
            else:
                return jsonify({"error": "Report must be in PDF format"}), 400

        # Handle Scan file
        if 'scan' in request.files and request.files['scan'].filename:
            scan = request.files['scan']
            if scan.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = Image.open(scan)
                biomed_summary = analyze_image_with_biomedclip(img)
                scan_text = biomed_summary
            else:
                return jsonify({"error": "Scan must be in image format (PNG, JPG, JPEG)"}), 400

        # If both files are uploaded, combine the analysis
        if report_text and scan_text:
            gemini_input = f"""
            Report Analysis:
            {report_text}

            Scan Analysis:
            {scan_text}

            Please analyze both the report and scan to verify if they match and provide a comprehensive assessment. 
            Pay special attention to:
            1. Whether the scan type matches what's mentioned in the report
            2. If the scan findings support the diagnosis in the report
            3. Any discrepancies between the scan and report
            4. If additional scans or tests are needed
            5. Whether the evidence is sufficient for insurance claim processing
            """
        elif report_text:
            # Check if report contains scan-related information
            scan_keywords = ['scan', 'x-ray', 'ct', 'mri', 'ultrasound', 'radiology', 'imaging']
            has_scan_info = any(keyword in report_text.lower() for keyword in scan_keywords)
            
            if has_scan_info:
                gemini_input = f"""
                Report Analysis:
                {report_text}

                Note: This report contains scan/imaging details, but no scan image was uploaded. 
                Please verify the scan details with the original scan images.
                """
            else:
                gemini_input = report_text

        # Generate Gemini analysis
        if gemini_input:
            gemini_response = ask_gemini(gemini_input, image_mode=bool(scan_text))
        else:
            gemini_response = "No report or scan provided for analysis."

        return render_template('report_template.html',
                             report_id=result_id,
                             biomedclip_result=biomed_summary,
                             gemini_summary=gemini_response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate_report', methods=['POST'])
def generate_report():
    medical_report = request.form.get("medical_report")
    
    if not medical_report:
        return "No report content provided", 400

    try:
        # Clean up the report content and add proper spacing
        cleaned_report = medical_report.replace('🏥', 'PATIENT SUMMARY')\
            .replace('📋', 'MEDICAL CONDITION ANALYSIS')\
            .replace('🩺', 'RECOMMENDED CLINICAL ACTIONS')\
            .replace('💼', 'INSURANCE CLAIM ASSESSMENT')\
            .replace('🚩', 'RISK AND RED FLAG CHECK')\
            .replace('📍', 'HOSPITAL & DOCUMENT METADATA')\
            .replace('**', '')\
            .replace('---', '')\
            .replace('*', '-')

        # Split the content into sections and format each section
        sections = cleaned_report.split('\n\n')
        formatted_sections = []
        
        for section in sections:
            if section.strip():
                # Split into lines and format each line
                lines = section.split('\n')
                formatted_lines = []
                for line in lines:
                    if line.strip():
                        if line.strip().startswith('-'):
                            formatted_lines.append(f'<li>{line.strip()[1:].strip()}</li>')
                        else:
                            formatted_lines.append(f'<h2 class="section-title">{line.strip()}</h2>')
                formatted_sections.append('\n'.join(formatted_lines))

        formatted_content = '\n'.join(formatted_sections)

        # Create a temporary HTML file with the report content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Medical Report Analysis</title>
            <style>
                @page {{
                    margin: 2.5cm;
                    @top-center {{
                        content: "Medical Report Analysis";
                        font-size: 9pt;
                        color: #666;
                    }}
                    @bottom-center {{
                        content: "Page " counter(page) " of " counter(pages);
                        font-size: 9pt;
                        color: #666;
                    }}
                }}
                body {{ 
                    font-family: Arial, sans-serif; 
                    line-height: 1.6; 
                    color: #2c3e50;
                    font-size: 11pt;
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 40px;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 20px;
                }}
                .header h1 {{
                    color: #2c3e50;
                    margin: 0;
                    font-size: 24pt;
                }}
                .header p {{
                    color: #666;
                    margin: 10px 0 0 0;
                    font-size: 12pt;
                }}
                .section {{
                    margin-bottom: 40px;
                    page-break-inside: avoid;
                }}
                .section-title {{
                    color: #2c3e50;
                    font-size: 14pt;
                    font-weight: bold;
                    margin: 30px 0 20px 0;
                    padding-bottom: 5px;
                    border-bottom: 1px solid #eee;
                }}
                ul {{
                    margin: 15px 0;
                    padding-left: 20px;
                    list-style-type: none;
                }}
                li {{
                    margin-bottom: 12px;
                    line-height: 1.6;
                    position: relative;
                    padding-left: 15px;
                }}
                li:before {{
                    content: "•";
                    position: absolute;
                    left: 0;
                    color: #3498db;
                }}
                .footer {{
                    margin-top: 40px;
                    padding-top: 20px;
                    border-top: 1px solid #eee;
                    font-size: 9pt;
                    color: #666;
                    text-align: center;
                }}
                .disclaimer {{
                    font-style: italic;
                    color: #666;
                    margin-top: 10px;
                }}
                .watermark {{
                    position: fixed;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%) rotate(-45deg);
                    font-size: 72pt;
                    color: rgba(52, 152, 219, 0.1);
                    z-index: -1;
                    white-space: nowrap;
                }}
            </style>
        </head>
        <body>
            <div class="watermark">CONFIDENTIAL</div>
            <div class="header">
                <h1>Medical Report Analysis</h1>
                <p>Professional Medical Claims Assessment</p>
            </div>
            
            <div class="content">
                {formatted_content}
            </div>
            
            <div class="footer">
                <p>Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p class="disclaimer">This is an AI-generated analysis and should be reviewed by a medical professional.</p>
            </div>
        </body>
        </html>
        """
        
        # Generate PDF
        pdf = generate_pdf_from_html(html_content)
        
        # Save PDF to a temporary file with a simple name
        pdf_filename = f"medical_report_{int(time.time())}.pdf"
        with open(pdf_filename, "wb") as f:
            f.write(pdf)
        
        # Send the file and then delete it
        response = send_file(
            pdf_filename,
            as_attachment=True,
            download_name=pdf_filename,
            mimetype='application/pdf'
        )
        
        @response.call_on_close
        def delete_file():
            try:
                os.remove(pdf_filename)
            except:
                pass
        
        return response
    except Exception as e:
        return str(e), 500

# ============================
# Run the App
# ============================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
