import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# Professional Corporate Palette (Corporate Blue Theme)
PRIMARY_COLOR = colors.HexColor("#0F2C59")     # Deep corporate navy blue
SECONDARY_COLOR = colors.HexColor("#1D4ED8")   # Bright corporate blue
TEXT_COLOR = colors.HexColor("#1F2937")        # Dark slate grey for readable text
BG_LIGHT = colors.HexColor("#F9FAFB")          # Very light grey for alternating rows
BORDER_COLOR = colors.HexColor("#E5E7EB")      # Soft border grey
HIGHLIGHT_COLOR = colors.HexColor("#3B82F6")   # Accent blue

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas renderer to handle running headers and dynamic footers 
    (e.g., 'Page X of Y') professionally across pages.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_elements(num_pages)
            super().showPage()
        super().save()

    def draw_page_elements(self, page_count):
        self.saveState()
        
        # Draw header and footer on all pages EXCEPT the title cover page (Page 1)
        if self._pageNumber > 1:
            # Header
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(SECONDARY_COLOR)
            self.drawString(54, 750, "PROJECT TECHNICAL REPORT: NON-CONTACT FACIAL STRESS DETECTION")
            
            # Header rule
            self.setStrokeColor(BORDER_COLOR)
            self.setLineWidth(0.75)
            self.line(54, 742, 558, 742)
            
            # Footer rule
            self.line(54, 52, 558, 52)
            
            # Footer text
            self.setFont("Helvetica", 8)
            self.setFillColor(TEXT_COLOR)
            page_str = f"Page {self._pageNumber} of {page_count}"
            self.drawRightString(558, 40, page_str)
            
        self.restoreState()

def create_report_pdf(output_filename="Project_Report.pdf"):
    # 0.75-inch side margins (54pt), 1.0-inch top/bottom margins (72pt)
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=72,
        bottomMargin=72
    )
    
    styles = getSampleStyleSheet()
    
    # Redesigned styles for a clean, highly professional corporate layout
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=PRIMARY_COLOR,
        alignment=TA_LEFT,
        spaceAfter=10
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=13,
        leading=17,
        textColor=SECONDARY_COLOR,
        alignment=TA_LEFT,
        spaceAfter=30
    )
    
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=PRIMARY_COLOR,
        spaceBefore=18,
        spaceAfter=10,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=SECONDARY_COLOR,
        spaceBefore=12,
        spaceAfter=8,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11, # Corporate body font size 11 pt
        leading=15,  # Appropriate line spacing (leading)
        textColor=TEXT_COLOR,
        spaceAfter=8,
        alignment=TA_LEFT # Left-aligned paragraphs for clean corporate presentation
    )
    
    body_bold = ParagraphStyle(
        'BodyTextBold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )

    bullet_style = ParagraphStyle(
        'BulletText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=TEXT_COLOR,
        leftIndent=24,
        firstLineIndent=-12,
        spaceAfter=6
    )
    
    callout_style = ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=TEXT_COLOR,
        spaceBefore=5,
        spaceAfter=5
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12,
        textColor=colors.white
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=TEXT_COLOR
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=table_cell_style,
        fontName='Helvetica-Bold'
    )

    story = []
    
    # ==========================================
    #             1. COVER PAGE
    # ==========================================
    story.append(Spacer(1, 40))
    
    # Category tag / Organization prefix
    story.append(Paragraph("TECHNICAL RESEARCH & PROJECT MONOGRAPH", ParagraphStyle(
        'CoverCategory',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=12,
        textColor=SECONDARY_COLOR,
        spaceAfter=15
    )))
    
    # Thick corporate visual separator
    accent_bar_data = [['']]
    accent_bar_table = Table(accent_bar_data, colWidths=[504], rowHeights=[4])
    accent_bar_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), PRIMARY_COLOR),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(accent_bar_table)
    story.append(Spacer(1, 20))
    
    # Document Title & Subtitle
    story.append(Paragraph("AI-BASED NON-CONTACT FACIAL<br/>STRESS DETECTION SYSTEM", title_style))
    story.append(Paragraph("Real-Time Multi-Modal Physiology and Affective Computing on CPU", subtitle_style))
    
    story.append(Spacer(1, 100))
    
    # Metadata Block Table
    meta_headers_style = ParagraphStyle('MetaHeader', parent=table_cell_bold, fontSize=9.5, leading=12, textColor=PRIMARY_COLOR)
    meta_vals_style = ParagraphStyle('MetaVal', parent=table_cell_style, fontSize=9.5, leading=12)
    
    meta_data = [
        [Paragraph("PROJECT TITLE", meta_headers_style), Paragraph("AI-Based Non-Contact Facial Stress Detection System", meta_vals_style)],
        [Paragraph("NAME", meta_headers_style), Paragraph("Vaira Selvi", meta_vals_style)],
        [Paragraph("DOMAIN AREA", meta_headers_style), Paragraph("Artificial Intelligence, Computer Vision, Affective Computing", meta_vals_style)],
        [Paragraph("APPLIED IMPACT", meta_headers_style), Paragraph("Occupational Stress Management, Human Factors Research", meta_vals_style)],
        [Paragraph("DEPLOYMENT FORMAT", meta_headers_style), Paragraph("Desktop Application (Windows / Linux / macOS)", meta_vals_style)],
        [Paragraph("SYSTEM REQUIREMENT", meta_headers_style), Paragraph("Standard RGB Webcam, CPU-only (No GPU Required)", meta_vals_style)],
        [Paragraph("DOCUMENT VERSION", meta_headers_style), Paragraph("Version 1.0.0 (Production Release)", meta_vals_style)],
        [Paragraph("RELEASE DATE", meta_headers_style), Paragraph("July 2026", meta_vals_style)]
    ]
    t_meta = Table(meta_data, colWidths=[140, 364])
    t_meta.setStyle(TableStyle([
        ('LINEBELOW', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 0), # No padding on left to align with document border
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_meta)
    
    story.append(PageBreak())
    
    # ==========================================
    #          2. EXECUTIVE SUMMARY & INTRO
    # ==========================================
    story.append(Paragraph("1. Executive Summary", h1_style))
    story.append(Paragraph(
        "This project monograph details the design, implementation, and evaluation of an <b>AI-Based Non-Contact Facial Stress Detection System</b> "
        "engineered to execute in real time on standard consumer hardware. The system establishes a novel, non-invasive bio-behavioral pipeline "
        "combining three key modalities: <b>remote photoplethysmography (rPPG)</b> to extract cardiac signals, <b>computer vision algorithms</b> "
        "to track facial movement patterns, and <b>machine learning</b> to evaluate facial emotional indicators.",
        body_style
    ))
    story.append(Paragraph(
        "By integrating physiological metrics (Heart Rate and Heart Rate Variability) with behavioral indexes (blink frequency, yawn occurrences, "
        "and head rotation pose), the system compiles a unified 16-dimensional descriptor vector. A trained <b>Random Forest Regressor</b> determines "
        "an objective, continuous Stress Score ranging from 0 to 100. In addition, the system incorporates SHAP (SHapley Additive exPlanations) "
        "values to provide a local explanation for every score generated, detailing exactly how each facial parameter influences the final stress assessment.",
        body_style
    ))
    
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("2. Introduction & Background", h1_style))
    story.append(Paragraph(
        "Chronic workplace stress constitutes a primary contributor to reduced productivity, mental exhaustion, and long-term cardiovascular diseases. "
        "Historically, objective stress measurement has relied on contact-based devices, such as electrocardiogram (ECG) chest straps and "
        "galvanic skin response (GSR) sensors. Although highly accurate, these devices are physically intrusive, costly, and difficult to deploy "
        "continuously in typical office environments.",
        body_style
    ))
    story.append(Paragraph(
        "Recent developments in computer vision and signal processing have enabled <b>non-contact physiological estimation</b>. Blood volume changes "
        "in the facial capillary bed cause minuscule variations in skin light absorption. By tracking facial region-of-interest (ROI) averages across "
        "subsequent video frames, these color shifts can be isolated to reconstruct the cardiac pulse waveform. This work presents an end-to-end "
        "desktop system that performs real-time rPPG extraction, facial action analysis, and classification on a single CPU thread, providing "
        "businesses and researchers with an administrative stress monitoring solution.",
        body_style
    ))
    
    story.append(PageBreak())
    
    # ==========================================
    #      3. PROBLEM STATEMENT & OBJECTIVES
    # ==========================================
    story.append(Paragraph("3. Problem Statement", h1_style))
    story.append(Paragraph(
        "Developing a reliable stress monitoring platform requires solving the challenge of non-invasive, real-time data collection. Specifically, "
        "the system must extract clear physiological signals from standard webcams under variable lighting and motion conditions without relying on "
        "external GPU servers. Additionally, because stress is a complex physical and mental state, relying on a single modality (such as expression alone) "
        "is prone to high error rates. This project addresses the challenge of <i>multi-modal sensor fusion</i> and provides an interpretable model "
        "to ensure that stress predictions can be audited and understood by technical and medical reviewers.",
        body_style
    ))
    
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("4. Project Objectives", h1_style))
    objectives = [
        "<b>Real-Time Face Tracking:</b> Establish robust face tracking operating at a minimum of 30 frames per second.",
        "<b>Sub-Region Segmentation:</b> Extract 468 facial mesh landmarks to delineate regions of interest (cheeks, forehead, eyes, and mouth).",
        "<b>Physiological Extraction:</b> Reconstruct the cardiac pulse waveform to estimate Heart Rate (HR) and Heart Rate Variability (HRV) contactlessly.",
        "<b>Behavioral Analytics:</b> Track blink frequency, yawn rate, and head pose orientation metrics dynamically.",
        "<b>Psychological Modeling:</b> Evaluate emotional configurations across seven standardized facial expression distributions.",
        "<b>Multi-Modal Fusion:</b> Compile physiological, behavioral, and emotional variables into a unified feature vector.",
        "<b>Explainable Classifications:</b> Utilize SHAP parameters to explain the quantitative factors driving individual predictions.",
        "<b>Interactive Visualization:</b> Present live outputs through a modern, responsive CustomTkinter GUI dashboard.",
        "<b>Automated Reporting:</b> Generate structured PDF medical-style summaries and log metrics locally to an SQLite database."
    ]
    for obj in objectives:
        story.append(Paragraph(f"• {obj}", bullet_style))
        
    story.append(PageBreak())

    # ==========================================
    #            5. SYSTEM METHODOLOGY
    # ==========================================
    story.append(Paragraph("5. Technical Methodology", h1_style))
    
    story.append(Paragraph("5.1 Face Detection & Tracking", h2_style))
    story.append(Paragraph(
        "The system processes incoming webcam streams at 640×480 resolution. Face detection is performed using MediaPipe's BlazeFace model. "
        "To track individuals across frames, a custom Centroid Tracker is implemented. The tracker computes the centroid coordinates of detected "
        "bounding boxes and associates them across frames using minimum Euclidean distance. This lightweight tracking module prevents identity "
        "switching on a single CPU thread.",
        body_style
    ))
    
    story.append(Paragraph("5.2 Facial Landmark Mesh & Sub-Region Segmentation", h2_style))
    story.append(Paragraph(
        "Upon face detection, the system applies MediaPipe Face Mesh to extract 468 3D landmark points. Specific landmark indices are grouped "
        "to define localized facial regions. A convex hull algorithm generates binary masks of the forehead and left/right cheek zones. These "
        "regions are isolated because they exhibit high vascularity and minimal muscular movement, making them optimal for cardiovascular pulse tracking.",
        body_style
    ))
    
    story.append(Paragraph("5.3 Remote Photoplethysmography (rPPG)", h2_style))
    story.append(Paragraph(
        "The system extracts heart metrics using the Plane-Orthogonal-to-Skin (POS) method. The mean RGB color values are computed across "
        "the cheek and forehead regions. The color signals are normalized and projected onto orthogonal axes to isolate blood volume variations "
        "from lighting changes. A 3rd-order Butterworth bandpass filter limits frequencies to the human heart rate range (0.75 to 2.5 Hz, "
        "equivalent to 45 to 150 BPM). The heart rate is estimated by computing the Fast Fourier Transform (FFT) and identifying the dominant "
        "spectral peak. Heart Rate Variability (HRV) is calculated as the standard deviation of successive peak-to-peak intervals (SDNN) in milliseconds.",
        body_style
    ))
    
    story.append(Paragraph("5.4 Behavioral & Emotional Features", h2_style))
    story.append(Paragraph(
        "Behavioral characteristics are tracked continuously. The Eye Aspect Ratio (EAR) estimates blink occurrences (triggered when EAR "
        "falls below 0.20 for two or more frames), and the Mouth Aspect Ratio (MAR) monitors yawning (triggered when MAR exceeds 0.50 for "
        "fifteen frames). Head pose is calculated by solving the Perspective-n-Point (solvePnP) problem, mapping 2D landmarks to a standard "
        "3D model to output Pitch, Yaw, and Roll angles. In parallel, spatial configurations of key mouth and eye landmarks are mapped and "
        "passed to a Random Forest Classifier to estimate the probability distributions of seven distinct emotions.",
        body_style
    ))

    story.append(Paragraph("5.5 Feature Fusion & Regression", h2_style))
    story.append(Paragraph(
        "A 16-dimensional vector is constructed, normalising the physiological, behavioral, and emotional variables. This fused vector is "
        "processed by a Random Forest Regressor to yield a continuous Stress Score (0 to 100). The predictions are explained using local "
        "SHAP values, displaying the contribution of each feature in the final report.",
        body_style
    ))

    story.append(PageBreak())

    # ==========================================
    #            6. SYSTEM SPECS & STACK
    # ==========================================
    story.append(Paragraph("6. Technology Stack & Core Components", h1_style))
    story.append(Paragraph(
        "The system is built on a highly optimized, cross-platform Python architecture. The table below outlines the specific libraries, "
        "versions, and core technical roles in the platform's execution:",
        body_style
    ))
    story.append(Spacer(1, 5))

    tech_headers = [Paragraph("Category", table_header_style), Paragraph("Component", table_header_style), Paragraph("Role & Implementation", table_header_style)]
    tech_rows = [
        [Paragraph("Development", table_cell_bold), Paragraph("Python 3.10+", table_cell_style), Paragraph("Core runtime and system script integration.", table_cell_style)],
        [Paragraph("Face Detection", table_cell_bold), Paragraph("MediaPipe Face Detection", table_cell_style), Paragraph("SSD-based BlazeFace network for face tracking.", table_cell_style)],
        [Paragraph("Facial Mesh", table_cell_bold), Paragraph("MediaPipe Face Mesh", table_cell_style), Paragraph("Generates 468-point 3D spatial coordinate models.", table_cell_style)],
        [Paragraph("Computer Vision", table_cell_bold), Paragraph("OpenCV (cv2)", table_cell_style), Paragraph("Webcam stream control and solvePnP head-pose solvers.", table_cell_style)],
        [Paragraph("Signal Analytics", table_cell_bold), Paragraph("SciPy", table_cell_style), Paragraph("Butterworth filters, peak-finding, and spectral FFT functions.", table_cell_style)],
        [Paragraph("Machine Learning", table_cell_bold), Paragraph("scikit-learn", table_cell_style), Paragraph("Random Forest Regressors and classifiers (.pkl models).", table_cell_style)],
        [Paragraph("Explainable AI", table_cell_bold), Paragraph("SHAP", table_cell_style), Paragraph("Calculates Shapley values to provide feature-level transparency.", table_cell_style)],
        [Paragraph("Local Database", table_cell_bold), Paragraph("SQLite3 & Pandas", table_cell_style), Paragraph("Manages session log tables and parses SQL inputs.", table_cell_style)],
        [Paragraph("User Interface", table_cell_bold), Paragraph("CustomTkinter & Matplotlib", table_cell_style), Paragraph("Builds the responsive dark-themed dashboard.", table_cell_style)],
        [Paragraph("PDF Compilation", table_cell_bold), Paragraph("ReportLab", table_cell_style), Paragraph("Generates the project and session reports.", table_cell_style)]
    ]
    t_tech_table = Table([tech_headers] + tech_rows, colWidths=[100, 140, 264])
    t_tech_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY_COLOR),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, BG_LIGHT]),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
    ]))
    story.append(t_tech_table)
    
    story.append(PageBreak())

    # ==========================================
    #          7. METRIC EXPLANATIONS
    # ==========================================
    story.append(Paragraph("7. Stress Indicator Explanations & Reference Limits", h1_style))
    story.append(Paragraph(
        "The diagnostic metrics measured by the system are detailed in the table below, summarizing what is measured, "
        "how it is calculated, and its corresponding clinical significance:",
        body_style
    ))
    story.append(Spacer(1, 5))

    metric_headers = [Paragraph("Indicator", table_header_style), Paragraph("Measurement Method", table_header_style), Paragraph("Clinical Significance", table_header_style)]
    metric_rows = [
        [Paragraph("Heart Rate (BPM)", table_cell_bold), Paragraph("Calculated using the POS chrominance algorithm and FFT peak frequencies on capillary color variations.", table_cell_style), Paragraph("Elevated heart rates (>90 BPM) indicate sympathetic nervous arousal under immediate stress.", table_cell_style)],
        [Paragraph("HRV SDNN (ms)", table_cell_bold), Paragraph("Standard deviation of successive peak-to-beat intervals across a 10-second window.", table_cell_style), Paragraph("High values (>100 ms) reflect strong parasympathetic recovery. Low values (<50 ms) signal acute distress.", table_cell_style)],
        [Paragraph("Blinks", table_cell_bold), Paragraph("Cumulative blinks tracked via the Eye Aspect Ratio (EAR) mathematical formula.", table_cell_style), Paragraph("Elevated blink counts point to mental fatigue, cognitive overload, or visual exhaustion.", table_cell_style)],
        [Paragraph("Yawns", table_cell_bold), Paragraph("Cumulative yawns tracked via the Mouth Aspect Ratio (MAR) vertical-to-horizontal distance ratios.", table_cell_style), Paragraph("Yawning indicates physical sleepiness, fatigue, or stress-related exhaustion.", table_cell_style)],
        [Paragraph("Dominant Emotion", table_cell_bold), Paragraph("Predicted using landmark distance ratios and a Random Forest Classifier.", table_cell_style), Paragraph("Frequent negative emotions (Angry, Fear, Sad) contribute to higher stress scores.", table_cell_style)],
        [Paragraph("Stress Score", table_cell_bold), Paragraph("Fused 16-D vector output evaluated by a Random Forest Regressor.", table_cell_style), Paragraph("Low: 0-32 (Relaxed baseline)<br/>Moderate: 33-65 (Focused/Active)<br/>High: 66-100 (Anxious/Distressed)", table_cell_style)]
    ]
    t_metric_table = Table([metric_headers] + metric_rows, colWidths=[110, 194, 200])
    t_metric_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY_COLOR),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, BG_LIGHT]),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
    ]))
    story.append(t_metric_table)
    
    story.append(PageBreak())

    # ==========================================
    #        8. SCREENSHOT REVIEW (EMBEDDED)
    # ==========================================
    story.append(Paragraph("8. Live System Dashboard Review", h1_style))
    story.append(Paragraph(
        "Below is the system screenshot captured during a live monitoring session, displaying the user interface, "
        "real-time parameters, and signal output:",
        body_style
    ))
    
    # Embed and center the dashboard screenshot
    img_path = "dashboard_screenshot.png"
    if os.path.exists(img_path):
        try:
            img_reader = ImageReader(img_path)
            img_w, img_h = img_reader.getSize()
            aspect = img_h / img_w
            target_w = 400
            target_h = target_w * aspect
            
            # Place image inside a table to ensure centered alignment and spacing
            t_img = Table([[Image(img_path, width=target_w, height=target_h)]], colWidths=[500])
            t_img.setStyle(TableStyle([
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('BOTTOMPADDING', (0,0), (-1,-1), 10),
                ('TOPPADDING', (0,0), (-1,-1), 5),
            ]))
            story.append(t_img)
        except Exception as e:
            story.append(Paragraph(f"<i>Error loading dashboard screenshot: {e}</i>", body_style))
            story.append(Spacer(1, 10))
    else:
        story.append(Paragraph("<i>Dashboard screenshot (dashboard_screenshot.png) not found in workspace root.</i>", body_style))
        story.append(Spacer(1, 10))
    
    # Review details of the screenshot in a structured callout
    analysis_text = (
        "<b>Review and Interpretation of Live Dashboard Parameters:</b><br/>"
        "• <b>Face Bounding Box & Landmarks</b>: The subject's face is successfully detected with tracking ID 10. "
        "The red overlay dots depict the real-time extraction of 468 landmark coordinates, which are used to define the capillarized "
        "regions of interest (forehead/cheeks) and facial aspect ratios.<br/>"
        "• <b>Contactless Cardiac Output</b>: The user shows a heart rate of <b>85.6 BPM</b>, which is well within normal resting boundaries. "
        "The estimated HRV of <b>172.9 ms</b> is high, indicating strong autonomic resilience and healthy parasympathetic activity.<br/>"
        "• <b>Behavioral Tracking</b>: Cumulative blinks are recorded at 6, and yawns at 0, indicating minimal fatigue.<br/>"
        "• <b>Affective Classification</b>: The emotional classifier identifies the dominant expression as <b>Happy</b>, suggesting a positive "
        "affective state that lowers the final stress evaluation weights.<br/>"
        "• <b>Stress Evaluation</b>: The system assigns a Stress Score of <b>46.0</b>, classifying the state as <b>Moderate</b> (highlighted in orange). "
        "This indicates a moderate physiological/cognitive load, which is expected during standard task focus or conversational interaction.<br/>"
        "• <b>rPPG Waveform</b>: The Matplotlib plot displays a periodic, filtered cardiac signal. The clean, regular peaks demonstrate high signal "
        "quality and reliable heart rate estimation."
    )
    
    t_analysis_callout = Table([[Paragraph(analysis_text, callout_style)]], colWidths=[500])
    t_analysis_callout.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F9FAF5")),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('BOX', (0,0), (-1,-1), 1, SECONDARY_COLOR),
    ]))
    story.append(t_analysis_callout)
    
    story.append(PageBreak())

    # ==========================================
    #       9. DESIGN JUSTIFICATION & DATA
    # ==========================================
    story.append(Paragraph("9. Design Justification & Architecture Performance", h1_style))
    story.append(Paragraph(
        "The system's modular architecture separates tasks to ensure reliable operation on standard CPUs. "
        "Below are the primary design choices and architectural justifications for this implementation:",
        body_style
    ))
    
    justifications_list = [
        "<b>CPU-Only Portability:</b> Utilizing MediaPipe Face Mesh allows the system to run on standard office computers without "
        "requiring dedicated GPUs, enabling scalable deployment across corporate environments.",
        "<b>POS rPPG Robustness:</b> The Plane-Orthogonal-to-Skin (POS) algorithm improves heart rate estimation compared to simple green-channel "
        "color tracking, maintaining performance under lighting variations and minor user movements.",
        "<b>Multi-Threaded UI:</b> Delegating image processing, signal analysis, and database logging to a dedicated worker thread "
        "ensures the custom GUI dashboard remains responsive and runs smoothly at 30 FPS.",
        "<b>Explainable Predictions:</b> Incorporating SHAP explainability allows developers and reviewers to audit and trace the exact "
        "contributions of the 16 physiological, behavioral, and emotional features for any stress prediction."
    ]
    for just in justifications_list:
        story.append(Paragraph(f"• {just}", bullet_style))

    story.append(Spacer(1, 10))

    # References section with spelling and encoding corrections (e.g. Cech, Soukupova)
    story.append(Paragraph("10. Bibliographic References", h1_style))
    refs = [
        "Wang, W., den Brinker, A. C., Stuijk, S. and de Haan, G. (2016). 'Algorithmic Principles of Remote PPG.' <i>IEEE Transactions on Biomedical Engineering</i>, 64(7), pp. 1479-1491.",
        "Google MediaPipe — Face Mesh Developer Solutions and Canonical Mesh Reference Guides (2024).",
        "Soukupova, T. and Cech, J. (2016). 'Real-Time Eye Blink Detection using Facial Landmarks.' <i>Proceedings of the Computer Vision Winter Workshop</i>, pp. 1-8.",
        "Lundberg, S. M. and Lee, S. I. (2017). 'A Unified Approach to Interpreting Model Predictions' (SHAP). <i>Advances in Neural Information Processing Systems</i>, 30, pp. 4765-4774.",
        "Schmidt, P., Reiss, A., Duerichen, R., Maritsch, M. and Van Laerhoven, K. (2018). 'Introducing WESAD: A Multimodal Dataset for Wearable Stress and Affect Detection.' <i>Proceedings of the 20th ACM International Conference on Multimodal Interaction</i>, pp. 400-408.",
        "Verkruysse, W., Svaasand, L. O. and Nelson, J. S. (2008). 'Remote plethysmographic imaging using ambient light.' <i>Optics Express</i>, 16(26), pp. 21434-21445.",
        "de Haan, G. and Jeanne, V. (2013). 'Robust Pulse Rate From Chrominance-Based rPPG.' <i>IEEE Transactions on Biomedical Engineering</i>, 60(10), pp. 2878-2886.",
        "Breiman, L. (2001). 'Random Forests.' <i>Machine Learning</i>, 45(1), pp. 5-32."
    ]
    for idx, ref_text in enumerate(refs, 1):
        story.append(Paragraph(f"[{idx}] {ref_text}", bullet_style))

    # Disclaimer Callout Box
    story.append(Spacer(1, 15))
    disclaimer = (
        "<b>Technical Disclaimer:</b> This stress detection system is developed for research and employee wellness tracking "
        "purposes only. It is not a certified medical device and should not be used as a replacement for clinical diagnosis. "
        "Estimated heart rate and stress score values are approximations and may fluctuate based on background illumination, "
        "subject movement, and facial lighting profiles."
    )
    t_disclaimer = Table([[Paragraph(disclaimer, ParagraphStyle('DiscStyle', parent=body_style, fontSize=8.5, leading=12, textColor=colors.HexColor("#4B5563")))]], colWidths=[500])
    t_disclaimer.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F3F4F6")),
        ('BOX', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(t_disclaimer)

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)

if __name__ == "__main__":
    create_report_pdf()
    print("Project Report PDF generated successfully.")
