from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import io
from PyPDF2 import PdfReader
import google.generativeai as genai

app = FastAPI()

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to allow specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

def get_gemini_response(input_text, pdf_content, prompt):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content([input_text, pdf_content[0], prompt])
    return response.text

def extract_text_from_pdf(uploaded_file):
    pdf_text = ""
    pdf_reader = PdfReader(uploaded_file)
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()
    return pdf_text

@app.post("/analyze-resume")
async def analyze_resume(job_description: str, resume_pdf: UploadFile = File(...)):
    pdf_bytes = await resume_pdf.read()
    pdf_text = extract_text_from_pdf(io.BytesIO(pdf_bytes))
    
    input_prompt = """
    You are an experienced Technical Human Resource Manager, your task is to review the provided resume against the job description. 
    Please share your professional evaluation on whether the candidate's profile aligns with the role. 
    Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
    """
    
    response = get_gemini_response(input_prompt, [pdf_text], job_description)
    return {"response": response}

@app.post("/evaluate-resume")
async def evaluate_resume(job_description: str, resume_pdf: UploadFile = File(...)):
    pdf_bytes = await resume_pdf.read()
    resume_text = extract_text_from_pdf(io.BytesIO(pdf_bytes))
    
    input_prompt = """
    You are an skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality, 
    your task is to evaluate the resume against the provided job description. give me the percentage of match if the resume matches
    the job description. First the output should come as marks on the scale of 10 and then keywords missing and last final thoughts.
    """
    
    response = get_gemini_response(input_prompt, [resume_text], job_description)
    return {"response": response}
