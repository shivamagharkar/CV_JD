import streamlit as st
import os
from io import BytesIO
import pdfplumber
import json
import openai
from dotenv import load_dotenv

# --- Load Environment Variables ---
load_dotenv()

# --- Streamlit Session State Initialization ---
if 'openAI_API_KEY' not in st.session_state:
    st.session_state['openAI_API_KEY'] = os.getenv("openAI_API_KEY")
if 'cv_results' not in st.session_state:
    st.session_state['cv_results'] = None
if 'jd_results' not in st.session_state:
    st.session_state['jd_results'] = None

# --- Helper Functions ---
def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def build_cv_prompt(resume_text):
    json_schema = {
        "name": "", "email": "", "phone": "", "country": "", "city": "", "summary": "",
        "skills": [{"specialized skill": "", "common skill": ""}],
        "experience": [{"job_title": "", "company": "", "start_date": "", "end_date": "", "description": ""}],
        "education": [{"degree": "", "institution": "", "start_year": "", "end_year": ""}],
        "enrichment parameters": [{
            "Employment Pattern & Progression": "", "Company Type & Sector": "", "Education Quality & Ranking": "",
            "Skill Demand & Market Relevance": "", "Leadership Experience": "", "Budget & Project Management": "",
            "International Experience & Mobility": "", "Soft Skills from Sales Calls": "",
            "Personality & Behavioral Traits": [{
                "Openness": "", "Conscientiousness": "", "Extraversion": "", "Agreeableness": "", "Neuroticism": ""
            }],
            "Future Career Goals (Sales-Inferred)": "", "Salary Expectations (Sales-Inferred)": "",
            "JD Enrichment with Implied Preferences": "", "Cultural Fit Indicators": ""
        }]
    }
    prompt = f"""You are an expert resume parser. Convert the resume text below into this JSON format. 
Fill in all the relevant fields. Leave the enrichment_parameters field empty.
The JSON schema is as follows:
{json.dumps(json_schema, indent=2)}
Resume:
\"\"\"
{resume_text}
\"\"\"
"""
    return prompt

def build_jd_prompt(jd_text):
    json_schema = {
        "country": "", "city": "", "summary": "",
        "skills": [{"specialized skill": "", "common skill": ""}],
        "experience": [{"job_title": "", "company": "", "start_date": "", "end_date": "", "description": ""}],
        "education": [{"degree": "", "institution": "", "start_year": "", "end_year": ""}],
        "enrichment parameters": [{
            "Employment Pattern & Progression": "", "Company Type & Sector": "", "Education Quality & Ranking": "",
            "Skill Demand & Market Relevance": "", "Leadership Experience": "", "Budget & Project Management": "",
            "International Experience & Mobility": "", "Soft Skills from Sales Calls": "",
            "Future Career Goals (Sales-Inferred)": "", "Salary Expectations (Sales-Inferred)": "",
            "JD Enrichment with Implied Preferences": "", "Cultural Fit Indicators": ""
        }]
    }
    prompt = f"""You are an expert Job description parser. Convert the job description text below into this JSON format. 
Fill in all the relevant fields. Leave the enrichment parameters field empty.
Note that the json schema resembles a resume schema. 
This is because the end goal is to match the resume with the job description. 
However, keep in mind that the schema is to be filled with the job description data.
Again, the enrichment parameters field should be left empty.
The JSON schema is as follows:
{json.dumps(json_schema, indent=2)}
Job Description:
\"\"\"
{jd_text}
\"\"\"
"""
    return prompt

def call_openai(prompt):
    openai.api_key = st.session_state['openAI_API_KEY']
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

def call_openai_for_enrichment(prompt):
    openai.api_key = st.session_state['openAI_API_KEY']
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    try:
        enriched_data = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON: {e}")
        st.text("Response content:")
        st.text(response.choices[0].message.content)
        return None
    return enriched_data

def build_cv_enrichment_prompt(cv_data):
    prompt = f"""You are an expert in CV enrichment. Analyze the provided CV data and infer the following enrichment parameters:
... [SNIPPED for brevity, keep your original prompt here] ...
Here is the CV data:
{json.dumps(cv_data, indent=2)}
Please analyze and fill in the enrichment parameters. Return the enriched CV data in JSON format. Please respond ONLY with raw JSON.
"""
    return prompt

def build_jd_enrichment_prompt(jd_data):
    prompt = f"""You are an expert in Job Description enrichment. Analyze the provided Job description data and infer the following enrichment parameters:
... [SNIPPED for brevity, keep your original prompt here] ...
Here is the job description data:
{json.dumps(jd_data, indent=2)}
Please analyze and fill in the enrichment parameters. Return the enriched job description data in JSON format. Please respond ONLY with raw JSON.
"""
    return prompt

# --- Streamlit App Interface ---
st.title("CV/JD Analyzer")

# API Key Input
api_key_input = st.text_input("Enter your OpenAI API Key", type="password")
if api_key_input:
    st.session_state['openAI_API_KEY'] = api_key_input

# --- CV Processing ---
st.header("CV Processing")
cv_file = st.file_uploader("Upload CV (PDF)", type="pdf", key="cv")

if cv_file is not None and st.button("Process CV"):
    with st.spinner("Processing CV..."):
        try:
            bytes_data = cv_file.getvalue()
            with BytesIO(bytes_data) as pdf_file:
                cv_text = extract_text_from_pdf(pdf_file)

            cv_prompt = build_cv_prompt(cv_text)
            parsed_cv = call_openai(cv_prompt)
            parsed_cv_data = json.loads(parsed_cv)

            enrich_cv_prompt = build_cv_enrichment_prompt(parsed_cv_data)
            enriched_cv = call_openai_for_enrichment(enrich_cv_prompt)

            if enriched_cv:
                parsed_cv_data["enrichment parameters"] = enriched_cv.get("enrichment parameters", {})
                st.session_state['cv_results'] = parsed_cv_data
                st.success("CV processed successfully!")
        except Exception as e:
            st.error(f"An error occurred during CV processing: {e}")

    if st.session_state['cv_results']:
        st.subheader("Parsed and Enriched CV")
        st.json(st.session_state['cv_results'])

# --- JD Processing ---
st.header("JD Processing")
jd_file = st.file_uploader("Upload JD (PDF)", type="pdf", key="jd")

if jd_file is not None and st.button("Process JD"):
    with st.spinner("Processing JD..."):
        try:
            bytes_data = jd_file.getvalue()
            with BytesIO(bytes_data) as pdf_file:
                jd_text = extract_text_from_pdf(pdf_file)

            jd_prompt = build_jd_prompt(jd_text)
            parsed_jd = call_openai(jd_prompt)
            parsed_jd_data = json.loads(parsed_jd)

            enrich_jd_prompt = build_jd_enrichment_prompt(parsed_jd_data)
            enriched_jd = call_openai_for_enrichment(enrich_jd_prompt)

            if enriched_jd:
                parsed_jd_data["enrichment parameters"] = enriched_jd.get("enrichment parameters", {})
                st.session_state['jd_results'] = parsed_jd_data
                st.success("JD processed successfully!")
        except Exception as e:
            st.error(f"An error occurred during JD processing: {e}")

    if st.session_state['jd_results']:
        st.subheader("Parsed and Enriched JD")
        st.json(st.session_state['jd_results'])
