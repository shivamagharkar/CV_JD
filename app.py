import streamlit as st
import os
from io import BytesIO
import pdfplumber
import json
import openai
from dotenv import load_dotenv


load_dotenv()
# Use session state to store API keys and processed data
if 'openAI_API_KEY' not in st.session_state:
    st.session_state['openAI_API_KEY'] = os.getenv("openAI_API_KEY")

if 'cv_results' not in st.session_state:
    st.session_state['cv_results'] = None
if 'jd_results' not in st.session_state:
    st.session_state['jd_results'] = None


# --- Helper Functions (Extracted from Notebook) ---
def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def build_cv_prompt(resume_text):
    json_schema = {
        "name": "",
        "email": "",
        "phone": "",
        "country": "",
        "city": "",
        "summary": "",
        "skills": [{"specialized skill": "", "common skill": ""}],
        "experience": [{"job_title": "", "company": "", "start_date": "", "end_date": "", "description": ""}],
        "education": [{"degree": "", "institution": "", "start_year": "", "end_year": ""}],
        "enrichment parameters": [{"Employment Pattern & Progression": "", "Company Type & Sector": "", "Education Quality & Ranking": "",
                                   "Skill Demand & Market Relevance": "", "Leadership Experience": "", "Budget & Project Management": "",
                                   "International Experience & Mobility": "", "Soft Skills from Sales Calls": "",
                                   "Personality & Behavioral Traits": [{"Openness": "", "Conscientiousness": "", "Extraversion": "",
                                                                        "Agreeableness": "", "Neuroticism": ""}],
                                   "Future Career Goals (Sales-Inferred)": "", "Salary Expectations (Sales-Inferred)": "",
                                   "JD Enrichment with Implied Preferences": "", "Cultural Fit Indicators": ""}]
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

def build_jd_prompt(jd_text):  # Modified for JD
    json_schema = {
        "country": "",
        "city": "",
        "summary": "",
        "skills": [{"specialized skill": "", "common skill": ""}],
        "experience": [{"job_title": "", "company": "", "start_date": "", "end_date": "", "description": ""}],
        "education": [{"degree": "", "institution": "", "start_year": "", "end_year": ""}],
        "enrichment parameters": [{"Employment Pattern & Progression": "", "Company Type & Sector": "", "Education Quality & Ranking": "",
                                   "Skill Demand & Market Relevance": "", "Leadership Experience": "", "Budget & Project Management": "",
                                   "International Experience & Mobility": "", "Soft Skills from Sales Calls": "",
                                   "Future Career Goals (Sales-Inferred)": "", "Salary Expectations (Sales-Inferred)": "",
                                   "JD Enrichment with Implied Preferences": "", "Cultural Fit Indicators": ""}]
    }

    prompt = f"""You are an expert Job description parser. Convert the job description text below into this JSON format. 
Fill in all the relevant fields. Leave the enrichment parameters field empty.
Note that the json schema resembles a resume schema. 
This is because the end goal is to match the resume with the job description. 
However, keep in mind that the schema is to be filled with the job description data.
Again , the enrichment parameters field should be left empty.
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
    client = OpenAI(api_key=st.session_state['openAI_API_KEY'])
    response = client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": prompt}], temperature=0)
    return response.choices[0].message.content


def build_cv_enrichment_prompt(cv_data):
    prompt = f"""You are an expert in CV enrichment. Analyze the provided CV data and infer the following enrichment parameters:
- Employment Pattern & Progression: Describe the career trajectory and progression.
- Company Type & Sector: Identify the type and sector of companies worked for.
- Education Quality & Ranking: Assess the quality and ranking of educational institutions.
- Skill Demand & Market Relevance: Evaluate the relevance of skills in the current market.
- Leadership Experience: Highlight leadership roles and responsibilities.
- Budget & Project Management: Detail experience in managing budgets and projects.
- International Experience & Mobility: Indicate international exposure and mobility.
- Soft Skills from Sales Calls: Infer soft skills demonstrated in sales or communication.
- Personality & Behavioral Traits: Deduce personality traits and behaviors.
- Future Career Goals (Sales-Inferred): Predict future career aspirations based on sales roles.
- Salary Expectations (Sales-Inferred): Estimate salary expectations based on experience.
- JD Enrichment with Implied Preferences: Enrich job descriptions with implied preferences.
- Cultural Fit Indicators: Suggest cultural fit indicators for potential roles.

Also, analyze the candidate's Personality & Behavioral Traits according to the Big Five (OCEAN) model. Use the resume's tone, accomplishments, language, career path and the above inferred enrichment parameters to estimate the following traits:
Personality & Behavioral Traits: 
    "Openness": "How open is the candidate to new experiences and ideas?"
    "Conscientiousness": "How organized and dependable is the candidate?"
    "Extraversion": "How outgoing and energetic is the candidate?"
    "Agreeableness": "How friendly and compassionate is the candidate?"
    "Neuroticism": "How emotionally stable is the candidate?"

For each Personality and Behavioral Trait, provide a rating (High, Moderate, or Low).

Here is the CV data:
{json.dumps(cv_data, indent=2)}

Please analyze and fill in the enrichment parameters. Return the enriched CV data in JSON format. Please respond ONLY with raw JSON. Do not include explanations, markdown, or code block formatting.
"""
    return prompt


def build_jd_enrichment_prompt(jd_data): # Modified for JD
    prompt = f"""You are an expert in Job Description enrichment. Analyze the provided Job description data and infer the following enrichment parameters:
- Employment Pattern & Progression: Describe the required career trajectory and progression for an ideal candidate.
- Company Type & Sector: Identify the type and sector of company.
- Education Quality & Ranking: potential quality and ranking of educational institutions of the candidate.
- Skill Demand & Market Relevance: Evaluate the relevance of skills in the current market.
- Leadership Experience: Highlight leadership roles and responsibilities for a potential candidate.
- Budget & Project Management: Detail experience in managing budgets and projects.
- International Experience & Mobility: Indicate international exposure and mobility for a potential candidate.
- Soft Skills from Sales Calls: Infer soft skills demonstrated in sales or communication.
- Personality & Behavioral Traits: Deduce personality traits and behaviors for this role.
- Future Career Goals (Sales-Inferred): Predict future career aspirations based on sales roles.
- Salary Expectations (Sales-Inferred): Estimate salary expectations for this role.
- JD Enrichment with Implied Preferences: leave this empty
- Cultural Fit Indicators: Suggest cultural fit indicators for potential candidates.

Also, analyze a potential candidate's Personality & Behavioral Traits according to the Big Five (OCEAN) model. Use the job description's tone, requiremets, language and the above inferred enrichment parameters to estimate the following traits:
Personality & Behavioral Traits: 
    "Openness": "How open is the candidate to new experiences and ideas?"
    "Conscientiousness": "How organized and dependable is the candidate?"
    "Extraversion": "How outgoing and energetic is the candidate?"
    "Agreeableness": "How friendly and compassionate is the candidate?"
    "Neuroticism": "How emotionally stable is the candidate?"

For each Personality and Behavioral Trait, provide a rating (High, Moderate, or Low).

Here is the job description data:
{json.dumps(jd_data, indent=2)}

Please analyze and fill in the enrichment parameters. Return the enriched job description data in JSON format. Please respond ONLY with raw JSON. Do not include explanations, markdown, or code block formatting.
"""
    return prompt


def call_openai_for_enrichment(prompt):
    openai.api_key = st.session_state['openAI_API_KEY']
    client = OpenAI(api_key=st.session_state['openAI_API_KEY'])
    response = client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": prompt}], temperature=0)
    try:
        enriched_data = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON: {e}")
        st.text("Response content:")
        st.text(response.choices[0].message.content)
        return None
    return enriched_data

# --- Streamlit App ---
st.title("CV/JD Analyzer")

# --- API Key Input ---
api_key_input = st.text_input("Enter your OpenAI API Key", type="password")
if api_key_input:
    st.session_state['openAI_API_KEY'] = api_key_input


# --- CV Processing Section ---
st.header("CV Processing")
cv_file = st.file_uploader("Upload CV (PDF)", type="pdf", key="cv")

if cv_file is not None:
    if st.button("Process CV"):
        with st.spinner("Processing CV..."):
            try:
                # Read the content of the uploaded file
                bytes_data = cv_file.getvalue()
                
                # Use BytesIO to create a file-like object
                with BytesIO(bytes_data) as pdf_file:
                    
                    # 2. Extract text
                    cv_text = extract_text_from_pdf(pdf_file)

                # 3. Parse CV
                cv_prompt = build_cv_prompt(cv_text)
                parsed_cv = call_openai(cv_prompt)
                parsed_cv_data = json.loads(parsed_cv)

                # 4. Enrich CV
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
            st.write(st.session_state['cv_results'])
            

# --- JD Processing Section ---
st.header("JD Processing")
jd_file = st.file_uploader("Upload JD (PDF)", type="pdf", key="jd")

if jd_file is not None:
    if st.button("Process JD"):
        with st.spinner("Processing JD..."):
            try:
                # Read the content of the uploaded file
                bytes_data = jd_file.getvalue()
                
                # Use BytesIO to create a file-like object
                with BytesIO(bytes_data) as pdf_file:
                    
                    # 2. Extract text
                    jd_text = extract_text_from_pdf(pdf_file)

                # 3. Parse JD
                jd_prompt = build_jd_prompt(jd_text)
                parsed_jd = call_openai(jd_prompt)
                parsed_jd_data = json.loads(parsed_jd)

                # 4. Enrich JD
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
            st.write(st.session_state['jd_results'])
