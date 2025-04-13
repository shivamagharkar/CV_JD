import os
import json
import pdfplumber
import streamlit as st
from fpdf import FPDF
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openAI_API_KEY = os.getenv("openAI_API_KEY")

# STEP 1: Load PDF and extract text
def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

# STEP 2: Define JSON schema and prompts
def buildCV_prompt(resume_text):
    json_schema = {
        "name": "",
        "email": "",
        "phone": "",
        "country": "",
        "city": "",
        "summary": "",
        "skills": [
            {
                'specialized skill': "",
                'common skill': ""
            }
        ],
        "experience": [
            {
                "job_title": "",
                "company": "",
                "start_date": "",
                "end_date": "",
                "description": ""
            }
        ],
        "education": [
            {
                "degree": "",
                "institution": "",
                "start_year": "",
                "end_year": ""
            }
        ],
       "enrichment parameters": [
            {
                "Employment Pattern & Progression": "",
                "Company Type & Sector": "",
                "Education Quality & Ranking": "",
                "Skill Demand & Market Relevance": "",
                "Leadership Experience": "",
                "Budget & Project Management": "",
                "International Experience & Mobility": "",
                "Soft Skills from Sales Calls": "",
                "Personality & Behavioral Traits": [
                    {
                    "Openness": "",
                    "Conscientiousness": "",
                    "Extraversion": "",
                    "Agreeableness": "",
                    "Neuroticism": ""
                    }
                ],
                "Future Career Goals (Sales-Inferred)": "",
                "Salary Expectations (Sales-Inferred)": "",
                "JD Enrichment with Implied Preferences": "",
                "Cultural Fit Indicators": ""
            }
        ]
    }

    prompt = f"""
You are an expert resume parser. Convert the resume text below into this JSON format. Fill in all the relevant fields. Leave the enrichment_parameters field empty.
The JSON schema is as follows:

{json.dumps(json_schema, indent=2)}

Resume:
\"\"\"
{resume_text}
\"\"\"
"""
    return prompt

def buildJD_prompt(job_description_text):
    json_schema = {
        "country": "",
        "city": "",
        "summary": "",
        "skills": [
            {
                'specialized skill': "",
                'common skill': ""
            }
        ],
        "experience": [
            {
                "job_title": "",
                "company": "",
                "start_date": "",
                "end_date": "",
                "description": ""
            }
        ],
        "education": [
            {
                "degree": "",
                "institution": "",
                "start_year": "",
                "end_year": ""
            }
        ],
       "enrichment parameters": [
            {
                "Employment Pattern & Progression": "",
                "Company Type & Sector": "",
                "Education Quality & Ranking": "",
                "Skill Demand & Market Relevance": "",
                "Leadership Experience": "",
                "Budget & Project Management": "",
                "International Experience & Mobility": "",
                "Soft Skills from Sales Calls": "",
                "Future Career Goals (Sales-Inferred)": "",
                "Salary Expectations (Sales-Inferred)": "",
                "JD Enrichment with Implied Preferences": "",
                "Cultural Fit Indicators": ""
            }
        ]
    }

    prompt = f"""
You are an expert Job description parser. Convert the job description text below into this JSON format. Fill in all the relevant fields. Leave the enrichment parameters field empty.
Note that the json schema resembles a resume schema. 
This is because the end goal is to match the resume with the job description. 
However, keep in mind that the schema is to be filled with the job description data.
Again, the enrichment parameters field should be left empty.
The JSON schema is as follows:

{json.dumps(json_schema, indent=2)}

Job Description:
\"\"\"
{job_description_text}
\"\"\"
"""
    return prompt

# STEP 3: Build enrichment prompts
def buildCV_enrichment_prompt(cv_data):
    prompt = f"""
You are an expert in CV enrichment. Analyze the provided CV data and infer the following enrichment parameters:
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

def buildJD_enrichment_prompt(jd_data):
    prompt = f"""
You are an expert in Job Description enrichment. Analyze the provided Job description data and infer the following enrichment parameters:
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

# STEP 4: OpenAI API calls
def call_openai(prompt):
    client = OpenAI(api_key=openAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

def call_openai_for_enrichment(prompt):
    client = OpenAI(api_key=openAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    enriched_cv_content = response.choices[0].message.content
    return json.loads(enriched_cv_content)  # Assuming the response is valid JSON

# Helper function to extract missing points from flagging response
def extract_missing_points(response):
    lines = response.split("\n")
    missing_points = []
    start_extracting = False
    
    for line in lines:
        if "Key Missing Information" in line or "missing information" in line.lower() or "key information" in line.lower():
            start_extracting = True
            continue
        if start_extracting:
            if line.strip().startswith("-") or line.strip().startswith("•") or line.strip().startswith("*"):
                missing_points.append(line.strip("- •*").strip())
            elif line.strip() == "":
                if missing_points:  # Only break if we've found some points
                    break
    
    # If no structured list found, try looking for numbered points
    if not missing_points:
        for line in lines:
            if line.strip().startswith("1.") and "missing" in line.lower():
                missing_points.append(line.strip("1. ").strip())
                for i in range(2, 6):  # Look for points 2-5
                    for next_line in lines[lines.index(line)+1:]:
                        if next_line.strip().startswith(f"{i}."):
                            missing_points.append(next_line.strip(f"{i}. ").strip())
                            break
                break
    
    return missing_points[:5]  # Limit to 5 points

# Main Streamlit application
def main():
    st.set_page_config(page_title="CV-JD Matching Application", layout="wide")
    
    st.title("CV-JD Matching Application")
    st.markdown("This application parses CVs and Job Descriptions, enriches them with AI, and provides insights and interview questions.")
    
    # Create tabs for the different sections
    tab1, tab2, tab3 = st.tabs(["Upload Files", "Results", "Download"])
    
    # Ensure necessary directories exist
    for directory in ["temp", "outputs"]:
        os.makedirs(directory, exist_ok=True)
    
    # State management
    if 'enriched_cv' not in st.session_state:
        st.session_state.enriched_cv = None
    if 'enriched_jd' not in st.session_state:
        st.session_state.enriched_jd = None
    if 'flagging_response' not in st.session_state:
        st.session_state.flagging_response = None
    if 'questionnaire_response' not in st.session_state:
        st.session_state.questionnaire_response = None
    
    # Tab 1: Upload Files
    with tab1:
        st.header("Upload CV and Job Description")
        
        col1, col2 = st.columns(2)
        
        with col1:
            cv_file = st.file_uploader("Upload CV (PDF format)", type="pdf")
        
        with col2:
            jd_file = st.file_uploader("Upload Job Description (PDF format)", type="pdf")
        
        if cv_file and jd_file:
            if st.button("Process Files"):
                with st.spinner("Processing files... This may take a moment."):
                    try:
                        # Save uploaded files temporarily
                        cv_path = os.path.join("temp", cv_file.name)
                        jd_path = os.path.join("temp", jd_file.name)
                        
                        with open(cv_path, "wb") as f:
                            f.write(cv_file.read())
                        with open(jd_path, "wb") as f:
                            f.write(jd_file.read())
                        
                        # Extract text from PDFs
                        cv_text = extract_text_from_pdf(cv_path)
                        jd_text = extract_text_from_pdf(jd_path)
                        
                        # Parse CVs and JDs
                        cv_prompt = buildCV_prompt(cv_text)
                        jd_prompt = buildJD_prompt(jd_text)
                        
                        parsed_cv = json.loads(call_openai(cv_prompt))
                        parsed_jd = json.loads(call_openai(jd_prompt))
                        
                        # Enrich parsed data
                        cv_enrichment_prompt = buildCV_enrichment_prompt(parsed_cv)
                        jd_enrichment_prompt = buildJD_enrichment_prompt(parsed_jd)
                        
                        enriched_cv = call_openai_for_enrichment(cv_enrichment_prompt)
                        enriched_jd = call_openai_for_enrichment(jd_enrichment_prompt)
                        
                        # Store in session state
                        st.session_state.enriched_cv = enriched_cv
                        st.session_state.enriched_jd = enriched_jd
                        
                        # Generate flagging and questionnaire
                        flagging_prompt = f"""
                        Candidate CV:
                        {json.dumps(enriched_cv, indent=2)}

                        Job Description:
                        {json.dumps(enriched_jd, indent=2)}

                        Please answer the following questions:
                        1. Could you please give me an overview of this candidate's CV?
                        2. Could you expand on the missing information that you pointed out? Please explain why they should be important.
                        3. This candidate is applying for the role described in the Job Description. Given the role, what key information is missing from the CV? Sum it up in 5 points.
                        """
                        
                        flagging_response = call_openai(flagging_prompt)
                        st.session_state.flagging_response = flagging_response
                        
                        missing_points = extract_missing_points(flagging_response)
                        
                        questionnaire_prompt = f"""
                        Based on the missing points identified earlier, please draw up a 5-question questionnaire to be asked during an interview with the candidate. 
                        These questions should be formulated in a way that can give the candidate the possibility to explain why that information is missing. 
                        One question for each of the points mentioned below:

                        {json.dumps(missing_points, indent=2)}
                        """
                        
                        questionnaire_response = call_openai(questionnaire_prompt)
                        st.session_state.questionnaire_response = questionnaire_response
                        
                        # Save outputs as PDFs
                        flagging_pdf_path = os.path.join("outputs", "flagging_output.pdf")
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font("Arial", size=12)
                        pdf.multi_cell(0, 10, flagging_response)
                        pdf.output(flagging_pdf_path)
                        
                        questionnaire_pdf_path = os.path.join("outputs", "questionnaire.pdf")
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font("Arial", size=12)
                        pdf.multi_cell(0, 10, questionnaire_response)
                        pdf.output(questionnaire_pdf_path)
                        
                        # Save JSON files
                        with open(os.path.join("outputs", "enriched_cv.json"), "w") as f:
                            json.dump(enriched_cv, f, indent=4)
                        
                        with open(os.path.join("outputs", "enriched_jd.json"), "w") as f:
                            json.dump(enriched_jd, f, indent=4)
                        
                        st.success("Processing complete! Go to the Results tab to view the output.")
                    
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
    
    # Tab 2: Results
    with tab2:
        if st.session_state.enriched_cv and st.session_state.enriched_jd:
            st.header("Analysis Results")
            
            st.subheader("CV Overview")
            cv_expander = st.expander("Show Enriched CV JSON")
            with cv_expander:
                st.json(st.session_state.enriched_cv)
            
            st.subheader("Job Description Overview")
            jd_expander = st.expander("Show Enriched JD JSON")
            with jd_expander:
                st.json(st.session_state.enriched_jd)
            
            st.subheader("CV Analysis and Missing Information")
            st.markdown(st.session_state.flagging_response)
            
            st.subheader("Interview Questions")
            st.markdown(st.session_state.questionnaire_response)
        else:
            st.info("Please upload and process files first.")
    
    # Tab 3: Download
    with tab3:
        if st.session_state.enriched_cv and st.session_state.enriched_jd:
            st.header("Download Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                with open(os.path.join("outputs", "flagging_output.pdf"), "rb") as file:
                    st.download_button(
                        label="Download CV Analysis",
                        data=file,
                        file_name="cv_analysis.pdf",
                        mime="application/pdf"
                    )
                
                with open(os.path.join("outputs", "enriched_cv.json"), "rb") as file:
                    st.download_button(
                        label="Download Enriched CV (JSON)",
                        data=file,
                        file_name="enriched_cv.json",
                        mime="application/json"
                    )
            
            with col2:
                with open(os.path.join("outputs", "questionnaire.pdf"), "rb") as file:
                    st.download_button(
                        label="Download Interview Questions",
                        data=file,
                        file_name="interview_questions.pdf",
                        mime="application/pdf"
                    )
                
                with open(os.path.join("outputs", "enriched_jd.json"), "rb") as file:
                    st.download_button(
                        label="Download Enriched JD (JSON)",
                        data=file,
                        file_name="enriched_jd.json",
                        mime="application/json"
                    )
        else:
            st.info("Please upload and process files first.")

if __name__ == "__main__":
    main()
