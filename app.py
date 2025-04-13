import os
import json
import streamlit as st
from fpdf import FPDF
from functions import extract_text_from_pdf, buildCV_prompt,buildJD_prompt, call_openai, buildCV_enrichment_prompt, buildJD_enrichment_prompt, call_openai_for_enrichment

# Streamlit app setup
st.title("CV-JD Parsing Workflow")

# Step 1: Upload CV and JD
st.header("Upload CV and Job Description")
cv_file = st.file_uploader("Upload CV (PDF format)", type="pdf")
jd_file = st.file_uploader("Upload Job Description (PDF format)", type="pdf")

if cv_file and jd_file:
    # Step 2: Parse CV and JD
    st.header("Parsed and Enriched JSON")

    # Save uploaded files temporarily
    cv_path = os.path.join("temp", cv_file.name)
    jd_path = os.path.join("temp", jd_file.name)
    os.makedirs("temp", exist_ok=True)
    with open(cv_path, "wb") as f:
        f.write(cv_file.read())
    with open(jd_path, "wb") as f:
        f.write(jd_file.read())

    # Extract text from PDFs
    cv_text = extract_text_from_pdf(cv_path)
    jd_text = extract_text_from_pdf(jd_path)

    # Build prompts and call OpenAI for parsing
    cv_prompt = buildCV_prompt(cv_text)
    jd_prompt = buildJD_prompt(jd_text)
    parsed_cv = json.loads(call_openai(cv_prompt))
    parsed_jd = json.loads(call_openai(jd_prompt))

    # Enrich parsed data
    cv_enrichment_prompt = buildCV_enrichment_prompt(parsed_cv)
    jd_enrichment_prompt = buildJD_enrichment_prompt(parsed_jd)
    enriched_cv = call_openai_for_enrichment(cv_enrichment_prompt)
    enriched_jd = call_openai_for_enrichment(jd_enrichment_prompt)

    print("Enriched CV:", enriched_cv)
    print("Enriched JD:", enriched_jd)
    # Display enriched JSON
    st.subheader("Enriched CV JSON")
    st.json(enriched_cv)

    st.subheader("Enriched JD JSON")
    st.json(enriched_jd)

    # Step 3: Flagging and Questionnaire
    st.header("Flagging and Questionnaire")

    # Combine CV and JD for context
    context = {
        "Candidate CV": enriched_cv,
        "Job Description": enriched_jd
    }

    # Build flagging prompt
    flagging_prompt = f"""
    Candidate CV:
    {json.dumps(enriched_cv, indent=2)}

    Job Description:
    {json.dumps(enriched_jd, indent=2)}

    Please answer the following questions:
    1. Could you please give me an overview of this candidate's CV?
    2. Could you expand on the missing information that you pointed out? Please explain why they should be important.
    3. This candidate is applying for the role of 'Chief Legal Officer.' Given the role, what key information is missing from his CV? Sum it up in 5 points.
    """

    flagging_response = call_openai(flagging_prompt)

    # Extract missing points
    def extract_missing_points(response):
        lines = response.split("\n")
        missing_points = []
        start_extracting = False
        for line in lines:
            if "Key Missing Information for Role" in line:
                start_extracting = True
                continue
            if start_extracting:
                if line.strip().startswith("-") or line.strip().startswith("•"):
                    missing_points.append(line.strip("- ").strip())
                elif line.strip() == "":
                    break
        return missing_points

    missing_points = extract_missing_points(flagging_response)

    # Display flagging response
    st.subheader("Flagging Response")
    st.text(flagging_response)

    # Build questionnaire prompt
    questionnaire_prompt = f"""
    Based on the missing points identified earlier, please draw up a 5-question questionnaire to be asked by our Sales Team during an interview with the candidate. These questions should be formulated in a way that can give the candidate the possibility to explain why that information is missing. One question for each of the points mentioned below:

    {json.dumps(missing_points, indent=2)}
    """

    questionnaire_response = call_openai(questionnaire_prompt)

    # Display questionnaire
    st.subheader("Questionnaire")
    st.text(questionnaire_response)

    # Save outputs as PDFs
    output_folder = "outputs"
    os.makedirs(output_folder, exist_ok=True)

    # Save flagging response
    flagging_pdf_path = os.path.join(output_folder, "flagging_output.pdf")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, flagging_response)
    pdf.output(flagging_pdf_path)

    # Save questionnaire
    questionnaire_pdf_path = os.path.join(output_folder, "questionnaire.pdf")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, questionnaire_response)
    pdf.output(questionnaire_pdf_path)

    st.success(f"Flagging and questionnaire saved to {output_folder}")
