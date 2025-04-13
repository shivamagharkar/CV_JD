import os
import pdfplumber
import json
import openai
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
API_KEY=os.getenv("API_KEY")
openAI_API_KEY=os.getenv("openAI_API_KEY")
# STEP 1: Load PDF and extract text

def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

# STEP 2: Define JSON schema and prompt
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

# STEP 3: Call OpenAI API
def call_openai(prompt):
    openai.api_key = openAI_API_KEY
    client= OpenAI(api_key=openAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

# STEP 4: Main function to process all PDFs
def main():
    input_folder = "./sample_CVs"
    output_folder = "./sentiment_CVs"

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".pdf"):  # Ignore non-PDF files
            pdf_path = os.path.join(input_folder, file_name)
            print(f"Processing: {file_name}")

            # Extract text from PDF
            resume_text = extract_text_from_pdf(pdf_path)

            # Build prompt and call OpenAI
            prompt = buildCV_prompt(resume_text)
            parsed_cv = call_openai(prompt)

            # Save the parsed CV to the output folder
            output_file = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_parsed.json")
            with open(output_file, "w") as f:
                f.write(parsed_cv)

            print(f"Saved parsed CV to: {output_file}")


# STEP 1: Build enrichment prompt
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

# STEP 2: Call OpenAI API for enrichment
def call_openai_for_enrichment(prompt):
    openai.api_key = openAI_API_KEY
    client= OpenAI(api_key=openAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    enriched_cv_content = response.choices[0].message.content
    return json.loads(enriched_cv_content)  # Assuming the response is valid JSON

# STEP 3: Main function to process all parsed CVs
def main():
    input_folder = "./gpt_parsed_CVs"
    output_folder = "./sentiment_CVs"

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".json"):
                    input_file_path = os.path.join(input_folder, file_name)
                    print(f"Processing: {file_name}")

                    # Load the parsed CV data
                    with open(input_file_path, "r") as f:
                        cv_data = json.load(f)

                    # Build enrichment prompt and call OpenAI
                    prompt = buildCV_enrichment_prompt(cv_data)
                    enriched_cv = call_openai_for_enrichment(prompt)

                    # Merge enrichment parameters into the original CV data
                    cv_data["enrichment parameters"] = enriched_cv.get("enrichment parameters", {})

                    # Save the enriched CV to the output folder
                    output_file_path = os.path.join(output_folder, file_name)
                    with open(output_file_path, "w") as f:
                        json.dump(cv_data, f, indent=4)

                    print(f"Saved enriched CV to: {output_file_path}")




# STEP 2: Define JSON schema and prompt
def buildJD_prompt(resume_text):
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
Again , the enrichment parameters field should be left empty.
The JSON schema is as follows:

{json.dumps(json_schema, indent=2)}

Job Description:
\"\"\"
{resume_text}
\"\"\"
"""
    return prompt

# STEP 3: Call OpenAI API
def call_openai(prompt):
    openai.api_key = openAI_API_KEY
    client= OpenAI(api_key=openAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

# STEP 4: Main function to process all PDFs
def main():
    input_folder = "./JD_pdfs"
    output_folder = "./Jd_parsed"

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".pdf"):  # Ignore non-PDF files
            pdf_path = os.path.join(input_folder, file_name)
            print(f"Processing: {file_name}")

            # Extract text from PDF
            resume_text = extract_text_from_pdf(pdf_path)

            # Build prompt and call OpenAI
            prompt = buildJD_prompt(resume_text)
            parsed_cv = call_openai(prompt)

            # Save the parsed CV to the output folder
            output_file = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_parsed.json")
            with open(output_file, "w") as f:
                f.write(parsed_cv)

            print(f"Saved parsed CV to: {output_file}")




# STEP 1: Build enrichment prompt
def buildJD_enrichment_prompt(cv_data):
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
{json.dumps(cv_data, indent=2)}

Please analyze and fill in the enrichment parameters. Return the enriched job description data in JSON format. Please respond ONLY with raw JSON. Do not include explanations, markdown, or code block formatting.

"""
    return prompt



# STEP 3: Main function to process all parsed CVs
def main():
    input_folder = "./JD_parsed"
    output_folder = "./JD_enriched"

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".json"):
                    input_file_path = os.path.join(input_folder, file_name)
                    print(f"Processing: {file_name}")

                    # Load the parsed CV data
                    with open(input_file_path, "r") as f:
                        cv_data = json.load(f)

                    # Build enrichment prompt and call OpenAI
                    prompt = buildJD_enrichment_prompt(cv_data)
                    enriched_cv = call_openai_for_enrichment(prompt)

                    # Merge enrichment parameters into the original CV data
                    cv_data["enrichment parameters"] = enriched_cv.get("enrichment parameters", {})

                    # Save the enriched CV to the output folder
                    output_file_path = os.path.join(output_folder, file_name)
                    with open(output_file_path, "w") as f:
                        json.dump(cv_data, f, indent=4)

                    print(f"Saved enriched CV to: {output_file_path}")




from fpdf import FPDF

# Load the JSON files
def load_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


# Function to extract missing points from the response
def extract_missing_points(response):
    # Split the response into lines
    lines = response.split("\n")
    missing_points = []

    # Look for the section with "Key Missing Information for CLO Role"
    start_extracting = False
    for line in lines:
        if "Key Missing Information for CLO Role" in line:
            start_extracting = True
            continue
        if start_extracting:
            # Extract points (assuming they are listed with a dash or number)
            if line.strip().startswith("-") or line.strip().startswith("•"):
                missing_points.append(line.strip("- ").strip())
            elif line.strip() == "":
                break  # Stop if there's an empty line after the points

    return missing_points

# Main function
def main():
    # Load the relevant JSON files
    candidate_cv = load_json("/Users/prayagsharma/Documents/ACP/gpt_enriched_CVs/Daniel_Carter_parsed.json")
    job_description = load_json("/Users/prayagsharma/Documents/ACP/JD_enriched/job_description_4_parsed.json")

    # Combine the JSON data into a single context
    context = f"""
    Candidate CV:
    {json.dumps(candidate_cv, indent=2)}

    Job Description:
    {json.dumps(job_description, indent=2)}
    """

    # Step 1: Ask questions 1, 2, and 3
    prompt_1_3 = f"""
    {context}

    Please answer the following questions:
    1. Could you please give me an overview of this candidate's CV?
    2. Could you expand on the missing information that you pointed out? Please explain why they should be important. (These can also be used as a base for questions, but they can be subjective to the interviewer, e.g., school grades or employment pattern.)
    3. This candidate is applying for the role of "Chief Legal Officer." Given the role, what key information is missing from his CV? Sum it up in 5 points.
    """
    response_1_3 = call_openai(prompt_1_3)
    print("Response to Questions 1, 2, and 3:")
    print(response_1_3)

    # Step 2: Extract the 5 missing points from the response
    print("\nExtracting missing points for follow-up questions...")
   # Parse the response to extract the missing points

    # Extract missing points from the response
    missing_points = extract_missing_points(response_1_3)

    flagging_output= "./flagging_output"
    # Ensure the "flagging_output" folder exists
    os.makedirs(flagging_output, exist_ok=True)
    # Save response_1_3 and missing_points as pdf
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    # Add content to the PDF
    pdf.multi_cell(0, 10, response_1_3)
    pdf.ln(10)  # Add a line break
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, "Missing Points:", ln=True)
    pdf.set_font("Arial", size=12)
    for point in missing_points:
        pdf.multi_cell(0, 10, f"- {point}")
    # Save the PDF to the "flagging_output" folder
    output_file = os.path.join(flagging_output, "flagging_output.pdf")
    pdf.output(output_file)
    print(f"Flagging output saved as PDF to: {output_file}")
    print("Missing Points:")
    for point in missing_points:
        print(f"- {point}")
    
    # Step 3: Ask question 4 based on the missing points
    prompt_4 = f"""
    Based on the missing points identified earlier, please draw up a 5-question questionnaire to be asked by our Sales Team during an interview with the candidate. These questions should be formulated in a way that can give the candidate the possibility to explain why that information is missing. One question for each of the points mentioned below:

    {json.dumps(missing_points, indent=2)}
    """
    response_4 = call_openai(prompt_4)
    # Ensure the "questionnaire" folder exists
    output_folder = "./questionnaire"
    os.makedirs(output_folder, exist_ok=True)

    # Save the response as a PDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Add content to the PDF
    pdf.multi_cell(0, 10, response_4)

    # Save the PDF to the "questionnaire" folder
    output_file = os.path.join(output_folder, "questionnaire.pdf")
    pdf.output(output_file)

    print(f"Questionnaire saved as PDF to: {output_file}")
    print("Response to Question 4:")
    print(response_4)



if __name__ == "__main__":
    main()