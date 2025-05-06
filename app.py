import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')




import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import spacy
# from youtubesearchpython import VideosSearch  # For YouTube suggestions

# Load environment variables from .env
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load or install spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# PDF text extraction
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# Skill extraction using spaCy
def extract_skills(text):
    doc = nlp(text)
    skills = [
        ent.text.lower()
        for ent in doc.ents
        if ent.label_ in ["SKILL", "ORG", "WORK_OF_ART", "PRODUCT", "LANGUAGE"]
    ]
    return list(set(skills))

# Calculate matching skill percentage
def get_skill_match_percentage(resume_skills, job_skills):
    if not job_skills:
        return 0
    common_skills = set(resume_skills).intersection(set(job_skills))
    return len(common_skills) / len(job_skills) * 100

# Suggest YouTube videos for missing skills
def suggest_youtube_videos(missing_skills):
    video_links = []
    for skill in missing_skills:
        search = VideosSearch(f'{skill} tutorial', limit=1)
        result = search.result()
        if result['result']:
            video_url = result['result'][0]['link']
            video_links.append((skill, video_url))
    return video_links

# Streamlit app layout
st.set_page_config(page_title="Resume & JD Analyzer", layout="centered")

# Sidebar UI
with st.sidebar:
    st.title("🔧 Navigation Menu")
    st.markdown("---")
    st.markdown("### 📂 Upload")
    st.info("Upload your resume and paste a JD to begin analysis.")

    st.markdown("### 💡 About")
    st.markdown("""
        This app analyzes resumes vs job descriptions using:
        - 🧠 Google Gemini AI
        - 📊 Skill comparison
        - 🎥 YouTube learning suggestions
    """)

    st.markdown("### 📬 Connect")
    st.markdown("[🌐 GitHub](https://github.com/yourprofile)")
    st.markdown("[📧 Email](mailto:your@email.com)")
    st.markdown("---")
    st.markdown("© 2025 Resume Analyzer")

# Main UI
st.title("🧠 Resume & Job Description Analyzer")

uploaded_resume = st.file_uploader("📄 Upload your resume (PDF)", type="pdf")
job_description = st.text_area("📝 Paste the job description here")

if uploaded_resume and job_description:
    if st.button("🔍 Analyze"):
        resume_text = extract_text_from_pdf(uploaded_resume)
        resume_skills = extract_skills(resume_text)
        job_skills = extract_skills(job_description)

        # Show extracted skills
        st.subheader("✅ Extracted Skills from Resume")
        st.write(resume_skills)

        st.subheader("🎯 Extracted Skills from Job Description")
        st.write(job_skills)

        # Match percentage
        match_percentage = get_skill_match_percentage(resume_skills, job_skills)
        st.subheader("📊 Skill Match Analysis")
        st.write(f"✅ Matching Skills Percentage: `{match_percentage:.2f}%`")

        # Missing skills and YouTube suggestions
        missing_skills = list(set(job_skills) - set(resume_skills))
        st.write(f"🛠️ Missing Skills: `{missing_skills}`")

        # if missing_skills:
        #     st.subheader("🎥 Suggested YouTube Tutorials")
        #     video_links = suggest_youtube_videos(missing_skills)
        #     for skill, link in video_links:
        #         st.markdown(f"- **{skill.title()}**: [Watch Tutorial]({link})")

        # Gemini AI analysis
        prompt = (
            "Compare the following resume and job description. Provide a percentage match, "
            "highlight missing skills, and suggest relevant learning resources.\n\n"
            f"Resume:\n{resume_text}\n\nJob Description:\n{job_description}"
        )

        try:
            model = genai.GenerativeModel("models/gemini-1.5-flash")
            response = model.generate_content(prompt)
            st.subheader("🤖 Gemini AI Analysis")
            st.write(response.text)
        except Exception as e:
            st.error(f"Gemini API error: {e}")
else:
    st.info("📥 Upload a resume and paste a job description to enable analysis.")
