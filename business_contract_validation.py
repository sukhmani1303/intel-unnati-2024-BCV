import re
import fitz  # PyMuPDF
from transformers import pipeline
import streamlit as st

# Define Streamlit page configuration
st.set_page_config(
    page_title="Business Contract Validation",
    page_icon="📃",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Load NER and summarizer models
@st.cache(allow_output_mutation=True)
def load_models():
    ner_pipeline = pipeline(
        "ner",
        model="dbmdz/bert-large-cased-finetuned-conll03-english",
        aggregation_strategy="simple",
    )
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return ner_pipeline, summarizer


ner_pipeline, summarizer = load_models()


# Function to preprocess text
def preprocess_text(text):
    return re.sub(r"\s+", " ", text)  # Normalize whitespace


# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        page_text = page.get_text("text")
        text += page_text.encode("latin1", "ignore").decode("utf-8")
    return text


# Function to extract clauses and titles from text
def extract_clauses_and_titles(text):
    clause_pattern = re.compile(r"(\d+(\.\d+)*)\.\s+([^\n]+)")
    matches = clause_pattern.findall(text)
    return [(match[0], match[2].strip()) for match in matches]


# Function to compare clauses and determine deviations
def compare_clauses(template_clauses, contract_clauses):
    template_clause_dict = {clause: title for clause, title in template_clauses}
    contract_clause_dict = {clause: title for clause, title in contract_clauses}
    deviations = []

    for clause, title in template_clause_dict.items():
        if clause not in contract_clause_dict:
            deviations.append((clause, title, "Missing in Contract"))
        elif contract_clause_dict[clause] != title:
            deviations.append(
                (
                    clause,
                    title,
                    f"Different in Contract: {contract_clause_dict[clause]}",
                )
            )

    for clause, title in contract_clause_dict.items():
        if clause not in template_clause_dict:
            deviations.append((clause, title, "Extra in Contract"))

    return deviations


# Function to generate detailed summary with highlights
def extract_detailed_summary(text, entities):
    text = preprocess_text(text)

    if len(text) < 50:
        return "Input text is too short for summarization."

    try:
        summary = summarizer(text, max_length=500, min_length=150, do_sample=False)
        text_summary = summary[0]["summary_text"]
    except Exception as e:
        return f"Summary generation failed: {e}"

    highlighted_summary = text_summary
    for entity in entities:
        entity_text = re.escape(entity["word"])
        highlighted_summary = re.sub(
            rf"\b{entity_text}\b",
            f'<span class="highlight">{entity["word"]}</span>',
            highlighted_summary,
        )

    return f"Text Summary:\n{highlighted_summary}\n\n"


# Streamlit application
st.title("Business Contract Validation 📃")
st.write("Upload your business contract for validation.")

# Upload template and contract PDF files
uploaded_template_file = st.file_uploader(
    "Choose a Template PDF file", type="pdf", key="template"
)
uploaded_contract_file = st.file_uploader(
    "Choose a Contract PDF file", type="pdf", key="contract"
)

# Process upon submission
if st.button("Submit"):
    if uploaded_template_file and uploaded_contract_file:
        with st.spinner("Processing..."):
            # Extract text from PDF files
            template_text = extract_text_from_pdf(uploaded_template_file)
            contract_text = extract_text_from_pdf(uploaded_contract_file)

            # Extract clauses and titles
            template_clauses = extract_clauses_and_titles(template_text)
            contract_clauses = extract_clauses_and_titles(contract_text)

            # Display clauses from template and contract
            st.subheader("Extracted Clauses and Titles from Template")
            for clause, title in template_clauses:
                st.markdown(f"**{clause}. {title}**")

            st.subheader("Extracted Clauses and Titles from Contract")
            for clause, title in contract_clauses:
                st.markdown(f"**{clause}. {title}**")

            # Compare clauses and show deviations
            deviations = compare_clauses(template_clauses, contract_clauses)
            st.subheader("Deviations")
            if deviations:
                for clause, title, deviation in deviations:
                    st.markdown(f"**{clause}. {title}** - {deviation}")
            else:
                st.write("No deviations detected.")

            # Perform NER on contract text
            entities = ner_pipeline(contract_text)

            # Display detailed summary
            st.subheader("Detailed Contract Summary")
            contract_summary = extract_detailed_summary(contract_text, entities)
            st.markdown(contract_summary, unsafe_allow_html=True)

            # Show detected entities
            st.subheader("Entities Detected")
            unique_entities = {
                entity["word"]: entity["entity_group"] for entity in entities
            }
            for entity, label in unique_entities.items():
                st.write(f"Entity: {entity}, Label: {label}")
    else:
        st.write("Please upload both template and contract PDF files.")
