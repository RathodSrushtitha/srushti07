import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import zipfile
import io

load_dotenv()

st.set_page_config(
    page_title="SOP Generator"
)

# CSS to hide the footer and GitHub logo
hide_streamlit_style = """    
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Helper function to create embeddings
def create_embeddings_from_pdf(file_path):
    """Load a PDF, split it into chunks, and create embeddings. Returns the vector store."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    texts = text_splitter.split_documents(documents)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(documents=texts, embedding=embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore

# Helper function to answer questions
def answer_question_from_vectorstore(question):
    """Answer a question using the provided vector store."""
    prompt = ChatPromptTemplate.from_template("""Please provide information for creating a Standard Operating Procedure (SOP) in the following format. Ensure the response aligns exactly with the structure below:  

1. **Title**  
   - SOP Number: [Provide a unique identifier for the SOP]  
   - SOP Title: [Provide a descriptive title for the procedure]  
   - Effective Date: [Specify the date the SOP becomes effective]  
   - Review Date: [Specify the scheduled review date]  
   - Version: [Provide the version number]  

2. **Purpose**  
   - Description: [State the purpose of the SOP and why the procedure exists]  

3. **Scope**  
   - Applies to: [Specify the departments, teams, or individuals the SOP applies to]  

4. **Responsibilities**  
   - Responsible Parties: [Define the roles and responsibilities for executing the SOP]  

5. **Procedure**  
   - Step-by-Step Instructions:  
     - Step 1: [Detailed description of the first step]  
     - Step 2: [Detailed description of the next step]  
     - Step 3: [Continue until all necessary steps are outlined]  

6. **Supporting Documentation**  
   - References: [Provide links or titles of relevant documents, manuals, or guidelines]  

7. **Definitions**  
   - Key Terms: [Explain any technical terms, abbreviations, or jargon used]  

8. **Review and Approval**  
   - Reviewed by: [Specify the name and position of the reviewer]  
   - Approved by: [Specify the name and position of the approver]  

Please fill in each section accurately and completely."
        Question: {question} 
        Context: {context} 
        Answer:
    """)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever(search_type="mmr", search_kwargs={"k": 20})
    llm = ChatGroq(temperature=0.2, model_name="llama-3.1-70b-versatile", max_tokens=8000)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, chain_type_kwargs={"prompt": prompt})
    result = qa_chain.invoke({"query": question})
    return result["result"]

# Helper function to generate PDF with reportlab, ensuring bold text for headers
# Helper function to generate PDF with reportlab, ensuring bold text for headers
def generate_pdf_with_reportlab(content, output_file="sop_output.pdf"):
    """Generate a PDF with bold headers for SOP content."""
    c = canvas.Canvas(output_file, pagesize=letter)
    width, height = letter

    # Start at the top of the page
    y_position = height - 50
    lines = content.split("\n")  # Split content into lines

    # Define left margin
    left_margin = 50
    max_width = width - left_margin - 50  # Adjust the width to avoid cutting off text

    for line in lines:
        # Check for bold header format (lines that start and end with '**')
        if line.startswith("**") and line.endswith("**"):
            c.setFont("Helvetica-Bold", 12)  # Set bold font for headings
            line = line.strip("**")  # Remove the ** symbols for bold formatting
        else:
            c.setFont("Helvetica", 10)  # Regular font for body text
        
        # Check if line width exceeds maximum available width
        text_width = c.stringWidth(line, "Helvetica", 10)
        
        if text_width > max_width:
            # If the text is too long, break it into smaller parts
            words = line.split()
            current_line = words[0]
            
            for word in words[1:]:
                # Check the width of adding this word to the current line
                if c.stringWidth(current_line + " " + word, "Helvetica", 10) < max_width:
                    current_line += " " + word
                else:
                    # Draw the current line, then move to the next line
                    c.drawString(left_margin, y_position, current_line)
                    y_position -= 15
                    current_line = word
            
            # Draw the remaining part of the line
            c.drawString(left_margin, y_position, current_line)
            y_position -= 15
        else:
            # If text is within width, simply draw it
            c.drawString(left_margin, y_position, line)
            y_position -= 15

        # Add a new page if text exceeds page height
        if y_position < 50:
            c.showPage()
            y_position = height - 50  # Reset the y-position when a new page is added

    c.save()
    return output_file

def create_zip(file_buffers, file_names):
    """Create a zip file in memory containing SOP PDFs."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for buf, name in zip(file_buffers, file_names):
            buf.seek(0)  # Ensure we're reading from the beginning of the buffer
            zip_file.writestr(name, buf.read())
    zip_buffer.seek(0)
    return zip_buffer

# Streamlit App
st.title("SOP Generator")

# File upload for multiple PDFs
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)


 
      

if uploaded_files:
    st.success(f"{len(uploaded_files)} files uploaded successfully.")
    if st.button("Generate"):

        # Create a list of generated SOP PDFs for each uploaded file
        file_buffers = []
        file_names = []

        # Process each uploaded PDF file
        for uploaded_file in uploaded_files:
            with open("uploaded_file.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Show loading message while processing
            with st.spinner("Please wait, we are generating the content for you..."):
                # Generate embeddings and answer question for SOP content
                create_embeddings_from_pdf("uploaded_file.pdf")
                summary = answer_question_from_vectorstore(
                    "Please provide information for creating a Standard Operating Procedure (SOP) as per provided format"
                )

                # Generate PDF for the SOP content
                pdf_path = generate_pdf_with_reportlab(summary)
                with open(pdf_path, "rb") as pdf_file:
                    file_buffers.append(io.BytesIO(pdf_file.read()))
                    file_names.append(f"SOP_Generated_{uploaded_file.name.split('.')[0]}.pdf")

        # Check if only one file was uploaded
        if len(uploaded_files) == 1:
            # Provide a download link for a single file
            st.download_button(
                label="Download SOP as PDF",
                data=file_buffers[0],
                file_name=file_names[0],
                mime="application/pdf"
            )
        else:
            # After processing all files, create a ZIP and provide a download link
            zip_buffer = create_zip(file_buffers, file_names)
            st.download_button(
                label="Download SOPs as ZIP",
                data=zip_buffer,
                file_name="SOPs_Generated.zip",
                mime="application/zip"
            )

else:
        st.info("Please upload files.")
