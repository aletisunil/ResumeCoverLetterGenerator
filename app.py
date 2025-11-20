import streamlit as st
import os
from typing import TypedDict
from pypdf import PdfReader


from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.graph import StateGraph, END

# ==========================================
# 1. UI CONFIGURATION
# ==========================================
st.set_page_config(page_title="AI Career Agent Team", layout="wide")

st.title("🤖 AI Career Agent Team")
st.markdown("""
This system uses a **Multi-Agent Architecture**:
1. **Evaluator Agent**: Analyzes the gap between your resume and the JD.
2. **Resume Writer**: Rewrites your resume (Slight or Aggressive mode).
3. **Cover Letter Writer**: Writes a targeted letter simultaneously.
""")

# Sidebar for Setup
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("OpenAI API Key", type="password")
    
    st.divider()
    
    modification_level = st.radio(
        "Modification Level",
        ["Slight (Polish & Keywords)", "Aggressive (Rewrite & Restructure)"],
        help="Slight: Fixes grammar and injects keywords. Aggressive: Rewrites bullet points to match JD impact."
    )
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

# ==========================================
# 2. DEFINE STATE & AGENTS
# ==========================================

# Define the shared state object (The "Memory" of the graph)
class AgentState(TypedDict):
    resume_text: str
    job_description: str
    modification_level: str
    analysis: dict          # Output from Evaluator
    tailored_resume: str    # Output from Resume Writer
    cover_letter: str       # Output from Cover Letter Writer

def get_llm():
    """Returns the LLM instance."""
    if not api_key:
        st.error("Please enter an API Key in the sidebar.")
        st.stop()
    return ChatOpenAI(model="gpt-4o", temperature=0.7)

# --- NODE 1: THE EVALUATOR AGENT ---
def evaluator_agent(state: AgentState):
    """Analyzes the gap between Resume and JD."""
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_template("""
    You are an expert Technical Recruiter. 
    Analyze the following Resume against the Job Description.
    
    RESUME:
    {resume_text}
    
    JOB DESCRIPTION:
    {job_description}
    
    Output a JSON object with the following keys:
    - "match_score": (0-100)
    - "missing_keywords": (List of skills in JD but not in Resume)
    - "strengths": (List of matching skills)
    - "gap_analysis": (Brief text explaining what is missing)
    
    Return ONLY valid JSON.
    """)
    
    chain = prompt | llm | JsonOutputParser()
    result = chain.invoke({
        "resume_text": state["resume_text"], 
        "job_description": state["job_description"]
    })
    
    return {"analysis": result}

# --- NODE 2: THE RESUME WRITER AGENT ---
def resume_writer_agent(state: AgentState):
    """Rewrites the resume based on modification level."""
    llm = get_llm()
    
    # Dynamic instructions based on modification level
    if "Slight" in state["modification_level"]:
        instruction = """
        MODE: SLIGHT MODIFICATION.
        1. Keep 95% of the original content and structure exactly as is.
        2. Only fix grammar and seamlessly insert these missing keywords: {missing_keywords}.
        3. Do not change the tone or voice significantly.
        """
    else:
        instruction = """
        MODE: AGGRESSIVE MODIFICATION.
        1. Re-prioritize sections to highlight the specific skills required in the JD.
        2. Use strong action verbs.
        3. If a section is weak/irrelevant to the JD, summarize or shorten it.
        """

    prompt = ChatPromptTemplate.from_template("""
    You are a Professional Resume Writer.
    
    GOAL: Tailor the User's Resume for the provided Job Description.
    
    INSTRUCTIONS:
    {instruction}
    
    FORMATTING RULE (CRITICAL):
    - You MUST output the result in clean MARKDOWN.
    - Analyze the structure of the original resume (Headers, Bullet points, Spacing).
    - MIMIC the original structure as closely as possible. 
    - Do not lose contact info or education details.
    
    Original Resume:
    {resume_text}
    
    Job Description:
    {job_description}
    
    Analysis Data:
    {analysis}
    
    Output the full tailored resume in Markdown:
    """)
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "instruction": instruction,
        "resume_text": state["resume_text"],
        "job_description": state["job_description"],
        "analysis": state["analysis"],
        "missing_keywords": ", ".join(state["analysis"].get("missing_keywords", []))
    })
    
    return {"tailored_resume": result}

# --- NODE 3: THE COVER LETTER AGENT ---
def cover_letter_agent(state: AgentState):
    """Writes a cover letter based on the analysis."""
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_template("""
    You are a persuasive Cover Letter Writer.
    
    Task: Write a cover letter for the User applying to the Job Description.
    
    Guidelines:
    1. Use the "Strengths" identified in the analysis to hook the reader.
    2. Address the "Gap Analysis" by highlighting transferable skills or eagerness to learn.
    3. Keep it professional but engaging.
    4. Use the Candidate's name from the resume (or [Candidate Name] if not found).
    
    Resume Context:
    {resume_text}
    
    Job Description:
    {job_description}
    
    Analysis:
    {analysis}
    
    Output the cover letter in Markdown:
    """)
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "resume_text": state["resume_text"],
        "job_description": state["job_description"],
        "analysis": state["analysis"]
    })
    
    return {"cover_letter": result}

# ==========================================
# 3. BUILD THE GRAPH
# ==========================================

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("evaluator", evaluator_agent)
workflow.add_node("resume_writer", resume_writer_agent)
workflow.add_node("cover_letter_writer", cover_letter_agent)

# Define edges
# 1. Start -> Evaluator
workflow.set_entry_point("evaluator")

# 2. Evaluator -> Parallel Agents (Resume Writer AND Cover Letter Writer)
workflow.add_edge("evaluator", "resume_writer")
workflow.add_edge("evaluator", "cover_letter_writer")

# 3. Writers -> End
workflow.add_edge("resume_writer", END)
workflow.add_edge("cover_letter_writer", END)

app_graph = workflow.compile()

# ==========================================
# 4. UI INTERACTION & EXECUTION
# ==========================================

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload Resume (PDF)")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    resume_text = ""
    if uploaded_file is not None:
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            resume_text += page.extract_text() + "\n"
        st.success("Resume Loaded Successfully!")
        with st.expander("View Extracted Resume Text"):
            st.text(resume_text[:500] + "...")

with col2:
    st.subheader("2. Paste Job Description")
    jd_text = st.text_area("Paste JD Text here", height=200)

# Action Button
if st.button("🚀 Launch Agent Team", type="primary"):
    if not api_key:
        st.error("Please provide an API Key.")
    elif not resume_text or not jd_text:
        st.error("Please upload a resume and provide a job description.")
    else:
        try:
            # Initial State
            inputs = {
                "resume_text": resume_text,
                "job_description": jd_text,
                "modification_level": modification_level,
                "analysis": {},
                "tailored_resume": "",
                "cover_letter": ""
            }
            
            # We'll accumulate results here as they stream in
            final_state = inputs.copy()

            with st.status("🚀 **Starting AI Agents...**", expanded=True) as status:
                
                st.write("🔍 **Evaluator Agent** is analyzing your resume and JD...")
                
                # Stream the graph execution
                for output in app_graph.stream(inputs):
                    for key, value in output.items():
                        # Merge the new data into our final state
                        final_state.update(value)
                        
                        if key == "evaluator":
                            st.write("✅ Gap Analysis complete! Match Score calculated.")
                            st.write("✍️ **Writers** (Resume & Cover Letter) are now working in parallel...")
                        elif key == "resume_writer":
                            st.write("📄 Tailored Resume has been generated.")
                        elif key == "cover_letter_writer":
                            st.write("✉️ Cover Letter has been drafted.")
                
                status.update(label="✅ All Agents Finished Successfully!", state="complete", expanded=False)

            # Display Results
            tab1, tab2, tab3 = st.tabs(["📄 Tailored Resume", "✉️ Cover Letter", "📊 Gap Analysis"])
            
            with tab1:
                st.subheader("Tailored Resume")
                st.markdown(final_state["tailored_resume"])
                st.download_button(
                    label="Download Resume (Markdown)",
                    data=final_state["tailored_resume"],
                    file_name="tailored_resume.md",
                    mime="text/markdown"
                )

            with tab2:
                st.subheader("Generated Cover Letter")
                st.markdown(final_state["cover_letter"])
                st.download_button(
                    label="Download Cover Letter (Markdown)",
                    data=final_state["cover_letter"],
                    file_name="cover_letter.md",
                    mime="text/markdown"
                )
            
            with tab3:
                st.subheader("Agent Analysis")
                analysis = final_state["analysis"]
                st.metric("Match Score", f"{analysis.get('match_score', 0)}/100")
                st.write("**Missing Keywords:**")
                st.write(analysis.get("missing_keywords", []))
                st.write("**Strategic Gap:**")
                st.info(analysis.get("gap_analysis", "N/A"))

        except Exception as e:
            st.error(f"An error occurred: {e}")