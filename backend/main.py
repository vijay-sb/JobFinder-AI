#!/usr/bin/env python3

import os
import json
import ssl
import http.client
import urllib.parse
from typing import List, Dict, Any, Annotated, Sequence
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.schema import Document
from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END, START, MessagesState
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from typing_extensions import TypedDict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Global variables for LLM and vector store
llm = None
vector_store = None
graph = None

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

class JobFinderService:
    def __init__(self):
        self.setup_llm()
        self.setup_vector_store()
        self.setup_graph()
    
    def setup_llm(self):
        global llm
        try:
            # Initialize LLM - make sure to set GOOGLE_API_KEY environment variable
            llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
            print("LLM initialized successfully")
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            raise e
    
    def setup_vector_store(self):
        global vector_store
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = InMemoryVectorStore(embeddings)
            print("Vector store initialized successfully")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise e
    
    def setup_graph(self):
        global graph
        try:
            # Setup agents and graph
            graph = self.create_job_finder_graph()
            print("Graph initialized successfully")
        except Exception as e:
            print(f"Error initializing graph: {e}")
            raise e

    def get_job_descriptions_func(self, query: str, n: int = 10) -> list:
        """
        Fetches top `n` job descriptions using JSearch API and returns them as LangChain Documents.
        """
        try:
            # Create SSL context to handle certificate verification
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            
            conn = http.client.HTTPSConnection("jsearch.p.rapidapi.com", context=context)
            headers = {
                'x-rapidapi-key': os.getenv("RAPIDAPI_KEY"),
                'x-rapidapi-host': "jsearch.p.rapidapi.com"
            }

            if not os.getenv("RAPIDAPI_KEY"):
                return [Document(page_content="Error: RAPIDAPI_KEY not set", metadata={})]

            encoded_query = urllib.parse.quote(query)
            url = f"/search?query={encoded_query}&page=1&num_pages=1&country=us&date_posted=all"

            conn.request("GET", url, headers=headers)
            res = conn.getresponse()
            data = res.read()
            conn.close()

            response_json = json.loads(data.decode("utf-8"))
            results = []

            if 'data' in response_json:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=400,
                    add_start_index=True
                )

                for job in response_json['data'][:n]:
                    title = job.get('job_title', 'N/A')
                    employer = job.get('employer_name', 'N/A')
                    apply_link = job.get('job_apply_link', 'N/A')
                    description = job.get('job_description', 'No description found.')

                    result_text = (
                        f"**Job Title**: {title}\n"
                        f"**Employer**: {employer}\n"
                        f"**Apply Link**: {apply_link}\n\n"
                        f"**Description**:\n{description}"
                    )

                    doc = Document(
                        page_content=result_text,
                        metadata={
                            "source": "jsearch", 
                            "job_title": title, 
                            "employer": employer,
                            "location": query, 
                            "type": "job_description"
                        }
                    )

                    splits = text_splitter.split_documents([doc])
                    results.extend(splits)
                    if vector_store:
                        vector_store.add_documents(splits)

            return results

        except Exception as e:
            print(f"Error in get_job_descriptions: {e}")
            return [Document(page_content=f"Error fetching jobs: {str(e)}", metadata={})]

    def get_resume(self, pdf_path: str):
        """Get resume from PDF file"""
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=50,
                add_start_index=True,
            )
            all_splits = text_splitter.split_documents(docs)
            
            # Add metadata to identify resume content
            for split in all_splits:
                split.metadata["type"] = "resume"
                split.metadata["source"] = "user_resume"
            
            if vector_store:
                vector_store.add_documents(documents=all_splits)
            return all_splits
        except Exception as e:
            print(f"Error loading resume: {e}")
            return [Document(page_content=f"Error loading resume: {str(e)}", metadata={})]

    def get_context_from_vectorstore(self, query: str, k: int = 5):
        """Retrieve relevant context from vector store"""
        try:
            if vector_store:
                docs = vector_store.similarity_search(query, k=k)
                return "\n\n".join([doc.page_content for doc in docs])
            return ""
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return ""

    def create_tools(self):
        """Create tools with proper decorators"""
        
        @tool
        def get_job_descriptions(query: str, n: int = 10) -> str:
            """
            Fetches top `n` job descriptions using JSearch API and returns them as text.
            """
            docs = self.get_job_descriptions_func(query, n)
            return "\n\n".join([doc.page_content for doc in docs])
        
        @tool
        def score_match() -> str:
            """Scores how well the resume matches the job description and provides reasons."""
            # Get context from vector store
            resume_context = self.get_context_from_vectorstore("resume skills experience", k=3)
            job_context = self.get_context_from_vectorstore("job requirements skills", k=3)
            
            prompt = f"""
            You are an expert HR and technical evaluator.
            
            Resume Content:
            {resume_context}
            
            Job Requirements:
            {job_context}
            
            Evaluate how well the candidate's resume matches the job description.
            Give a **final score out of 100** that reflects the overall suitability of the candidate for this job.
            Provide specific reasons for the score based on skills match, experience relevance, and requirements alignment.
            """
            if llm:
                llm_response = llm.invoke(prompt)
                return llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
            return "LLM not available"

        @tool
        def suggest_resume_improvements() -> str:
            """Suggests formatting and content improvements for the resume."""
            resume_context = self.get_context_from_vectorstore("resume content", k=5)
            
            prompt = f"""
            You are a professional resume reviewer.
            
            Resume Content:
            {resume_context}
            
            Please provide a detailed critique that includes the following:
            1. **Formatting Improvements** – layout, section ordering, consistency.
            2. **Grammar and Language** – any grammatical errors, clarity, and tone.
            3. **Bullet Points** – rewrite vague or passive bullets using action verbs and measurable outcomes.
            4. **Missing Sections** – suggest missing but valuable sections (e.g., Certifications, Projects, Publications).
            5. **Overall Suggestions** – anything that would make this resume stand out for competitive roles.
            Format your response in bullet points under each category for easy readability.
            """
            if llm:
                llm_response = llm.invoke(prompt)
                return llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
            return "LLM not available"

        @tool
        def identify_skill_gaps() -> str:
            """Identifies missing skills by comparing resume and job description."""
            resume_context = self.get_context_from_vectorstore("resume skills experience", k=3)
            job_context = self.get_context_from_vectorstore("job requirements skills", k=3)
            
            prompt = f"""
            You are an expert career advisor and skills analyst.
            
            Resume Content:
            {resume_context}
            
            Job Requirements:
            {job_context}
            
            Your task is to identify **missing or underrepresented skills** in the candidate's resume when compared to the job description.
            
            Please list the following:
            1. **Missing Skills** – Key skills mentioned in the job description that are not present in the resume.
            2. **Partially Mentioned Skills** – Skills that are mentioned weakly or lack sufficient detail in the resume.
            3. **Suggestions** – Brief recommendations on how the candidate could gain or demonstrate those skills.
            4. **Course/Certificates** - Mention about courses or certificates which can taken to obtain the required skills
            Format your response clearly under each section.
            """
            if llm:
                llm_response = llm.invoke(prompt)
                return llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
            return "LLM not available"

        @tool
        def generate_resume_feedback() -> str:
            """Generates structured feedback to improve resume alignment with the job description."""
            resume_context = self.get_context_from_vectorstore("resume", k=5)
            job_context = self.get_context_from_vectorstore("job description requirements", k=3)
            
            prompt = f"""
            You are an expert recruiter and resume optimization specialist.
            
            Resume Content:
            {resume_context}
            
            Job Requirements:
            {job_context}
            
            Provide a detailed analysis under the following headings:
            1. **Strengths**: What aspects of the resume align well with the job description?
            2. **Weaknesses**: Where does the resume fall short in terms of required skills, experience, or relevance?
            3. **ATS Optimization Suggestions**: List any formatting, keyword usage, or structural improvements to increase Applicant Tracking System (ATS) compatibility.
            4. **Section Suggestions**: Recommend any missing or underutilized sections (e.g., Certifications, Projects, Skills).
            5. **Overall Alignment Score (0–100)**: Estimate how well the resume aligns with the job.
            Be specific and actionable in your feedback.
            """
            if llm:
                llm_response = llm.invoke(prompt)
                return llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
            return "LLM not available"
        
        return {
            'get_job_descriptions': get_job_descriptions,
            'score_match': score_match,
            'suggest_resume_improvements': suggest_resume_improvements,
            'identify_skill_gaps': identify_skill_gaps,
            'generate_resume_feedback': generate_resume_feedback
        }

    def create_job_finder_graph(self):
        """Create a simplified graph for job analysis"""
        # Create tools
        tools = self.create_tools()
        
        # Create a unified analysis agent
        analysis_agent = create_react_agent(
            model=llm,
            tools=list(tools.values()),
            prompt="""
            You are an expert job-resume matching AI assistant.
            
            Your role is to:
            1. Analyze job descriptions and resume content
            2. Provide match scores and detailed feedback
            3. Identify skill gaps and improvement areas
            4. Give actionable recommendations
            
            When analyzing, use all available tools to provide comprehensive feedback:
            - Use score_match to evaluate overall fit
            - Use identify_skill_gaps to find missing skills
            - Use suggest_resume_improvements for resume enhancement
            - Use generate_resume_feedback for structured analysis
            
            Provide clear, actionable insights that help the candidate improve their job prospects.
            """,
            name="analysis_agent"
        )
        
        # Simple graph with just the analysis agent
        graph_builder = StateGraph(State)
        graph_builder.add_node("analysis_agent", analysis_agent)
        graph_builder.set_entry_point("analysis_agent")
        graph_builder.add_edge("analysis_agent", END)
        
        memory = MemorySaver()
        return graph_builder.compile(checkpointer=memory)

# Initialize service with error handling
try:
    job_finder_service = JobFinderService()
    print("Job finder service initialized successfully")
except Exception as e:
    print(f"Failed to initialize job finder service: {e}")
    job_finder_service = None

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Job Finder API is running", 
        "endpoints": [
            "/api/health",
            "/api/search-jobs",
            "/api/upload-resume", 
            "/api/analyze",
            "/api/generate-cover-letter"
        ]
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "service_initialized": job_finder_service is not None,
        "endpoints_available": True
    })

@app.route('/api/search-jobs', methods=['POST'])
def search_jobs():
    try:
        if not job_finder_service:
            return jsonify({"error": "Service not initialized"}), 500
            
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        query = data.get('query', '')
        num_jobs = data.get('num_jobs', 10)
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        results = job_finder_service.get_job_descriptions_func(query, num_jobs)
        job_data = []
        
        for doc in results:
            content = doc.page_content
            metadata = doc.metadata
            job_data.append({
                "content": content,
                "metadata": metadata
            })
        
        return jsonify({"jobs": job_data, "count": len(job_data)})
    
    except Exception as e:
        print(f"Error in search_jobs: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload-resume', methods=['POST'])
def upload_resume():
    try:
        if not job_finder_service:
            return jsonify({"error": "Service not initialized"}), 500
            
        if 'resume' not in request.files:
            return jsonify({"error": "No resume file provided"}), 400
        
        file = request.files['resume']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if file and file.filename.lower().endswith('.pdf'):
            # Create temp directory if it doesn't exist
            os.makedirs("/tmp", exist_ok=True)
            
            # Save file temporarily
            temp_path = f"/tmp/{file.filename}"
            file.save(temp_path)
            
            # Process resume
            resume_docs = job_finder_service.get_resume(temp_path)
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            resume_content = []
            for doc in resume_docs:
                resume_content.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            
            return jsonify({
                "resume": resume_content, 
                "message": "Resume uploaded successfully",
                "count": len(resume_content)
            })
        
        return jsonify({"error": "Only PDF files are allowed"}), 400
    
    except Exception as e:
        print(f"Error in upload_resume: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_resume_jobs():
    try:
        if not job_finder_service or not job_finder_service.graph:
            return jsonify({"error": "Service or graph not initialized"}), 500
            
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        query = data.get('query', '')
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        config = {"configurable": {"thread_id": "analysis_session"}}
        
        # Create a human message for the graph
        human_message = HumanMessage(content=query)
        
        # Run the graph
        result = job_finder_service.graph.invoke(
            {"messages": [human_message]},
            config=config
        )
        
        # Extract the final result
        if result and "messages" in result:
            final_message = result["messages"][-1]
            content = final_message.content if hasattr(final_message, 'content') else str(final_message)
            return jsonify({
                "analysis": content,
                "status": "success"
            })
        
        return jsonify({"error": "No analysis result generated"}), 500
    
    except Exception as e:
        print(f"Error in analyze_resume_jobs: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@app.route('/api/generate-cover-letter', methods=['POST'])
def generate_cover_letter():
    try:
        if not job_finder_service:
            return jsonify({"error": "Service not initialized"}), 500
            
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        job_title = data.get('job_title', '')
        company = data.get('company', '')
        
        if not job_title or not company:
            return jsonify({"error": "job_title and company are required"}), 400
        
        # Get relevant context from vector store
        resume_context = job_finder_service.get_context_from_vectorstore("resume skills experience", k=3)
        job_context = job_finder_service.get_context_from_vectorstore(f"{job_title} {company}", k=2)
        
        prompt = f"""
        Generate a professional cover letter for the position of {job_title} at {company}.
        
        Resume Information:
        {resume_context}
        
        Job Information:
        {job_context}
        
        Create a compelling cover letter that:
        1. Shows enthusiasm for the specific role and company
        2. Highlights relevant skills and experiences from the resume
        3. Demonstrates understanding of the job requirements
        4. Is professional, concise, and engaging
        5. Follows standard cover letter format
        
        Make it personalized and specific to this opportunity.
        """
        
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        return jsonify({
            "cover_letter": content,
            "job_title": job_title,
            "company": company
        })
    
    except Exception as e:
        print(f"Error in generate_cover_letter: {e}")
        return jsonify({"error": str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Not Found",
        "message": "The requested endpoint does not exist",
        "available_endpoints": [
            "/",
            "/api/health",
            "/api/search-jobs",
            "/api/upload-resume", 
            "/api/analyze",
            "/api/generate-cover-letter"
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal Server Error",
        "message": "An unexpected error occurred"
    }), 500

if __name__ == '__main__':
    # Verify environment variables
    required_env_vars = ['GOOGLE_API_KEY', 'RAPIDAPI_KEY']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Warning: Missing environment variables: {missing_vars}")
        print("Please set these variables in your .env file or environment")
    
    print("Starting Flask application...")
    print("Available endpoints:")
    print("- GET  /")
    print("- GET  /api/health")
    print("- POST /api/search-jobs")
    print("- POST /api/upload-resume")
    print("- POST /api/analyze")
    print("- POST /api/generate-cover-letter")
    
    app.run(debug=True, host='0.0.0.0', port=5001)