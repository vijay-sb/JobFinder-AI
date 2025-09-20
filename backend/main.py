#!/usr/bin/env python3

import os
import json
import ssl
import http.client
import urllib.parse
import re
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Annotated, Sequence
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.tools import tool
from langchain.schema import Document
from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from typing_extensions import TypedDict
from dotenv import load_dotenv
import time
from urllib.parse import urljoin, urlparse
import traceback
import logging
import uuid
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)  # Changed from DEBUG to INFO
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

def clean_markdown_formatting(text: str) -> str:
    """Remove markdown formatting and clean up text for better readability"""
    if not text:
        return ""
    
    # Remove markdown headers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # Remove bold/italic formatting but keep the text
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    
    # Clean up bullet points
    text = re.sub(r'^\s*[-\*\+]\s+', '• ', text, flags=re.MULTILINE)
    
    # Clean up numbered lists
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # Remove extra whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Clean up any remaining markdown artifacts
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    return text.strip()

def format_for_display(text: str) -> str:
    """Format text for clean display with proper structure"""
    if not text:
        return ""
    
    # First clean markdown
    text = clean_markdown_formatting(text)
    
    # Add proper section breaks
    text = re.sub(r'(Current Strengths:|Skills Development|Technical Skills|Certification Recommendations|Resume Content|Learning Path|Action Items|Overview|Key Areas|Preparation Strategy|Resources for Practice)', r'\n\n\1', text)
    
    # Ensure proper line breaks before lists
    text = re.sub(r'([.:])\s*\n([•])', r'\1\n\n\2', text)
    
    return text.strip()

class MockInterviewSession:
    """Class to manage mock interview sessions"""
    def __init__(self, session_id: str, job_title: str, company: str, difficulty: str, resume_context: str):
        self.session_id = session_id
        self.job_title = job_title
        self.company = company
        self.difficulty = difficulty
        self.resume_context = resume_context
        self.questions_asked = []
        self.answers_given = []
        self.feedback_history = []
        self.current_question_index = 0
        self.created_at = datetime.now()
        self.status = "active"  # active, paused, completed
        
    def to_dict(self):
        return {
            "session_id": self.session_id,
            "job_title": self.job_title,
            "company": self.company,
            "difficulty": self.difficulty,
            "questions_asked": self.questions_asked,
            "answers_given": self.answers_given,
            "feedback_history": self.feedback_history,
            "current_question_index": self.current_question_index,
            "created_at": self.created_at.isoformat(),
            "status": self.status,
            "total_questions": len(self.questions_asked)
        }

class JobFinderService:
    def __init__(self):
        self.llm = None
        self.vector_store = None
        self.initialization_errors = []
        self.mock_sessions = {}  # Store active mock interview sessions
        
        try:
            self.setup_google_ai()
        except Exception as e:
            self.initialization_errors.append(f"Google AI setup failed: {str(e)}")
            logger.error(f"Google AI setup failed: {e}")
        
        try:
            self.setup_vector_store()
        except Exception as e:
            self.initialization_errors.append(f"Vector store setup failed: {str(e)}")
            logger.error(f"Vector store setup failed: {e}")
    
    def setup_google_ai(self):
        """Setup Google Cloud AI tools"""
        try:
            # Check if API key is available
            if not os.getenv("GOOGLE_API_KEY"):
                raise ValueError("GOOGLE_API_KEY environment variable not set")
            
            # Initialize Google's Gemini model directly
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.7
            )
            print("Google AI (Gemini) initialized successfully")
        except Exception as e:
            print(f"Error initializing Google AI: {e}")
            raise e
    
    def setup_vector_store(self):
        """Setup vector store with Google embeddings"""
        try:
            # Try Google embeddings first, fallback to HuggingFace
            try:
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/text-embedding-004",
                    google_api_key=os.getenv("GOOGLE_API_KEY")
                )
                print("Using Google embeddings")
            except Exception as e:
                print(f"Google embeddings failed: {e}")
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                print("Using HuggingFace embeddings as fallback")
            
            self.vector_store = InMemoryVectorStore(embeddings)
            print("Vector store initialized successfully")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise e

    def is_healthy(self):
        """Check if the service is properly initialized"""
        return (self.llm is not None and 
                self.vector_store is not None and 
                len(self.initialization_errors) == 0)

    def get_job_descriptions_func(self, query: str, n: int = 10) -> list:
        """Fetches job descriptions and extracts location information"""
        try:
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
                    
                    # Extract location information
                    location = job.get('job_city', '') + ', ' + job.get('job_state', '') + ', ' + job.get('job_country', '')
                    location = location.strip(', ')
                    
                    # Create Google Maps link
                    maps_link = f"https://www.google.com/maps/search/?api=1&query={urllib.parse.quote(employer + ' ' + location)}" if location else "N/A"

                    result_text = f"""Job Title: {title}
Employer: {employer}
Location: {location}
Apply Link: {apply_link}
Google Maps: {maps_link}

Description:
{description}"""

                    doc = Document(
                        page_content=result_text,
                        metadata={
                            "source": "jsearch", 
                            "job_title": title, 
                            "employer": employer,
                            "location": location,
                            "maps_link": maps_link,
                            "apply_link": apply_link,
                            "type": "job_description"
                        }
                    )

                    splits = text_splitter.split_documents([doc])
                    results.extend(splits)
                    if self.vector_store:
                        self.vector_store.add_documents(splits)

            return results

        except Exception as e:
            logger.error(f"Error in get_job_descriptions: {e}")
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
            
            for split in all_splits:
                split.metadata["type"] = "resume"
                split.metadata["source"] = "user_resume"
            
            if self.vector_store:
                self.vector_store.add_documents(documents=all_splits)
            return all_splits
        except Exception as e:
            logger.error(f"Error loading resume: {e}")
            return [Document(page_content=f"Error loading resume: {str(e)}", metadata={})]

    def get_context_from_vectorstore(self, query: str, k: int = 5):
        """Retrieve relevant context from vector store"""
        try:
            if self.vector_store:
                docs = self.vector_store.similarity_search(query, k=k)
                return "\n\n".join([doc.page_content for doc in docs])
            return ""
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return ""

    def analyze_resume_with_google_ai(self, query: str) -> str:
        """Analyze resume using Google AI"""
        try:
            resume_context = self.get_context_from_vectorstore("resume", k=10)
            
            if not resume_context:
                return "No resume content found. Please upload your resume first."
            
            prompt = f"""Analyze this resume and provide insights:

Resume Content:
{resume_context}

User Query: {query}

Please provide a clear, well-structured analysis including:

1. CURRENT STRENGTHS
   List the candidate's key skills and experiences

2. SKILLS ASSESSMENT
   Technical Skills: Evaluate current technical capabilities
   Soft Skills: Assess communication, leadership, etc.
   Experience Level: Overall assessment

3. IMPROVEMENT RECOMMENDATIONS
   Provide 3-5 specific, actionable recommendations

4. CAREER DEVELOPMENT SUGGESTIONS
   Skills to develop
   Experience to gain
   Learning resources

Format your response in clean, readable text without markdown formatting.
Use clear section headings and bullet points for easy reading."""
            
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            return format_for_display(content)
            
        except Exception as e:
            logger.error(f"Error in Google AI analysis: {e}")
            return f"Analysis error: {str(e)}"

    def get_resume_improvement_advice(self) -> str:
        """Get resume improvement advice using Google AI"""
        try:
            resume_context = self.get_context_from_vectorstore("resume skills experience education", k=10)
            
            if not resume_context:
                return "No resume content found. Please upload your resume first."
            
            prompt = f"""Analyze this resume and provide comprehensive improvement advice:

Resume Content:
{resume_context}

Please provide detailed recommendations in the following areas:

1. CURRENT STRENGTHS
   Identify 3-5 key strengths in the resume

2. SKILLS DEVELOPMENT RECOMMENDATIONS
   
   Technical Skills to Develop:
   - List specific technical skills with explanations
   - Suggest learning resources (Coursera, Udemy, etc.)
   - Provide realistic timelines
   
   Soft Skills to Enhance:
   - Communication, leadership, teamwork skills
   - How to develop them

3. CERTIFICATION RECOMMENDATIONS
   Suggest relevant certifications with providers and reasons

4. RESUME CONTENT IMPROVEMENTS
   Formatting suggestions
   Keywords to include
   Experience descriptions improvements

5. LEARNING PATH (Next 6 months)
   Month 1-2: Priority skills
   Month 3-4: Next phase
   Month 5-6: Advanced skills

6. RECOMMENDED LEARNING PLATFORMS
   Specific platforms and what to learn on each

7. ACTION ITEMS (Priority Order)
   Top 5 most important actions to take

Format your response in clean, readable text without any markdown symbols.
Use clear headings and organize information logically."""
            
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            return format_for_display(content)
            
        except Exception as e:
            logger.error(f"Error generating resume advice: {e}")
            return f"Error generating advice: {str(e)}"

    def scrape_interview_resources(self, job_title: str) -> Dict[str, List[str]]:
        """Scrape interview resources and find YouTube links"""
        resources = {
            "websites": [],
            "youtube_links": [],
            "content_summary": ""
        }
        
        try:
            # Define target websites for different job types
            prep_sites = {
                "data scientist": [
                    "https://www.geeksforgeeks.org/data-science-interview-questions/",
                    "https://www.interviewbit.com/data-science-interview-questions/"
                ],
                "software engineer": [
                    "https://www.geeksforgeeks.org/software-engineering-interview-questions/",
                    "https://www.interviewbit.com/software-engineer-interview-questions/"
                ],
                "machine learning": [
                    "https://www.geeksforgeeks.org/machine-learning-interview-questions/",
                    "https://www.interviewbit.com/machine-learning-interview-questions/"
                ],
                "python developer": [
                    "https://www.geeksforgeeks.org/python-interview-questions/",
                    "https://www.interviewbit.com/python-interview-questions/"
                ]
            }
            
            # YouTube search queries for different roles
            youtube_searches = {
                "data scientist": [
                    "data scientist interview questions",
                    "machine learning interview preparation",
                    "statistics interview questions"
                ],
                "software engineer": [
                    "software engineer interview questions",
                    "coding interview preparation",
                    "system design interview"
                ],
                "machine learning": [
                    "machine learning interview questions",
                    "deep learning interview prep",
                    "ML algorithms interview"
                ],
                "python developer": [
                    "python interview questions",
                    "python coding interview",
                    "django flask interview"
                ]
            }
            
            # Find relevant sites
            job_lower = job_title.lower()
            relevant_sites = []
            youtube_queries = []
            
            for key, sites in prep_sites.items():
                if key in job_lower:
                    relevant_sites.extend(sites)
                    youtube_queries.extend(youtube_searches.get(key, []))
            
            if not relevant_sites:
                relevant_sites = prep_sites["software engineer"]
                youtube_queries = youtube_searches["software engineer"]
            
            # Add sites to resources
            resources["websites"] = relevant_sites
            
            # Generate YouTube links (these are search URLs that will show relevant videos)
            for query in youtube_queries[:3]:  # Limit to 3 searches
                youtube_url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}"
                resources["youtube_links"].append({
                    "title": f"YouTube: {query.title()}",
                    "url": youtube_url
                })
            
            # Try to scrape some content for summary
            scraped_content = []
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            for site in relevant_sites[:1]:  # Just scrape first site for summary
                try:
                    time.sleep(1)
                    response = requests.get(site, headers=headers, timeout=10)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        paragraphs = soup.find_all('p')
                        text_content = ' '.join([p.get_text(strip=True) for p in paragraphs[:10]])
                        if text_content:
                            scraped_content.append(text_content[:500])
                except Exception as e:
                    logger.error(f"Error scraping {site}: {e}")
                    continue
            
            resources["content_summary"] = ' '.join(scraped_content)
            
        except Exception as e:
            logger.error(f"Error scraping resources: {e}")
        
        return resources

    def get_interview_preparation_guide(self, job_title: str, company: str = "") -> str:
        """Generate comprehensive interview preparation guide"""
        try:
            resume_context = self.get_context_from_vectorstore("resume skills experience", k=8)
            
            if not resume_context:
                return "No resume content found for personalized preparation."
            
            # Get interview resources
            resources = self.scrape_interview_resources(job_title)
            
            prompt = f"""Create a comprehensive interview preparation guide for a {job_title} position{' at ' + company if company else ''}.

Candidate's Background:
{resume_context}

Available Resources Summary:
{resources.get('content_summary', 'General interview preparation resources available')}

Please create a detailed preparation guide with:

1. OVERVIEW
   What to expect in this type of interview

2. KEY AREAS TO FOCUS ON
   
   Technical Skills Assessment:
   Based on the candidate's resume, list specific skills they should be ready to discuss
   Include sample questions for each skill

3. COMMON INTERVIEW QUESTIONS
   
   Technical Questions:
   5-7 specific technical questions for this role
   
   Behavioral Questions:
   5 behavioral questions with STAR method examples

4. PREPARATION STRATEGY
   
   1 Week Before Interview:
   Specific tasks and study areas
   
   2-3 Days Before:
   Final preparation activities
   
   Day Before:
   Last-minute preparation items

5. PRACTICE RESOURCES
   Coding Practice: Specific platforms and problem types
   System Design: If applicable to role
   Domain Knowledge: Key topics to review

6. QUESTIONS TO ASK THE INTERVIEWER
   5-7 thoughtful questions about role, team, and company

7. RED FLAGS TO AVOID
   Common mistakes for this type of role

8. FINAL TIPS
   Personalized advice based on candidate's background

9. HELPFUL RESOURCES
   Mention that additional resources are available

Format in clean, readable text without markdown. Use clear section headings and organize logically."""
            
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Add resource links to the content
            formatted_content = format_for_display(content)
            
            # Append resource links
            resource_section = "\n\nADDITIONAL RESOURCES:\n\n"
            resource_section += "Recommended Websites:\n"
            for site in resources["websites"]:
                resource_section += f"• {site}\n"
            
            resource_section += "\nYouTube Study Resources:\n"
            for yt_resource in resources["youtube_links"]:
                resource_section += f"• {yt_resource['title']}: {yt_resource['url']}\n"
            
            return formatted_content + resource_section
            
        except Exception as e:
            logger.error(f"Error generating interview prep: {e}")
            return f"Error generating interview preparation: {str(e)}"

    # Mock Interview Simulator Methods
    def create_mock_interview_session(self, job_title: str, company: str = "", difficulty: str = "intermediate") -> MockInterviewSession:
        """Create a new mock interview session"""
        try:
            session_id = str(uuid.uuid4())
            resume_context = self.get_context_from_vectorstore("resume skills experience", k=8)
            
            if not resume_context:
                raise ValueError("No resume content found. Please upload your resume first.")
            
            session = MockInterviewSession(
                session_id=session_id,
                job_title=job_title,
                company=company,
                difficulty=difficulty,
                resume_context=resume_context
            )
            
            # Generate initial questions for the session
            initial_questions = self.generate_interview_questions(session)
            session.questions_asked = initial_questions
            
            # Store the session
            self.mock_sessions[session_id] = session
            
            return session
            
        except Exception as e:
            logger.error(f"Error creating mock interview session: {e}")
            raise e

    def generate_interview_questions(self, session: MockInterviewSession) -> List[Dict[str, Any]]:
        """Generate interview questions based on job role and resume"""
        try:
            prompt = f"""Generate a set of interview questions for a {session.job_title} position{' at ' + session.company if session.company else ''}.

Candidate's Background:
{session.resume_context}

Difficulty Level: {session.difficulty}

Please generate exactly 8 interview questions in the following categories:

1. WARM-UP QUESTIONS (2 questions)
   - Easy introductory questions to start the interview

2. TECHNICAL QUESTIONS (3 questions)
   - Based on the candidate's skills and the job requirements
   - Adjust complexity based on difficulty level

3. BEHAVIORAL QUESTIONS (2 questions)
   - STAR method applicable questions
   - Leadership, teamwork, problem-solving scenarios

4. SITUATIONAL QUESTIONS (1 question)
   - Role-specific scenarios they might face

For each question, provide the response in this exact format:

QUESTION 1
Category: warm-up
Difficulty: easy
Question: Tell me about yourself and your background
Key Points: Should cover relevant experience, skills, and career goals

QUESTION 2
Category: warm-up  
Difficulty: easy
Question: What interests you about this role?
Key Points: Should show research about company and role alignment

Continue this format for all 8 questions..."""

            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse the generated questions
            questions = self.parse_generated_questions(content)
            
            # If parsing fails, create default questions
            if len(questions) == 0:
                questions = self.create_default_questions(session.job_title, session.difficulty)
            
            return questions
            
        except Exception as e:
            logger.error(f"Error generating interview questions: {e}")
            # Return default questions as fallback
            return self.create_default_questions(session.job_title, session.difficulty)

    def create_default_questions(self, job_title: str, difficulty: str) -> List[Dict[str, Any]]:
        """Create default questions if AI generation fails"""
        default_questions = [
            {
                "id": 1,
                "category": "warm-up",
                "difficulty": "easy",
                "question": "Tell me about yourself and your background.",
                "key_points": "Should cover relevant experience, skills, and career goals"
            },
            {
                "id": 2,
                "category": "warm-up",
                "difficulty": "easy", 
                "question": f"What interests you about this {job_title} position?",
                "key_points": "Should show research about the role and company alignment"
            },
            {
                "id": 3,
                "category": "technical",
                "difficulty": difficulty,
                "question": f"Describe your experience with the key technologies used in {job_title} roles.",
                "key_points": "Should demonstrate technical knowledge and hands-on experience"
            },
            {
                "id": 4,
                "category": "technical",
                "difficulty": difficulty,
                "question": "Walk me through how you would approach solving a complex technical problem.",
                "key_points": "Should show problem-solving methodology and technical thinking"
            },
            {
                "id": 5,
                "category": "technical",
                "difficulty": difficulty,
                "question": f"What are the current trends and challenges in {job_title}?",
                "key_points": "Should demonstrate industry knowledge and awareness"
            },
            {
                "id": 6,
                "category": "behavioral",
                "difficulty": "medium",
                "question": "Tell me about a time when you had to work with a difficult team member.",
                "key_points": "Should use STAR method and show conflict resolution skills"
            },
            {
                "id": 7,
                "category": "behavioral",
                "difficulty": "medium",
                "question": "Describe a project where you had to learn something new quickly.",
                "key_points": "Should demonstrate learning ability and adaptability"
            },
            {
                "id": 8,
                "category": "situational",
                "difficulty": difficulty,
                "question": f"How would you handle a situation where you disagreed with your manager's technical decision in a {job_title} project?",
                "key_points": "Should show professional communication and problem-solving"
            }
        ]
        
        return default_questions

    def parse_generated_questions(self, content: str) -> List[Dict[str, Any]]:
        """Parse AI-generated questions into structured format"""
        questions = []
        
        try:
            # Split by QUESTION markers
            sections = re.split(r'QUESTION\s+\d+', content)
            
            for i, section in enumerate(sections[1:], 1):  # Skip first empty section
                lines = [line.strip() for line in section.split('\n') if line.strip()]
                
                question_data = {
                    "id": i,
                    "category": "general",
                    "difficulty": "medium", 
                    "question": "",
                    "key_points": ""
                }
                
                for line in lines:
                    if line.lower().startswith('category:'):
                        question_data['category'] = line.split(':', 1)[1].strip().lower()
                    elif line.lower().startswith('difficulty:'):
                        question_data['difficulty'] = line.split(':', 1)[1].strip().lower()
                    elif line.lower().startswith('question:'):
                        question_data['question'] = line.split(':', 1)[1].strip()
                    elif line.lower().startswith('key points:'):
                        question_data['key_points'] = line.split(':', 1)[1].strip()
                
                # Only add if we have a valid question
                if question_data['question']:
                    questions.append(question_data)
                    
            logger.info(f"Successfully parsed {len(questions)} questions")
            return questions
            
        except Exception as e:
            logger.error(f"Error parsing questions: {e}")
            return []

    def get_next_question(self, session_id: str) -> Dict[str, Any]:
        """Get the next question for the interview"""
        try:
            if session_id not in self.mock_sessions:
                raise ValueError("Session not found")
                
            session = self.mock_sessions[session_id]
            
            if session.current_question_index >= len(session.questions_asked):
                return {
                    "question": None,
                    "message": "Interview completed! Great job!",
                    "is_complete": True,
                    "session_summary": self.generate_session_summary(session)
                }
            
            current_question = session.questions_asked[session.current_question_index]
            
            return {
                "question": current_question,
                "question_number": session.current_question_index + 1,
                "total_questions": len(session.questions_asked),
                "is_complete": False
            }
            
        except Exception as e:
            logger.error(f"Error getting next question: {e}")
            return {"error": str(e)}

    def submit_answer_and_get_feedback(self, session_id: str, answer: str) -> Dict[str, Any]:
        """Submit an answer and get AI feedback"""
        try:
            if session_id not in self.mock_sessions:
                raise ValueError("Session not found")
                
            session = self.mock_sessions[session_id]
            
            if session.current_question_index >= len(session.questions_asked):
                raise ValueError("No active question to answer")
                
            current_question = session.questions_asked[session.current_question_index]
            
            # Store the answer
            session.answers_given.append({
                "question_id": current_question.get('id', session.current_question_index + 1),
                "question": current_question.get('question', ''),
                "answer": answer,
                "timestamp": datetime.now().isoformat()
            })
            
            # Generate feedback
            feedback = self.generate_answer_feedback(current_question, answer, session)
            session.feedback_history.append(feedback)
            
            # Move to next question
            session.current_question_index += 1
            
            # Get next question info
            next_question_info = self.get_next_question(session_id)
            
            return {
                "feedback": feedback,
                "next_question_info": next_question_info,
                "progress": {
                    "current": session.current_question_index,
                    "total": len(session.questions_asked),
                    "percentage": min(100, (session.current_question_index / len(session.questions_asked)) * 100)
                }
            }
            
        except Exception as e:
            logger.error(f"Error submitting answer: {e}")
            return {"error": str(e)}

    def generate_answer_feedback(self, question: Dict[str, Any], answer: str, session: MockInterviewSession) -> Dict[str, Any]:
        """Generate AI feedback for an answer"""
        try:
            prompt = f"""Evaluate this interview answer and provide constructive feedback:

Job Role: {session.job_title}
Question Category: {question.get('category', 'general')}
Question: {question.get('question', '')}
Expected Key Points: {question.get('key_points', '')}

Candidate's Answer: {answer}

Candidate's Background:
{session.resume_context[:500]}...

Please provide feedback in the following format:

STRENGTHS:
- What the candidate did well in their answer

AREAS FOR IMPROVEMENT:
- What could be enhanced or added

SCORE: X/10 (where 10 is excellent)

SUGGESTIONS:
- Specific tips for improving this type of answer

FOLLOW-UP TIPS:
- What to focus on for similar questions

Keep feedback constructive, specific, and encouraging. Focus on actionable improvements."""

            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse feedback to extract score
            score_match = re.search(r'SCORE:\s*(\d+(?:\.\d+)?)', content)
            score = float(score_match.group(1)) if score_match else 5.0
            
            return {
                "question_id": question.get('id', 0),
                "question": question.get('question', ''),
                "answer": answer,
                "feedback_text": format_for_display(content),
                "score": score,
                "category": question.get('category', 'general'),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating feedback: {e}")
            return {
                "question_id": question.get('id', 0),
                "feedback_text": f"Error generating feedback: {str(e)}",
                "score": 0.0,
                "timestamp": datetime.now().isoformat()
            }

    def generate_session_summary(self, session: MockInterviewSession) -> Dict[str, Any]:
        """Generate a comprehensive session summary"""
        try:
            # Calculate overall score
            scores = [feedback.get('score', 0) for feedback in session.feedback_history if 'score' in feedback]
            overall_score = sum(scores) / len(scores) if scores else 0
            
            # Categorize performance by question type
            category_scores = {}
            for feedback in session.feedback_history:
                category = feedback.get('category', 'general')
                score = feedback.get('score', 0)
                if category not in category_scores:
                    category_scores[category] = []
                category_scores[category].append(score)
            
            # Calculate average scores per category
            category_averages = {
                category: sum(scores) / len(scores) 
                for category, scores in category_scores.items()
            }
            
            # Generate AI summary
            prompt = f"""Generate a comprehensive interview performance summary:

Job Role: {session.job_title}
Company: {session.company}
Total Questions: {len(session.questions_asked)}
Overall Score: {overall_score:.1f}/10

Category Performance:
{chr(10).join([f"- {category}: {avg:.1f}/10" for category, avg in category_averages.items()])}

Questions and Feedback:
{chr(10).join([f"Q: {feedback.get('question', '')[:100]}... Score: {feedback.get('score', 0)}/10" for feedback in session.feedback_history[:5]])}

Please provide:

1. OVERALL PERFORMANCE ASSESSMENT
   Brief summary of how the candidate performed

2. STRENGTHS IDENTIFIED
   Top 3-4 strengths demonstrated during the interview

3. AREAS FOR IMPROVEMENT
   Key areas that need development

4. CATEGORY ANALYSIS
   Performance breakdown by question type

5. NEXT STEPS
   Specific recommendations for interview preparation

6. CONFIDENCE BUILDING TIPS
   Encouragement and motivation for future interviews

Format in clean, readable text without markdown."""

            response = self.llm.invoke(prompt)
            summary_text = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "session_id": session.session_id,
                "overall_score": round(overall_score, 1),
                "category_scores": {k: round(v, 1) for k, v in category_averages.items()},
                "total_questions": len(session.questions_asked),
                "completion_rate": 100,
                "summary_text": format_for_display(summary_text),
                "detailed_feedback": session.feedback_history,
                "session_duration": (datetime.now() - session.created_at).total_seconds() / 60,  # minutes
                "recommendations": self.generate_improvement_recommendations(session, category_averages)
            }
            
        except Exception as e:
            logger.error(f"Error generating session summary: {e}")
            return {
                "error": f"Error generating summary: {str(e)}",
                "overall_score": 0,
                "total_questions": len(session.questions_asked)
            }

    def generate_improvement_recommendations(self, session: MockInterviewSession, category_scores: Dict[str, float]) -> List[str]:
        """Generate specific improvement recommendations"""
        recommendations = []
        
        # Identify weak areas
        weak_categories = [cat for cat, score in category_scores.items() if score < 6.0]
        
        for category in weak_categories:
            if category == 'technical':
                recommendations.append(f"Practice more technical questions related to {session.job_title}")
                recommendations.append("Review fundamental concepts and practice coding problems")
            elif category == 'behavioral':
                recommendations.append("Practice STAR method for behavioral questions")
                recommendations.append("Prepare specific examples from your experience")
            elif category == 'situational':
                recommendations.append("Research common workplace scenarios for your role")
                recommendations.append("Think through problem-solving approaches")
        
        if not recommendations:
            recommendations.append("Continue practicing to maintain your strong performance")
            recommendations.append("Focus on advanced topics to stand out from other candidates")
        
        return recommendations[:5]  # Limit to top 5 recommendations

    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get current status of a mock interview session"""
        try:
            if session_id not in self.mock_sessions:
                return {"error": "Session not found"}
                
            session = self.mock_sessions[session_id]
            return session.to_dict()
            
        except Exception as e:
            logger.error(f"Error getting session status: {e}")
            return {"error": str(e)}

    def pause_session(self, session_id: str) -> Dict[str, Any]:
        """Pause a mock interview session"""
        try:
            if session_id not in self.mock_sessions:
                return {"error": "Session not found"}
                
            session = self.mock_sessions[session_id]
            session.status = "paused"
            
            return {"message": "Session paused successfully", "session_id": session_id}
            
        except Exception as e:
            logger.error(f"Error pausing session: {e}")
            return {"error": str(e)}

    def resume_session(self, session_id: str) -> Dict[str, Any]:
        """Resume a paused mock interview session"""
        try:
            if session_id not in self.mock_sessions:
                return {"error": "Session not found"}
                
            session = self.mock_sessions[session_id]
            session.status = "active"
            
            return {"message": "Session resumed successfully", "session_id": session_id}
            
        except Exception as e:
            logger.error(f"Error resuming session: {e}")
            return {"error": str(e)}

    def end_session(self, session_id: str) -> Dict[str, Any]:
        """End a mock interview session and generate final summary"""
        try:
            if session_id not in self.mock_sessions:
                return {"error": "Session not found"}
                
            session = self.mock_sessions[session_id]
            session.status = "completed"
            
            # Generate final summary
            summary = self.generate_session_summary(session)
            
            return {
                "message": "Session completed successfully",
                "session_id": session_id,
                "final_summary": summary
            }
            
        except Exception as e:
            logger.error(f"Error ending session: {e}")
            return {"error": str(e)}

# Initialize service
try:
    job_finder_service = JobFinderService()
    print("Job finder service initialization completed")
    if job_finder_service.initialization_errors:
        print("Initialization errors:")
        for error in job_finder_service.initialization_errors:
            print(f"  - {error}")
except Exception as e:
    print(f"Failed to initialize job finder service: {e}")
    job_finder_service = None

@app.route('/', methods=['GET'])
def home():
    service_status = "initialized" if job_finder_service and job_finder_service.is_healthy() else "initialization_errors"
    return jsonify({
        "message": "Enhanced Job Finder API with Google Cloud AI & Mock Interview Simulator", 
        "service_status": service_status,
        "ai_provider": "Google Cloud AI (Gemini)",
        "initialization_errors": job_finder_service.initialization_errors if job_finder_service else [],
        "endpoints": [
            "/api/health",
            "/api/search-jobs",
            "/api/upload-resume", 
            "/api/analyze",
            "/api/resume-advice",
            "/api/interview-prep",
            "/api/generate-cover-letter",
            "/api/mock-interview/start",
            "/api/mock-interview/next-question",
            "/api/mock-interview/submit-answer",
            "/api/mock-interview/session-status",
            "/api/mock-interview/pause",
            "/api/mock-interview/resume",
            "/api/mock-interview/end"
        ]
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    if job_finder_service:
        return jsonify({
            "status": "healthy" if job_finder_service.is_healthy() else "degraded",
            "service_initialized": True,
            "ai_provider": "Google Cloud AI (Gemini)",
            "llm_available": job_finder_service.llm is not None,
            "vector_store_available": job_finder_service.vector_store is not None,
            "mock_sessions_active": len(job_finder_service.mock_sessions),
            "initialization_errors": job_finder_service.initialization_errors
        })
    else:
        return jsonify({
            "status": "unhealthy",
            "service_initialized": False,
            "error": "Service failed to initialize"
        }), 500

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
            content = clean_markdown_formatting(doc.page_content)  # Clean the content
            metadata = doc.metadata
            job_data.append({
                "content": content,
                "metadata": metadata
            })
        
        return jsonify({"jobs": job_data, "count": len(job_data)})
    
    except Exception as e:
        logger.error(f"Error in search_jobs: {e}")
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
            os.makedirs("/tmp", exist_ok=True)
            temp_path = f"/tmp/{file.filename}"
            file.save(temp_path)
            
            resume_docs = job_finder_service.get_resume(temp_path)
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            resume_content = []
            for doc in resume_docs:
                content = clean_markdown_formatting(doc.page_content)  # Clean the content
                resume_content.append({
                    "content": content,
                    "metadata": doc.metadata
                })
            
            return jsonify({
                "resume": resume_content, 
                "message": "Resume uploaded successfully",
                "count": len(resume_content)
            })
        
        return jsonify({"error": "Only PDF files are allowed"}), 400
    
    except Exception as e:
        logger.error(f"Error in upload_resume: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_resume_jobs():
    try:
        if not job_finder_service:
            return jsonify({"error": "Service not initialized"}), 500
            
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        query = data.get('query', '')
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        # Use Google AI for analysis
        analysis_result = job_finder_service.analyze_resume_with_google_ai(query)
        
        return jsonify({
            "analysis": analysis_result,
            "ai_provider": "Google Cloud AI (Gemini)",
            "status": "success"
        })
    
    except Exception as e:
        logger.error(f"Error in analyze_resume_jobs: {e}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@app.route('/api/resume-advice', methods=['POST'])
def get_resume_advice():
    """Get resume improvement advice"""
    try:
        if not job_finder_service:
            return jsonify({"error": "Service not initialized"}), 500
            
        resume_content = job_finder_service.get_context_from_vectorstore("resume", k=1)
        if not resume_content:
            return jsonify({"error": "Please upload your resume first"}), 400
        
        advice = job_finder_service.get_resume_improvement_advice()
        
        return jsonify({
            "advice": advice,
            "ai_provider": "Google Cloud AI (Gemini)",
            "status": "success"
        })
    
    except Exception as e:
        logger.error(f"Error in get_resume_advice: {e}")
        return jsonify({"error": f"Failed to generate advice: {str(e)}"}), 500

@app.route('/api/interview-prep', methods=['POST'])
def get_interview_prep():
    """Get interview preparation guidance with resources"""
    try:
        if not job_finder_service:
            return jsonify({"error": "Service not initialized"}), 500
            
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        job_title = data.get('job_title', '')
        company = data.get('company', '')
        
        if not job_title:
            return jsonify({"error": "job_title is required"}), 400
        
        resume_content = job_finder_service.get_context_from_vectorstore("resume", k=1)
        if not resume_content:
            return jsonify({"error": "Please upload your resume first for personalized preparation"}), 400
        
        prep_guide = job_finder_service.get_interview_preparation_guide(job_title, company)
        
        return jsonify({
            "preparation_guide": prep_guide,
            "job_title": job_title,
            "company": company,
            "ai_provider": "Google Cloud AI (Gemini)",
            "status": "success"
        })
    
    except Exception as e:
        logger.error(f"Error in get_interview_prep: {e}")
        return jsonify({"error": f"Failed to generate preparation guide: {str(e)}"}), 500

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
        
        resume_context = job_finder_service.get_context_from_vectorstore("resume skills experience", k=3)
        job_context = job_finder_service.get_context_from_vectorstore(f"{job_title} {company}", k=2)
        
        prompt = f"""Generate a professional cover letter for the position of {job_title} at {company}.

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

Format the response as clean, readable text without any markdown formatting.
Make it personalized and specific to this opportunity."""
        
        response = job_finder_service.llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        clean_content = format_for_display(content)
        
        return jsonify({
            "cover_letter": clean_content,
            "job_title": job_title,
            "company": company,
            "ai_provider": "Google Cloud AI (Gemini)"
        })
    
    except Exception as e:
        logger.error(f"Error in generate_cover_letter: {e}")
        return jsonify({"error": str(e)}), 500

# Mock Interview Simulator Endpoints
@app.route('/api/mock-interview/start', methods=['POST'])
def start_mock_interview():
    """Start a new mock interview session"""
    try:
        if not job_finder_service:
            return jsonify({"error": "Service not initialized"}), 500
            
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        job_title = data.get('job_title', '')
        company = data.get('company', '')
        difficulty = data.get('difficulty', 'intermediate')
        
        if not job_title:
            return jsonify({"error": "job_title is required"}), 400
        
        logger.info(f"Starting mock interview for {job_title} at {company}")
        
        # Create new session
        session = job_finder_service.create_mock_interview_session(job_title, company, difficulty)
        
        logger.info(f"Created session {session.session_id} with {len(session.questions_asked)} questions")
        
        # Get first question
        first_question_info = job_finder_service.get_next_question(session.session_id)
        
        logger.info(f"First question info: {first_question_info}")
        
        return jsonify({
            "session_id": session.session_id,
            "message": "Mock interview session started successfully",
            "session_info": session.to_dict(),
            "first_question": first_question_info,
            "ai_provider": "Google Cloud AI (Gemini)",
            "status": "success"
        })
    
    except Exception as e:
        logger.error(f"Error starting mock interview: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/mock-interview/next-question/<session_id>', methods=['GET'])
def get_next_question(session_id):
    """Get the next question in the interview"""
    try:
        if not job_finder_service:
            return jsonify({"error": "Service not initialized"}), 500
            
        question_info = job_finder_service.get_next_question(session_id)
        
        return jsonify({
            "session_id": session_id,
            "question_info": question_info,
            "status": "success"
        })
    
    except Exception as e:
        logger.error(f"Error getting next question: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/mock-interview/submit-answer', methods=['POST'])
def submit_interview_answer():
    """Submit an answer and get feedback"""
    try:
        if not job_finder_service:
            return jsonify({"error": "Service not initialized"}), 500
            
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        session_id = data.get('session_id', '')
        answer = data.get('answer', '')
        
        if not session_id or not answer:
            return jsonify({"error": "session_id and answer are required"}), 400
        
        # Submit answer and get feedback
        result = job_finder_service.submit_answer_and_get_feedback(session_id, answer)
        
        if "error" in result:
            return jsonify({"error": result["error"]}), 400
        
        return jsonify({
            "session_id": session_id,
            "feedback": result.get("feedback"),
            "next_question_info": result.get("next_question_info"),
            "progress": result.get("progress"),
            "ai_provider": "Google Cloud AI (Gemini)",
            "status": "success"
        })
    
    except Exception as e:
        logger.error(f"Error submitting answer: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/mock-interview/session-status/<session_id>', methods=['GET'])
def get_session_status(session_id):
    """Get current session status"""
    try:
        if not job_finder_service:
            return jsonify({"error": "Service not initialized"}), 500
            
        status = job_finder_service.get_session_status(session_id)
        
        return jsonify({
            "session_id": session_id,
            "session_status": status,
            "status": "success"
        })
    
    except Exception as e:
        logger.error(f"Error getting session status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/mock-interview/pause/<session_id>', methods=['POST'])
def pause_mock_interview(session_id):
    """Pause a mock interview session"""
    try:
        if not job_finder_service:
            return jsonify({"error": "Service not initialized"}), 500
            
        result = job_finder_service.pause_session(session_id)
        
        if "error" in result:
            return jsonify({"error": result["error"]}), 400
        
        return jsonify({
            "session_id": session_id,
            "message": result["message"],
            "status": "success"
        })
    
    except Exception as e:
        logger.error(f"Error pausing session: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/mock-interview/resume/<session_id>', methods=['POST'])
def resume_mock_interview(session_id):
    """Resume a paused mock interview session"""
    try:
        if not job_finder_service:
            return jsonify({"error": "Service not initialized"}), 500
            
        result = job_finder_service.resume_session(session_id)
        
        if "error" in result:
            return jsonify({"error": result["error"]}), 400
        
        return jsonify({
            "session_id": session_id,
            "message": result["message"],
            "status": "success"
        })
    
    except Exception as e:
        logger.error(f"Error resuming session: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/mock-interview/end/<session_id>', methods=['POST'])
def end_mock_interview(session_id):
    """End a mock interview session"""
    try:
        if not job_finder_service:
            return jsonify({"error": "Service not initialized"}), 500
            
        result = job_finder_service.end_session(session_id)
        
        if "error" in result:
            return jsonify({"error": result["error"]}), 400
        
        return jsonify({
            "session_id": session_id,
            "message": result["message"],
            "final_summary": result.get("final_summary"),
            "ai_provider": "Google Cloud AI (Gemini)",
            "status": "success"
        })
    
    except Exception as e:
        logger.error(f"Error ending session: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    required_env_vars = ['GOOGLE_API_KEY', 'RAPIDAPI_KEY']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Warning: Missing environment variables: {missing_vars}")
    
    if job_finder_service:
        print(f"Service health: {'healthy' if job_finder_service.is_healthy() else 'degraded'}")
        print(f"AI Provider: Google Cloud AI (Gemini)")
        print(f"Mock Interview Simulator: Enabled")
        if job_finder_service.initialization_errors:
            for error in job_finder_service.initialization_errors:
                print(f"  - {error}")
    
    print("Starting Enhanced Flask application with Google Cloud AI and Mock Interview Simulator...")
    app.run(debug=True, host='0.0.0.0', port=5001)