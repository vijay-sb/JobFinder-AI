import React, { useState, useRef } from 'react';
import { 
  Upload, Search, Briefcase, Target, MessageSquare, Download, 
  Loader2, AlertCircle, CheckCircle, BookOpen, Users, Lightbulb, 
  MapPin, ExternalLink, Play, Square, SkipForward, Award, TrendingUp 
} from 'lucide-react';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL;

// Utility Components
const ErrorMessage = ({ message }) => (
  <div className="flex items-center gap-2 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
    <AlertCircle className="h-4 w-4 flex-shrink-0" />
    <span>{message}</span>
  </div>
);

const SuccessMessage = ({ message }) => (
  <div className="flex items-center gap-2 p-3 bg-green-50 border border-green-200 rounded-lg text-green-700 text-sm">
    <CheckCircle className="h-4 w-4 flex-shrink-0" />
    <span>{message}</span>
  </div>
);

const LoadingSpinner = ({ text }) => (
  <div className="flex items-center justify-center gap-3 p-8">
    <Loader2 className="h-6 w-6 animate-spin text-blue-600" />
    <span className="text-gray-600">{text}</span>
  </div>
);

const ProgressBar = ({ current, total, percentage }) => (
  <div className="w-full bg-gray-200 rounded-full h-2">
    <div 
      className="bg-blue-600 h-2 rounded-full transition-all duration-300" 
      style={{ width: `${percentage}%` }}
    />
    <div className="flex justify-between items-center mt-1 text-sm text-gray-600">
      <span>Question {current} of {total}</span>
      <span>{Math.round(percentage)}% Complete</span>
    </div>
  </div>
);

// Main Component
const JobFinderApp = () => {
  // State Management
  const [activeTab, setActiveTab] = useState('search');
  const [searchQuery, setSearchQuery] = useState('');
  const [numJobs, setNumJobs] = useState(10);
  const [jobs, setJobs] = useState([]);
  const [resumeFile, setResumeFile] = useState(null);
  const [resumeContent, setResumeContent] = useState(null);
  const [analysis, setAnalysis] = useState('');
  const [coverLetter, setCoverLetter] = useState('');
  const [resumeAdvice, setResumeAdvice] = useState('');
  const [interviewPrep, setInterviewPrep] = useState('');
  const [selectedJob, setSelectedJob] = useState(null);

  // Mock Interview States
  const [mockSession, setMockSession] = useState(null);
  const [currentQuestion, setCurrentQuestion] = useState(null);
  const [currentAnswer, setCurrentAnswer] = useState('');
  const [feedback, setFeedback] = useState(null);
  const [sessionSummary, setSessionSummary] = useState(null);
  const [interviewProgress, setInterviewProgress] = useState({ current: 0, total: 0, percentage: 0 });

  // Loading States
  const [loading, setLoading] = useState({
    jobs: false,
    resume: false,
    analysis: false,
    coverLetter: false,
    resumeAdvice: false,
    interviewPrep: false,
    mockInterview: false,
    submitAnswer: false
  });
  const [jobSpecificLoading, setJobSpecificLoading] = useState({});
  const [errors, setErrors] = useState({});
  const fileInputRef = useRef(null);

  // Utility Functions
  const setLoadingState = (key, value) => setLoading(prev => ({ ...prev, [key]: value }));
  const setJobLoadingState = (jobIndex, actionType, value) => {
    setJobSpecificLoading(prev => ({ ...prev, [`${jobIndex}_${actionType}`]: value }));
  };
  const setError = (key, value) => setErrors(prev => ({ ...prev, [key]: value }));
  const clearError = (key) => setErrors(prev => {
    const newErrors = { ...prev };
    delete newErrors[key];
    return newErrors;
  });

  // API Functions
  const searchJobs = async () => {
    if (!searchQuery.trim()) {
      setError('jobs', 'Please enter a search query');
      return;
    }
    setLoadingState('jobs', true);
    clearError('jobs');
    try {
      const response = await fetch(`${API_BASE_URL}/search-jobs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: searchQuery, num_jobs: numJobs })
      });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      setJobs(data.jobs || []);
      setActiveTab('results');
    } catch (error) {
      setError('jobs', `Failed to search jobs: ${error.message}`);
    } finally {
      setLoadingState('jobs', false);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setError('resume', 'Please select a PDF file');
      return;
    }
    setResumeFile(file);
    setLoadingState('resume', true);
    clearError('resume');
    const formData = new FormData();
    formData.append('resume', file);
    try {
      const response = await fetch(`${API_BASE_URL}/upload-resume`, {
        method: 'POST',
        body: formData
      });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      setResumeContent(data.resume);
    } catch (error) {
      setError('resume', `Failed to upload resume: ${error.message}`);
    } finally {
      setLoadingState('resume', false);
    }
  };

  const analyzeResumeJobs = async () => {
    if (!resumeContent || jobs.length === 0) {
      setError('analysis', 'Please upload a resume and search for jobs first');
      return;
    }
    setLoadingState('analysis', true);
    clearError('analysis');
    try {
      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: `Analyze my resume against these job opportunities: ${searchQuery}` })
      });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      setAnalysis(data.analysis);
      setActiveTab('analysis');
    } catch (error) {
      setError('analysis', `Failed to analyze: ${error.message}`);
    } finally {
      setLoadingState('analysis', false);
    }
  };

  const generateCoverLetter = async (jobTitle, company, jobIndex) => {
    setJobLoadingState(jobIndex, 'coverLetter', true);
    clearError('coverLetter');
    try {
      const response = await fetch(`${API_BASE_URL}/generate-cover-letter`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_title: jobTitle, company: company })
      });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      setCoverLetter(data.cover_letter);
      setActiveTab('coverLetter');
    } catch (error) {
      setError('coverLetter', `Failed to generate cover letter: ${error.message}`);
    } finally {
      setJobLoadingState(jobIndex, 'coverLetter', false);
    }
  };

  const getResumeAdvice = async () => {
    if (!resumeContent) {
      setError('resumeAdvice', 'Please upload your resume first');
      return;
    }
    setLoadingState('resumeAdvice', true);
    clearError('resumeAdvice');
    try {
      const response = await fetch(`${API_BASE_URL}/resume-advice`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      setResumeAdvice(data.advice);
      setActiveTab('resumeAdvice');
    } catch (error) {
      setError('resumeAdvice', `Failed to get advice: ${error.message}`);
    } finally {
      setLoadingState('resumeAdvice', false);
    }
  };

  const getInterviewPrep = async (jobTitle, company = '', jobIndex) => {
    if (!resumeContent) {
      setError('interviewPrep', 'Please upload your resume first for personalized preparation');
      return;
    }
    setJobLoadingState(jobIndex, 'interviewPrep', true);
    clearError('interviewPrep');
    try {
      const response = await fetch(`${API_BASE_URL}/interview-prep`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_title: jobTitle, company: company })
      });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      setInterviewPrep(data.preparation_guide);
      setSelectedJob({ title: jobTitle, company });
      setActiveTab('interviewPrep');
    } catch (error) {
      setError('interviewPrep', `Failed to get interview preparation: ${error.message}`);
    } finally {
      setJobLoadingState(jobIndex, 'interviewPrep', false);
    }
  };

  // Mock Interview Functions
  const startMockInterview = async (jobTitle, company = '', difficulty = 'intermediate') => {
    if (!resumeContent) {
      setError('mockInterview', 'Please upload your resume first');
      return;
    }
    setLoadingState('mockInterview', true);
    clearError('mockInterview');
    try {
      const response = await fetch(`${API_BASE_URL}/mock-interview/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_title: jobTitle, company: company, difficulty: difficulty })
      });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      setMockSession(data.session_info);
      setCurrentQuestion(data.first_question.question);
      setInterviewProgress({
        current: 1,
        total: data.first_question.total_questions,
        percentage: (1 / data.first_question.total_questions) * 100
      });
      setFeedback(null);
      setSessionSummary(null);
      setCurrentAnswer('');
      setActiveTab('mockInterview');
    } catch (error) {
      setError('mockInterview', `Failed to start mock interview: ${error.message}`);
    } finally {
      setLoadingState('mockInterview', false);
    }
  };

  const submitAnswer = async () => {
    if (!mockSession || !currentAnswer.trim()) {
      setError('mockInterview', 'Please provide an answer before submitting');
      return;
    }
    setLoadingState('submitAnswer', true);
    clearError('mockInterview');
    try {
      const response = await fetch(`${API_BASE_URL}/mock-interview/submit-answer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: mockSession.session_id, answer: currentAnswer })
      });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      setFeedback(data.feedback);
      setInterviewProgress(data.progress);
      
      if (data.next_question_info.is_complete) {
        setCurrentQuestion(null);
        setSessionSummary(data.next_question_info.session_summary);
      } else {
        setCurrentQuestion(data.next_question_info.question);
      }
      setCurrentAnswer('');
    } catch (error) {
      setError('mockInterview', `Failed to submit answer: ${error.message}`);
    } finally {
      setLoadingState('submitAnswer', false);
    }
  };

  const endMockInterview = async () => {
    if (!mockSession) return;
    try {
      const response = await fetch(`${API_BASE_URL}/mock-interview/end/${mockSession.session_id}`, {
        method: 'POST'
      });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      setSessionSummary(data.final_summary);
      setCurrentQuestion(null);
      setCurrentAnswer('');
    } catch (error) {
      setError('mockInterview', `Failed to end interview: ${error.message}`);
    }
  };

  // Helper Functions
  const extractJobInfo = (content, metadata) => {
    if (metadata) {
      return {
        title: metadata.job_title || 'Unknown Position',
        company: metadata.employer || 'Unknown Company',
        location: metadata.location || '',
        mapsLink: metadata.maps_link || '',
        applyLink: metadata.apply_link || ''
      };
    }
    // Fallback parsing logic here if needed
    return {
      title: 'Unknown Position',
      company: 'Unknown Company',
      location: '',
      mapsLink: '',
      applyLink: ''
    };
  };

  const formatContent = (content) => {
    if (!content) return '';
    return content.split('\n').map((line, index) => {
      line = line.trim();
      if (!line) return null;
      
      if (line === line.toUpperCase() && line.length > 3 && line.length < 50) {
        return <h3 key={index} className="font-semibold text-lg text-gray-800 mt-4 mb-2">{line}</h3>;
      }
      if (line.match(/^(Current Strengths|Skills Development|Technical Skills|Certification|Resume Content|Learning Path|Action Items|Overview|Key Areas|Preparation Strategy|Resources for Practice):/i)) {
        return <h4 key={index} className="font-medium text-base text-gray-700 mt-3 mb-2">{line}</h4>;
      }
      if (line.startsWith('•') || line.startsWith('-')) {
        return <li key={index} className="ml-4 mb-1 text-gray-700">{line.substring(1).trim()}</li>;
      }
      return <p key={index} className="text-gray-700 mb-2 leading-relaxed">{line}</p>;
    }).filter(Boolean);
  };

  const formatMockInterviewQuestion = (question) => {
    if (!question) return '';
    return (
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-blue-600 bg-blue-100 px-3 py-1 rounded-full">
            {question.category || 'General'}
          </span>
          <span className="text-sm text-gray-500">
            Difficulty: {question.difficulty || 'Medium'}
          </span>
        </div>
        <div className="text-lg font-medium text-gray-900 leading-relaxed">
          {question.question}
        </div>
        {question.key_points && (
          <div className="text-sm text-gray-600 bg-gray-50 p-3 rounded-lg">
            <strong>Key points to consider:</strong> {question.key_points}
          </div>
        )}
      </div>
    );
  };

  const downloadFile = (content, filename) => {
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  const resetMockInterview = () => {
    setMockSession(null);
    setCurrentQuestion(null);
    setCurrentAnswer('');
    setFeedback(null);
    setSessionSummary(null);
    setInterviewProgress({ current: 0, total: 0, percentage: 0 });
  };

  // Render Methods
  const renderSearchTab = () => (
    <div className="bg-white rounded-2xl shadow-lg p-8">
      <div className="flex items-center gap-3 mb-6">
        <Search className="h-6 w-6 text-blue-600" />
        <h2 className="text-2xl font-semibold text-gray-900">Search for Jobs</h2>
      </div>

      <div className="space-y-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Job Search Query</label>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="e.g., Python developer, Machine Learning Engineer, Data Scientist"
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Number of Jobs</label>
          <select
            value={numJobs}
            onChange={(e) => setNumJobs(parseInt(e.target.value))}
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value={5}>5 jobs</option>
            <option value={10}>10 jobs</option>
            <option value={15}>15 jobs</option>
            <option value={20}>20 jobs</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Upload Resume (PDF)</label>
          <div className="flex items-center gap-4">
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf"
              onChange={handleFileUpload}
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={loading.resume}
              className="flex items-center gap-2 px-6 py-3 bg-gray-100 border border-gray-300 rounded-lg hover:bg-gray-50 focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
            >
              <Upload className="h-5 w-5" />
              Choose File
            </button>
            {resumeFile && <span className="text-sm text-gray-600">{resumeFile.name}</span>}
          </div>
          {loading.resume && <LoadingSpinner text="Uploading and processing resume..." />}
          {resumeContent && <SuccessMessage message="Resume uploaded and processed successfully!" />}
        </div>

        {errors.jobs && <ErrorMessage message={errors.jobs} />}
        {errors.resume && <ErrorMessage message={errors.resume} />}

        <div className="flex gap-4 flex-wrap">
          <button
            onClick={searchJobs}
            disabled={loading.jobs || !searchQuery.trim()}
            className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
          >
            {loading.jobs ? <Loader2 className="h-5 w-5 animate-spin" /> : <Search className="h-5 w-5" />}
            Search Jobs
          </button>

          {resumeContent && jobs.length > 0 && (
            <button
              onClick={analyzeResumeJobs}
              disabled={loading.analysis}
              className="flex items-center gap-2 px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 focus:ring-2 focus:ring-green-500 disabled:opacity-50"
            >
              {loading.analysis ? <Loader2 className="h-5 w-5 animate-spin" /> : <Target className="h-5 w-5" />}
              Analyze Match
            </button>
          )}

          {resumeContent && (
            <button
              onClick={getResumeAdvice}
              disabled={loading.resumeAdvice}
              className="flex items-center gap-2 px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 focus:ring-2 focus:ring-purple-500 disabled:opacity-50"
            >
              {loading.resumeAdvice ? <Loader2 className="h-5 w-5 animate-spin" /> : <Lightbulb className="h-5 w-5" />}
              Get Career Advice
            </button>
          )}
        </div>
      </div>
    </div>
  );

  const renderJobResultsTab = () => (
    <div className="bg-white rounded-2xl shadow-lg p-8">
      <div className="flex items-center gap-3 mb-6">
        <Briefcase className="h-6 w-6 text-blue-600" />
        <h2 className="text-2xl font-semibold text-gray-900">Job Results</h2>
        <span className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm font-medium">
          {jobs.length} jobs found
        </span>
      </div>

      {loading.jobs ? (
        <LoadingSpinner text="Searching for jobs..." />
      ) : jobs.length > 0 ? (
        <div className="space-y-6">
          {jobs.map((job, index) => {
            const { title, company, location, mapsLink, applyLink } = extractJobInfo(job.content, job.metadata);
            return (
              <div key={index} className="border border-gray-200 rounded-lg p-6 hover:shadow-md transition-shadow">
                <div className="flex justify-between items-start mb-4">
                  <div className="flex-1">
                    <h3 className="text-xl font-semibold text-gray-900 mb-1">{title}</h3>
                    <p className="text-gray-600 mb-2">{company}</p>
                    {location && (
                      <div className="flex items-center gap-2 text-sm text-gray-500 mb-2">
                        <MapPin className="h-4 w-4" />
                        <span>{location}</span>
                        {mapsLink && mapsLink !== 'N/A' && (
                          <a href={mapsLink} target="_blank" rel="noopener noreferrer" className="flex items-center gap-1 text-blue-600 hover:text-blue-800">
                            <ExternalLink className="h-3 w-3" />
                            View on Map
                          </a>
                        )}
                      </div>
                    )}
                    {applyLink && applyLink !== 'N/A' && (
                      <a href={applyLink} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1 text-sm text-green-600 hover:text-green-800 font-medium">
                        <ExternalLink className="h-3 w-3" />
                        Apply Now
                      </a>
                    )}
                  </div>
                  
                  <div className="flex gap-2 flex-wrap ml-4">
                    <button
                      onClick={() => generateCoverLetter(title, company, index)}
                      disabled={jobSpecificLoading[`${index}_coverLetter`]}
                      className="flex items-center gap-2 px-3 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 focus:ring-2 focus:ring-purple-500 disabled:opacity-50 text-sm"
                    >
                      {jobSpecificLoading[`${index}_coverLetter`] ? <Loader2 className="h-4 w-4 animate-spin" /> : <MessageSquare className="h-4 w-4" />}
                      Cover Letter
                    </button>
                    
                    {resumeContent && (
                      <>
                        <button
                          onClick={() => getInterviewPrep(title, company, index)}
                          disabled={jobSpecificLoading[`${index}_interviewPrep`]}
                          className="flex items-center gap-2 px-3 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 focus:ring-2 focus:ring-orange-500 disabled:opacity-50 text-sm"
                        >
                          {jobSpecificLoading[`${index}_interviewPrep`] ? <Loader2 className="h-4 w-4 animate-spin" /> : <BookOpen className="h-4 w-4" />}
                          Interview Prep
                        </button>
                        
                        <button
                          onClick={() => startMockInterview(title, company, 'intermediate')}
                          disabled={loading.mockInterview}
                          className="flex items-center gap-2 px-3 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 focus:ring-2 focus:ring-green-500 disabled:opacity-50 text-sm"
                        >
                          {loading.mockInterview ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                          Mock Interview
                        </button>
                      </>
                    )}
                  </div>
                </div>
                
                <div className="bg-gray-50 rounded-lg p-4">
                  <div className="text-sm text-gray-700 leading-relaxed whitespace-pre-line">
                    {job.content}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="text-center py-12">
          <Briefcase className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No jobs found</h3>
          <p className="text-gray-600">Try searching with different keywords or criteria.</p>
        </div>
      )}
    </div>
  );

  const renderMockInterviewTab = () => (
    <div className="bg-white rounded-2xl shadow-lg p-8">
      <div className="flex items-center gap-3 mb-6">
        <Play className="h-6 w-6 text-blue-600" />
        <h2 className="text-2xl font-semibold text-gray-900">AI Mock Interview Simulator</h2>
        {mockSession && (
          <div className="flex items-center gap-2 text-sm text-gray-600 bg-blue-50 px-3 py-1 rounded-full">
            <Users className="h-4 w-4" />
            {mockSession.job_title} at {mockSession.company || 'Various Companies'}
          </div>
        )}
      </div>

      {errors.mockInterview && <ErrorMessage message={errors.mockInterview} />}

      {/* Interview Not Started */}
      {!mockSession && !loading.mockInterview && (
        <div className="space-y-6">
          <div className="text-center py-8">
            <Play className="h-16 w-16 text-blue-600 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-gray-900 mb-2">Start Your Mock Interview</h3>
            <p className="text-gray-600 mb-6">Practice with AI-powered interview questions tailored to your resume and target job</p>
            
            {!resumeContent ? (
              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-6">
                <p className="text-yellow-800">Please upload your resume first to get personalized interview questions.</p>
              </div>
            ) : jobs.length === 0 ? (
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
                <p className="text-blue-800">Search for jobs to practice interviews for specific positions, or start a general interview.</p>
                <div className="mt-4">
                  <button
                    onClick={() => startMockInterview('Software Engineer', '', 'intermediate')}
                    className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:ring-2 focus:ring-blue-500"
                  >
                    Start General Interview
                  </button>
                </div>
              </div>
            ) : (
              <div className="text-gray-600">
                Click "Mock Interview" on any job in the results to start practicing for that specific position.
              </div>
            )}
          </div>
        </div>
      )}

      {/* Loading State */}
      {loading.mockInterview && <LoadingSpinner text="Setting up your personalized mock interview with AI..." />}

      {/* Active Interview */}
      {mockSession && currentQuestion && !sessionSummary && (
        <div className="space-y-6">
          {/* Progress Bar */}
          <div className="bg-gray-50 rounded-lg p-4">
            <ProgressBar 
              current={interviewProgress.current}
              total={interviewProgress.total}
              percentage={interviewProgress.percentage}
            />
          </div>

          {/* Current Question */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Question {interviewProgress.current}</h3>
            {formatMockInterviewQuestion(currentQuestion)}
          </div>

          {/* Answer Input */}
          <div className="space-y-4">
            <label className="block text-sm font-medium text-gray-700">Your Answer</label>
            <textarea
              value={currentAnswer}
              onChange={(e) => setCurrentAnswer(e.target.value)}
              placeholder="Type your detailed answer here... Use the STAR method (Situation, Task, Action, Result) for behavioral questions."
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent min-h-[120px] resize-vertical"
            />
            <div className="flex gap-3">
              <button
                onClick={submitAnswer}
                disabled={loading.submitAnswer || !currentAnswer.trim()}
                className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading.submitAnswer ? <Loader2 className="h-5 w-5 animate-spin" /> : <SkipForward className="h-5 w-5" />}
                Submit Answer
              </button>
              
              <button
                onClick={endMockInterview}
                className="flex items-center gap-2 px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 focus:ring-2 focus:ring-red-500"
              >
                <Square className="h-5 w-5" />
                End Interview
              </button>
            </div>
          </div>

          {/* Recent Feedback */}
          {feedback && (
            <div className="bg-green-50 border border-green-200 rounded-lg p-6">
              <div className="flex items-center gap-2 mb-3">
                <Award className="h-5 w-5 text-green-600" />
                <h4 className="font-semibold text-green-900">Feedback for Previous Answer</h4>
                <span className="bg-green-100 text-green-800 px-2 py-1 rounded-full text-sm font-medium">
                  Score: {feedback.score}/10
                </span>
              </div>
              <div className="prose max-w-none text-green-800">
                {formatContent(feedback.feedback_text)}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Interview Completed - Session Summary */}
      {sessionSummary && (
        <div className="space-y-6">
          <div className="text-center py-6 bg-green-50 border border-green-200 rounded-lg">
            <Award className="h-16 w-16 text-green-600 mx-auto mb-4" />
            <h3 className="text-2xl font-semibold text-green-900 mb-2">Interview Completed!</h3>
            <p className="text-green-700">Great job! Here's your detailed performance analysis.</p>
          </div>

          {/* Performance Summary */}
          <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-6 border-l-4 border-blue-500">
            <div className="flex items-center justify-between mb-4">
              <h4 className="text-lg font-semibold text-gray-900">Performance Summary</h4>
              <div className="flex items-center gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-600">{sessionSummary.overall_score}/10</div>
                  <div className="text-sm text-gray-600">Overall Score</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600">{sessionSummary.total_questions}</div>
                  <div className="text-sm text-gray-600">Questions</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-purple-600">{Math.round(sessionSummary.session_duration)}m</div>
                  <div className="text-sm text-gray-600">Duration</div>
                </div>
              </div>
            </div>

            {/* Category Scores */}
            {sessionSummary.category_scores && Object.keys(sessionSummary.category_scores).length > 0 && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                {Object.entries(sessionSummary.category_scores).map(([category, score]) => (
                  <div key={category} className="bg-white rounded-lg p-3 text-center">
                    <div className="text-lg font-semibold text-gray-900">{score}/10</div>
                    <div className="text-sm text-gray-600 capitalize">{category}</div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Detailed Analysis */}
          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <h4 className="text-lg font-semibold text-gray-900 mb-4">Detailed Analysis</h4>
            <div className="prose max-w-none">
              {formatContent(sessionSummary.summary_text)}
            </div>
          </div>

          {/* Recommendations */}
          {sessionSummary.recommendations && sessionSummary.recommendations.length > 0 && (
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6">
              <h4 className="text-lg font-semibold text-yellow-900 mb-4">Recommendations for Improvement</h4>
              <ul className="space-y-2">
                {sessionSummary.recommendations.map((rec, index) => (
                  <li key={index} className="flex items-start gap-2 text-yellow-800">
                    <TrendingUp className="h-4 w-4 mt-1 flex-shrink-0" />
                    <span>{rec}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex gap-4 justify-center">
            <button
              onClick={() => downloadFile(JSON.stringify(sessionSummary, null, 2), 'interview-performance-report.json')}
              className="flex items-center gap-2 px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 focus:ring-2 focus:ring-green-500"
            >
              <Download className="h-5 w-5" />
              Download Report
            </button>
            
            <button
              onClick={resetMockInterview}
              className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:ring-2 focus:ring-blue-500"
            >
              <Play className="h-5 w-5" />
              Start New Interview
            </button>
          </div>
        </div>
      )}
    </div>
  );

  const renderContentTab = (title, icon, content, downloadFilename, loadingKey, errorKey) => (
    <div className="bg-white rounded-2xl shadow-lg p-8">
      <div className="flex items-center gap-3 mb-6">
        {icon}
        <h2 className="text-2xl font-semibold text-gray-900">{title}</h2>
        {selectedJob && title === 'Interview Preparation' && (
          <div className="flex items-center gap-2 text-sm text-gray-600 bg-blue-50 px-3 py-1 rounded-full">
            <Users className="h-4 w-4" />
            {selectedJob.title} at {selectedJob.company}
          </div>
        )}
        {content && downloadFilename && (
          <button
            onClick={() => downloadFile(content, downloadFilename)}
            className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 focus:ring-2 focus:ring-green-500 text-sm ml-auto"
          >
            <Download className="h-4 w-4" />
            Download
          </button>
        )}
      </div>

      {errors[errorKey] && <ErrorMessage message={errors[errorKey]} />}

      {loading[loadingKey] ? (
        <LoadingSpinner text={`Generating ${title.toLowerCase()} with Google AI...`} />
      ) : content ? (
        <div className={`rounded-lg p-6 ${title === 'Career & Resume Advice' ? 'bg-gradient-to-r from-purple-50 to-blue-50 border-l-4 border-purple-500' : 
          title === 'Interview Preparation' ? 'bg-gradient-to-r from-orange-50 to-yellow-50 border-l-4 border-orange-500' : 'bg-gray-50'}`}>
          <div className="prose max-w-none">
            {formatContent(content)}
          </div>
          {title === 'Interview Preparation' && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mt-6">
              <p className="text-blue-800 text-sm">
                <strong>Note:</strong> This preparation guide is personalized based on your resume and includes 
                resources from top interview preparation websites and YouTube channels. Ready to practice? 
                Try the Mock Interview Simulator for hands-on experience!
              </p>
            </div>
          )}
        </div>
      ) : (
        <div className="text-center py-12">
          {icon}
          <h3 className="text-lg font-medium text-gray-900 mb-2">No {title.toLowerCase()} available</h3>
          <p className="text-gray-600">
            {title === 'Resume Analysis' ? 'Upload your resume and search for jobs to get a detailed analysis.' :
             title === 'Cover Letter' ? 'Click "Cover Letter" on any job in the results to create a personalized cover letter.' :
             title === 'Career & Resume Advice' ? 'Upload your resume and click "Get Career Advice" to receive personalized improvement suggestions.' :
             'Click "Interview Prep" on any job in the results to get a personalized preparation guide.'}
          </p>
        </div>
      )}
    </div>
  );

  // Main Render
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <div className="container mx-auto px-4 py-8 max-w-6xl">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Job Finder AI Assistant</h1>
          <p className="text-gray-600 text-lg">
            Powered by Google Cloud AI - Find jobs, analyze your resume, get career advice, and practice interviews
          </p>
        </div>

        {/* Navigation Tabs */}
        <div className="flex flex-wrap justify-center mb-8 bg-white rounded-xl shadow-sm p-2">
          {[
            { id: 'search', label: 'Search Jobs', icon: Search },
            { id: 'results', label: 'Job Results', icon: Briefcase },
            { id: 'analysis', label: 'Resume Analysis', icon: Target },
            { id: 'coverLetter', label: 'Cover Letter', icon: MessageSquare },
            { id: 'resumeAdvice', label: 'Career Advice', icon: Lightbulb },
            { id: 'interviewPrep', label: 'Interview Prep', icon: BookOpen },
            { id: 'mockInterview', label: 'Mock Interview', icon: Play }
          ].map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id)}
              className={`flex items-center gap-2 px-4 py-3 rounded-lg font-medium transition-all text-sm ${
                activeTab === id
                  ? 'bg-blue-600 text-white shadow-lg transform scale-105'
                  : 'text-gray-600 hover:bg-gray-50 hover:text-blue-600'
              }`}
            >
              <Icon className="h-4 w-4" />
              {label}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        {activeTab === 'search' && renderSearchTab()}
        {activeTab === 'results' && renderJobResultsTab()}
        {activeTab === 'mockInterview' && renderMockInterviewTab()}
        {activeTab === 'analysis' && renderContentTab(
          'Resume Analysis', 
          <Target className="h-6 w-6 text-blue-600" />, 
          analysis, 
          null, 
          'analysis', 
          'analysis'
        )}
        {activeTab === 'coverLetter' && renderContentTab(
          'Cover Letter', 
          <MessageSquare className="h-6 w-6 text-blue-600" />, 
          coverLetter, 
          'cover-letter.txt', 
          'coverLetter', 
          'coverLetter'
        )}
        {activeTab === 'resumeAdvice' && renderContentTab(
          'Career & Resume Advice', 
          <Lightbulb className="h-6 w-6 text-blue-600" />, 
          resumeAdvice, 
          'career-advice.txt', 
          'resumeAdvice', 
          'resumeAdvice'
        )}
        {activeTab === 'interviewPrep' && renderContentTab(
          'Interview Preparation', 
          <BookOpen className="h-6 w-6 text-blue-600" />, 
          interviewPrep, 
          'interview-prep-guide.txt', 
          'interviewPrep', 
          'interviewPrep'
        )}
      </div>
    </div>
  );
};

export default JobFinderApp;