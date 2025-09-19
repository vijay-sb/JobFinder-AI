import React, { useState, useRef } from 'react';
import { Upload, Search, FileText, Briefcase, Target, MessageSquare, Download, Loader2, AlertCircle, CheckCircle } from 'lucide-react';

const JobFinderApp = () => {
  const [activeTab, setActiveTab] = useState('search');
  const [searchQuery, setSearchQuery] = useState('');
  const [numJobs, setNumJobs] = useState(10);
  const [jobs, setJobs] = useState([]);
  const [resumeFile, setResumeFile] = useState(null);
  const [resumeContent, setResumeContent] = useState(null);
  const [analysis, setAnalysis] = useState('');
  const [coverLetter, setCoverLetter] = useState('');
  const [loading, setLoading] = useState({
    jobs: false,
    resume: false,
    analysis: false,
    coverLetter: false
  });
  const [errors, setErrors] = useState({});
  const fileInputRef = useRef(null);

  const API_BASE_URL = 'http://localhost:5001/api';

  const setLoadingState = (key, value) => {
    setLoading(prev => ({ ...prev, [key]: value }));
  };

  const setError = (key, value) => {
    setErrors(prev => ({ ...prev, [key]: value }));
  };

  const clearError = (key) => {
    setErrors(prev => {
      const newErrors = { ...prev };
      delete newErrors[key];
      return newErrors;
    });
  };

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
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: searchQuery,
          num_jobs: numJobs
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

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

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

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
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: `Analyze my resume against these job opportunities: ${searchQuery}`
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setAnalysis(data.analysis);
      setActiveTab('analysis');
    } catch (error) {
      setError('analysis', `Failed to analyze: ${error.message}`);
    } finally {
      setLoadingState('analysis', false);
    }
  };

  const generateCoverLetter = async (jobTitle, company) => {
    setLoadingState('coverLetter', true);
    clearError('coverLetter');

    try {
      const response = await fetch(`${API_BASE_URL}/generate-cover-letter`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          job_title: jobTitle,
          company: company
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setCoverLetter(data.cover_letter);
      setActiveTab('coverLetter');
    } catch (error) {
      setError('coverLetter', `Failed to generate cover letter: ${error.message}`);
    } finally {
      setLoadingState('coverLetter', false);
    }
  };

  const extractJobInfo = (content) => {
    const lines = content.split('\n');
    let title = 'Unknown Position';
    let company = 'Unknown Company';
    
    lines.forEach(line => {
      if (line.includes('**Job Title**:')) {
        title = line.replace('**Job Title**:', '').trim();
      }
      if (line.includes('**Employer**:')) {
        company = line.replace('**Employer**:', '').trim();
      }
    });
    
    return { title, company };
  };

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

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <div className="container mx-auto px-4 py-8 max-w-6xl">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Job Finder AI Assistant
          </h1>
          <p className="text-gray-600 text-lg">
            Find jobs, analyze your resume, and get personalized recommendations
          </p>
        </div>

        {/* Navigation Tabs */}
        <div className="flex flex-wrap justify-center mb-8 bg-white rounded-xl shadow-sm p-2">
          {[
            { id: 'search', label: 'Search Jobs', icon: Search },
            { id: 'results', label: 'Job Results', icon: Briefcase },
            { id: 'analysis', label: 'Resume Analysis', icon: Target },
            { id: 'coverLetter', label: 'Cover Letter', icon: MessageSquare }
          ].map(({ id, label }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id)}
              className={`flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-all ${
                activeTab === id
                  ? 'bg-blue-600 text-white shadow-lg transform scale-105'
                  : 'text-gray-600 hover:bg-gray-50 hover:text-blue-600'
              }`}
            >
              <Icon className="h-5 w-5" />
              {label}
            </button>
          ))}
        </div>

        {/* Search Jobs Tab */}
        {activeTab === 'search' && (
          <div className="bg-white rounded-2xl shadow-lg p-8">
            <div className="flex items-center gap-3 mb-6">
              <Search className="h-6 w-6 text-blue-600" />
              <h2 className="text-2xl font-semibold text-gray-900">Search for Jobs</h2>
            </div>

            <div className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Job Search Query
                </label>
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="e.g., Python developer, Machine Learning Engineer, Data Scientist"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Number of Jobs
                </label>
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
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Upload Resume (PDF)
                </label>
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
                  {resumeFile && (
                    <span className="text-sm text-gray-600">{resumeFile.name}</span>
                  )}
                </div>
                {loading.resume && <LoadingSpinner text="Uploading and processing resume..." />}
                {resumeContent && (
                  <SuccessMessage message="Resume uploaded and processed successfully!" />
                )}
              </div>

              {errors.jobs && <ErrorMessage message={errors.jobs} />}
              {errors.resume && <ErrorMessage message={errors.resume} />}

              <div className="flex gap-4">
                <button
                  onClick={searchJobs}
                  disabled={loading.jobs || !searchQuery.trim()}
                  className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading.jobs ? (
                    <Loader2 className="h-5 w-5 animate-spin" />
                  ) : (
                    <Search className="h-5 w-5" />
                  )}
                  Search Jobs
                </button>

                {resumeContent && jobs.length > 0 && (
                  <button
                    onClick={analyzeResumeJobs}
                    disabled={loading.analysis}
                    className="flex items-center gap-2 px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 focus:ring-2 focus:ring-green-500 disabled:opacity-50"
                  >
                    {loading.analysis ? (
                      <Loader2 className="h-5 w-5 animate-spin" />
                    ) : (
                      <Target className="h-5 w-5" />
                    )}
                    Analyze Match
                  </button>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Job Results Tab */}
        {activeTab === 'results' && (
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
                  const { title, company } = extractJobInfo(job.content);
                  return (
                    <div key={index} className="border border-gray-200 rounded-lg p-6 hover:shadow-md transition-shadow">
                      <div className="flex justify-between items-start mb-4">
                        <div>
                          <h3 className="text-xl font-semibold text-gray-900 mb-1">
                            {title}
                          </h3>
                          <p className="text-gray-600">{company}</p>
                        </div>
                        <button
                          onClick={() => generateCoverLetter(title, company)}
                          disabled={loading.coverLetter}
                          className="flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 focus:ring-2 focus:ring-purple-500 disabled:opacity-50 text-sm"
                        >
                          {loading.coverLetter ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <MessageSquare className="h-4 w-4" />
                          )}
                          Generate Cover Letter
                        </button>
                      </div>
                      <div className="prose prose-sm max-w-none">
                        <pre className="whitespace-pre-wrap font-sans text-sm text-gray-700 bg-gray-50 p-4 rounded-lg overflow-x-auto">
                          {job.content}
                        </pre>
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
        )}

        {/* Analysis Tab */}
        {activeTab === 'analysis' && (
          <div className="bg-white rounded-2xl shadow-lg p-8">
            <div className="flex items-center gap-3 mb-6">
              <Target className="h-6 w-6 text-blue-600" />
              <h2 className="text-2xl font-semibold text-gray-900">Resume Analysis</h2>
            </div>

            {errors.analysis && <ErrorMessage message={errors.analysis} />}

            {loading.analysis ? (
              <LoadingSpinner text="Analyzing your resume against job opportunities..." />
            ) : analysis ? (
              <div className="prose max-w-none">
                <div className="bg-gray-50 rounded-lg p-6">
                  <pre className="whitespace-pre-wrap font-sans text-gray-700 leading-relaxed">
                    {analysis}
                  </pre>
                </div>
              </div>
            ) : (
              <div className="text-center py-12">
                <Target className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">No analysis available</h3>
                <p className="text-gray-600">Upload your resume and search for jobs to get a detailed analysis.</p>
              </div>
            )}
          </div>
        )}

        {/* Cover Letter Tab */}
        {activeTab === 'coverLetter' && (
          <div className="bg-white rounded-2xl shadow-lg p-8">
            <div className="flex items-center gap-3 mb-6">
              <MessageSquare className="h-6 w-6 text-blue-600" />
              <h2 className="text-2xl font-semibold text-gray-900">Cover Letter</h2>
              {coverLetter && (
                <button
                  onClick={() => {
                    const blob = new Blob([coverLetter], { type: 'text/plain' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'cover-letter.txt';
                    a.click();
                    URL.revokeObjectURL(url);
                  }}
                  className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 focus:ring-2 focus:ring-green-500 text-sm ml-auto"
                >
                  <Download className="h-4 w-4" />
                  Download
                </button>
              )}
            </div>

            {errors.coverLetter && <ErrorMessage message={errors.coverLetter} />}

            {loading.coverLetter ? (
              <LoadingSpinner text="Generating personalized cover letter..." />
            ) : coverLetter ? (
              <div className="prose max-w-none">
                <div className="bg-gray-50 rounded-lg p-6">
                  <pre className="whitespace-pre-wrap font-sans text-gray-700 leading-relaxed">
                    {coverLetter}
                  </pre>
                </div>
              </div>
            ) : (
              <div className="text-center py-12">
                <MessageSquare className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">No cover letter generated</h3>
                <p className="text-gray-600">Click "Generate Cover Letter" on any job in the results to create a personalized cover letter.</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default JobFinderApp;