/**
 * API Configuration for Vercel Deployment
 * Automatically detects environment and sets base URL
 */

const getBaseUrl = () => {
  // Check for Vercel environment
  if (process.env.VERCEL) {
    return ''; // Use relative URLs (Vercel API routes)
  }
  
  // Check for custom backend URL
  if (process.env.VITE_API_URL) {
    return process.env.VITE_API_URL;
  }
  
  // Default to local backend
  return 'http://localhost:5000';
};

const API_BASE_URL = getBaseUrl();

// API Endpoints
export const API_ENDPOINTS = {
  HEALTH: `${API_BASE_URL}/api/v1/health`,
  ANALYZE_UPLOAD: `${API_BASE_URL}/api/v1/analyze/upload`,
  ANALYZE_TEXT: `${API_BASE_URL}/api/v1/analyze/text`,
  STREAM_START: `${API_BASE_URL}/api/v1/analyze/stream/start`,
  MODELS: `${API_BASE_URL}/api/v1/models`,
  DOCS: `${API_BASE_URL}/api/v1/docs`,
};

// Vercel Serverless API Routes
export const VERCEL_API_ENDPOINTS = {
  HEALTH: '/api/health',
  ANALYZE: '/api/analyze',
};

// Determine which API to use
export const useVercelAPI = () => !!process.env.VERCEL;

export default {
  baseUrl: API_BASE_URL,
  endpoints: API_ENDPOINTS,
  vercelEndpoints: VERCEL_API_ENDPOINTS,
  useVercel: useVercelAPI(),
};
