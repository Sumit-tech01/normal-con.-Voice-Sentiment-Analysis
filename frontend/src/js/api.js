/**
 * API Client - Supports both Vercel and Local Backend
 */
import config, { VERCEL_API_ENDPOINTS } from './config.js';

const { useVercel } = config;

class ApiClient {
  constructor() {
    this.baseUrl = config.baseUrl;
  }

  async request(endpoint, options = {}) {
    const url = endpoint.startsWith('/') 
      ? `${this.baseUrl}${endpoint}` 
      : endpoint;

    const defaultHeaders = {
      'Content-Type': 'application/json',
    };

    const response = await fetch(url, {
      ...options,
      headers: {
        ...defaultHeaders,
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.error || `HTTP ${response.status}`);
    }

    return response.json();
  }

  // Health check
  async checkHealth() {
    if (useVercel) {
      return this.request(VERCEL_API_ENDPOINTS.HEALTH);
    }
    return this.request(config.endpoints.HEALTH);
  }

  // Analyze text sentiment
  async analyzeText(text) {
    if (useVercel) {
      return this.request(VERCEL_API_ENDPOINTS.ANALYZE, {
        method: 'POST',
        body: JSON.stringify({ text }),
      });
    }
    return this.request(config.endpoints.ANALYZE_TEXT, {
      method: 'POST',
      body: JSON.stringify({ text }),
    });
  }

  // Upload audio file for analysis
  async uploadAudio(file, onProgress = null) {
    if (useVercel) {
      throw new Error('Audio upload requires local backend. Deploy backend separately.');
    }

    return new Promise((resolve, reject) => {
      const formData = new FormData();
      formData.append('audio', file);

      const xhr = new XMLHttpRequest();
      
      if (onProgress) {
        xhr.upload.addEventListener('progress', (e) => {
          if (e.lengthComputable) {
            onProgress((e.loaded / e.total) * 100);
          }
        });
      }

      xhr.addEventListener('load', () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText));
          } catch (e) {
            reject(new Error('Invalid response'));
          }
        } else {
          reject(new Error(`Upload failed: ${xhr.status}`));
        }
      });

      xhr.addEventListener('error', () => reject(new Error('Upload failed')));
      xhr.open('POST', config.endpoints.ANALYZE_UPLOAD);
      xhr.send(formData);
    });
  }

  // Get model info
  async getModelInfo() {
    if (useVercel) {
      return { 
        status: 'demo',
        message: 'ML models require local backend deployment' 
      };
    }
    return this.request(config.endpoints.MODELS);
  }
}

export const api = new ApiClient();
export default api;
