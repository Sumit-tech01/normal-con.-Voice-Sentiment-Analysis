/**
 * WebSocket Client Module
 * Handles real-time communication with the server
 */

export class WebSocketClient {
    constructor(url = null) {
        this.socket = null;
        this.url = url || this.getDefaultUrl();
        this.connected = false;
        this.sessionId = null;
        this.eventHandlers = {};
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
    }

    /**
     * Get default WebSocket URL based on environment
     */
    getDefaultUrl() {
        const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:';
        
        // For development: backend runs on port 5000
        // For production with nginx: backend is on same host
        const isLocalhost = window.location.hostname === 'localhost' || 
                           window.location.hostname === '127.0.0.1';
        
        if (isLocalhost) {
            // Development: backend on port 5000
            return `${protocol}//localhost:5000`;
        } else {
            // Production: use same origin
            return `${protocol}//${window.location.host}`;
        }
    }

    /**
     * Connect to WebSocket server
     */
    connect() {
        if (this.socket?.connected) {
            console.log('Already connected');
            return;
        }

        console.log('Connecting to WebSocket server...');

        try {
            this.socket = io(this.url, {
                path: '/socket.io',
                transports: ['websocket', 'polling'],
                reconnection: true,
                reconnectionAttempts: this.maxReconnectAttempts,
                reconnectionDelay: this.reconnectDelay,
                timeout: 10000,
            });

            this.setupEventListeners();
        } catch (error) {
            console.error('Failed to connect:', error);
            this.emit('error', error);
        }
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Connection events
        this.socket.on('connect', () => {
            console.log('WebSocket connected');
            this.connected = true;
            this.reconnectAttempts = 0;
            this.emit('connect');
        });

        this.socket.on('disconnect', (reason) => {
            console.log('WebSocket disconnected:', reason);
            this.connected = false;
            this.emit('disconnect', reason);
        });

        this.socket.on('connect_error', (error) => {
            console.error('Connection error:', error);
            this.reconnectAttempts++;
            this.emit('error', error);
        });

        this.socket.on('connect_timeout', (timeout) => {
            console.error('Connection timeout:', timeout);
            this.emit('error', new Error('Connection timeout'));
        });

        // Custom events
        this.socket.on('connected', (data) => {
            console.log('Server acknowledged connection:', data);
            this.emit('server_connected', data);
        });

        this.socket.on('session_started', (data) => {
            console.log('Session started:', data);
            this.sessionId = data.session_id;
            this.emit('session_started', data);
        });

        this.socket.on('sentiment_update', (data) => {
            console.log('Sentiment update:', data);
            this.emit('sentiment_update', data);
        });

        this.socket.on('sentiment_complete', (data) => {
            console.log('Sentiment complete:', data);
            this.emit('sentiment_complete', data);
        });

        this.socket.on('session_ended', (data) => {
            console.log('Session ended:', data);
            this.emit('session_ended', data);
        });

        this.socket.on('error', (data) => {
            console.error('Server error:', data);
            this.emit('error', new Error(data.error));
        });

        this.socket.on('status', (data) => {
            console.log('Status:', data);
            this.emit('status', data);
        });

        this.socket.on('pong', (data) => {
            console.log('Pong:', data);
            this.emit('pong', data);
        });
    }

    /**
     * Start a new streaming session
     */
    startSession(language = null) {
        if (!this.connected) {
            console.warn('Not connected to server');
            return;
        }

        const data = language ? { language } : {};
        this.socket.emit('start_session', data);
    }

    /**
     * Send an audio chunk
     */
    sendAudioChunk(audioData, chunkId, isFinal = false, timestamp = null) {
        if (!this.connected) {
            console.warn('Not connected to server');
            return false;
        }

        const data = {
            data: audioData,  // Base64 encoded audio
            chunk_id: chunkId,
            is_final: isFinal,
            timestamp: timestamp || Date.now() / 1000,
            session_id: this.sessionId,
        };

        this.socket.emit('audio_chunk', data);
        return true;
    }

    /**
     * End current session
     */
    endSession() {
        if (!this.connected || !this.sessionId) {
            return;
        }

        this.socket.emit('end_session', { session_id: this.sessionId });
        this.sessionId = null;
    }

    /**
     * Send ping (keep-alive)
     */
    ping() {
        if (!this.connected) {
            return;
        }

        this.socket.emit('ping');
    }

    /**
     * Request status
     */
    getStatus() {
        if (!this.connected) {
            return;
        }

        this.socket.emit('get_status');
    }

    /**
     * Disconnect from server
     */
    disconnect() {
        if (this.socket) {
            this.socket.disconnect();
            this.socket = null;
            this.connected = false;
            this.sessionId = null;
            console.log('Disconnected from server');
        }
    }

    /**
     * Register event handler
     */
    on(event, handler) {
        if (!this.eventHandlers[event]) {
            this.eventHandlers[event] = [];
        }
        this.eventHandlers[event].push(handler);
    }

    /**
     * Remove event handler
     */
    off(event, handler) {
        if (!this.eventHandlers[event]) {
            return;
        }

        const index = this.eventHandlers[event].indexOf(handler);
        if (index > -1) {
            this.eventHandlers[event].splice(index, 1);
        }
    }

    /**
     * Emit event to handlers
     */
    emit(event, data) {
        if (!this.eventHandlers[event]) {
            return;
        }

        this.eventHandlers[event].forEach(handler => {
            try {
                handler(data);
            } catch (error) {
                console.error(`Error in event handler for ${event}:`, error);
            }
        });
    }

    /**
     * Check if connected
     */
    isConnected() {
        return this.connected;
    }

    /**
     * Get current session ID
     */
    getSessionId() {
        return this.sessionId;
    }
}

