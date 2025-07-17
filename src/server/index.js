const express = require('express');
const cors = require('cors');
const http = require('http');
const socketIo = require('socket.io');
const path = require('path');
const fs = require('fs-extra');

// Import the native addon
const { LLMNodeBinding } = require('../../build/Release/llm_node');

class LLMServer {
    constructor() {
        this.app = express();
        this.server = http.createServer(this.app);
        this.io = socketIo(this.server, {
            cors: {
                origin: "*",
                methods: ["GET", "POST"]
            }
        });
        
        this.llm = new LLMNodeBinding();
        this.isInitialized = false;
        
        this.setupMiddleware();
        this.setupRoutes();
        this.setupWebSocket();
    }
    
    setupMiddleware() {
        this.app.use(cors());
        this.app.use(express.json());
        this.app.use(express.static(path.join(process.cwd(), 'src/ui/build')));
    }
    
    setupRoutes() {
        // Health check
        this.app.get('/api/health', (req, res) => {
            res.json({
                status: 'ok',
                initialized: this.isInitialized,
                systemInfo: LLMNodeBinding.getSystemInfo()
            });
        });
        
        // Initialize model
        this.app.post('/api/initialize', async (req, res) => {
            try {
                const config = req.body;
                
                // Validate required fields
                if (!config.modelPath) {
                    return res.status(400).json({ error: 'modelPath is required' });
                }
                
                // Check if model file exists
                if (!await fs.pathExists(config.modelPath)) {
                    return res.status(400).json({ error: 'Model file not found' });
                }
                
                const success = this.llm.initialize(config);
                this.isInitialized = success;
                
                if (success) {
                    res.json({ 
                        success: true, 
                        message: 'Model initialized successfully',
                        modelInfo: this.llm.getModelInfo()
                    });
                } else {
                    res.status(500).json({ error: 'Failed to initialize model' });
                }
            } catch (error) {
                console.error('Initialization error:', error);
                res.status(500).json({ error: error.message });
            }
        });
        
        // Generate text
        this.app.post('/api/generate', async (req, res) => {
            try {
                if (!this.isInitialized) {
                    return res.status(400).json({ error: 'Model not initialized' });
                }
                
                const { prompt, maxTokens = 256 } = req.body;
                
                if (!prompt) {
                    return res.status(400).json({ error: 'prompt is required' });
                }
                
                const result = this.llm.generate(prompt, maxTokens);
                res.json({ result });
                
            } catch (error) {
                console.error('Generation error:', error);
                res.status(500).json({ error: error.message });
            }
        });
        
        // Update parameters
        this.app.post('/api/parameters', async (req, res) => {
            try {
                const { temperature, topP, topK, repeatPenalty } = req.body;
                
                if (temperature !== undefined) {
                    this.llm.setTemperature(temperature);
                }
                if (topP !== undefined) {
                    this.llm.setTopP(topP);
                }
                if (topK !== undefined) {
                    this.llm.setTopK(topK);
                }
                if (repeatPenalty !== undefined) {
                    this.llm.setRepeatPenalty(repeatPenalty);
                }
                
                res.json({ success: true });
                
            } catch (error) {
                console.error('Parameter update error:', error);
                res.status(500).json({ error: error.message });
            }
        });
        
        // Get model info
        this.app.get('/api/model-info', (req, res) => {
            try {
                const info = this.llm.getModelInfo();
                res.json({ info });
            } catch (error) {
                console.error('Model info error:', error);
                res.status(500).json({ error: error.message });
            }
        });
        
        // File upload for RAG
        this.app.post('/api/upload', async (req, res) => {
            try {
                // This would be implemented with multer for file uploads
                res.json({ message: 'File upload endpoint (to be implemented)' });
            } catch (error) {
                console.error('Upload error:', error);
                res.status(500).json({ error: error.message });
            }
        });
        
        // Serve React app
        this.app.get('*', (req, res) => {
            res.sendFile(path.join(process.cwd(), 'src/ui/build/index.html'));
        });
    }
    
    setupWebSocket() {
        this.io.on('connection', (socket) => {
            console.log('Client connected:', socket.id);
            
            // Handle streaming generation
            socket.on('generate-stream', async (data) => {
                try {
                    if (!this.isInitialized) {
                        socket.emit('stream-error', { error: 'Model not initialized' });
                        return;
                    }
                    
                    const { prompt, maxTokens = 256 } = data;
                    
                    if (!prompt) {
                        socket.emit('stream-error', { error: 'prompt is required' });
                        return;
                    }
                    
                    // Start streaming generation
                    this.llm.generateStream(prompt, (text) => {
                        socket.emit('stream-chunk', { text });
                    }, maxTokens);
                    
                } catch (error) {
                    console.error('Stream generation error:', error);
                    socket.emit('stream-error', { error: error.message });
                }
            });
            
            // Handle stop generation
            socket.on('stop-generation', () => {
                try {
                    this.llm.stopGeneration();
                    socket.emit('generation-stopped');
                } catch (error) {
                    console.error('Stop generation error:', error);
                }
            });
            
            socket.on('disconnect', () => {
                console.log('Client disconnected:', socket.id);
            });
        });
    }
    
    start(port = 3001) {
        this.server.listen(port, () => {
            console.log(`🚀 LLM Server running on port ${port}`);
            console.log(`📊 System Info: ${LLMNodeBinding.getSystemInfo()}`);
            console.log(`🌐 Web UI available at http://localhost:${port}`);
        });
    }
}

// Start server if this file is run directly
if (require.main === module) {
    const server = new LLMServer();
    const port = process.env.PORT || 3001;
    server.start(port);
}

module.exports = LLMServer; 