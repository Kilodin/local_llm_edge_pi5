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
        this.activeDownloads = new Map(); // Track active downloads
        
        this.setupMiddleware();
        this.setupRoutes();
        this.setupWebSocket();
    }
    
    setupMiddleware() {
        this.app.use(cors());
        this.app.use(express.json());
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

        // Server status endpoint
        this.app.get('/api/status', (req, res) => {
            res.json({
                status: 'running',
                timestamp: new Date().toISOString(),
                modelInitialized: this.isInitialized
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
                
                const { prompt, maxTokens = 512 } = req.body;
                
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
                const {
                    temperature, topP, topK, minP, typicalP, tfsZ, topA,
                    repeatPenalty, repeatPenaltyLastN, frequencyPenalty, presencePenalty,
                    mirostatTau, mirostatEta, mirostatM,
                    ropeFreqBase, ropeFreqScale,
                    yarnExtFactor, yarnAttnFactor, yarnBetaFast, yarnBetaSlow, yarnOrigCtx,
                    defragThold, flashAttn, offloadKqv, embeddings,
                    threadsBatch, ubatchSize
                } = req.body;
                
                // Basic sampling parameters
                if (temperature !== undefined) this.llm.setTemperature(temperature);
                if (topP !== undefined) this.llm.setTopP(topP);
                if (topK !== undefined) this.llm.setTopK(topK);
                if (minP !== undefined) this.llm.setMinP(minP);
                if (typicalP !== undefined) this.llm.setTypicalP(typicalP);
                if (tfsZ !== undefined) this.llm.setTfsZ(tfsZ);
                if (topA !== undefined) this.llm.setTopA(topA);
                
                // Penalty parameters
                if (repeatPenalty !== undefined) this.llm.setRepeatPenalty(repeatPenalty);
                if (repeatPenaltyLastN !== undefined) this.llm.setRepeatPenaltyLastN(repeatPenaltyLastN);
                if (frequencyPenalty !== undefined) this.llm.setFrequencyPenalty(frequencyPenalty);
                if (presencePenalty !== undefined) this.llm.setPresencePenalty(presencePenalty);
                
                // Mirostat parameters
                if (mirostatTau !== undefined) this.llm.setMirostatTau(mirostatTau);
                if (mirostatEta !== undefined) this.llm.setMirostatEta(mirostatEta);
                if (mirostatM !== undefined) this.llm.setMirostatM(mirostatM);
                
                // RoPE parameters
                if (ropeFreqBase !== undefined) this.llm.setRopeFreqBase(ropeFreqBase);
                if (ropeFreqScale !== undefined) this.llm.setRopeFreqScale(ropeFreqScale);
                
                // YaRN parameters
                if (yarnExtFactor !== undefined) this.llm.setYarnExtFactor(yarnExtFactor);
                if (yarnAttnFactor !== undefined) this.llm.setYarnAttnFactor(yarnAttnFactor);
                if (yarnBetaFast !== undefined) this.llm.setYarnBetaFast(yarnBetaFast);
                if (yarnBetaSlow !== undefined) this.llm.setYarnBetaSlow(yarnBetaSlow);
                if (yarnOrigCtx !== undefined) this.llm.setYarnOrigCtx(yarnOrigCtx);
                
                // Memory and optimization parameters
                if (defragThold !== undefined) this.llm.setDefragThold(defragThold);
                if (flashAttn !== undefined) this.llm.setFlashAttn(flashAttn);
                if (offloadKqv !== undefined) this.llm.setOffloadKqv(offloadKqv);
                if (embeddings !== undefined) this.llm.setEmbeddings(embeddings);
                if (threadsBatch !== undefined) this.llm.setThreadsBatch(threadsBatch);
                if (ubatchSize !== undefined) this.llm.setUbatchSize(ubatchSize);
                
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

        // List available models
        this.app.get('/api/models', async (req, res) => {
            try {
                const modelsDir = '/home/kilodin/local_llm_edge_pi5/models';
                const models = [];
                
                if (await fs.pathExists(modelsDir)) {
                    const files = await fs.readdir(modelsDir);
                    
                    for (const file of files) {
                        const filePath = path.join(modelsDir, file);
                        const stats = await fs.stat(filePath);
                        
                        if (stats.isFile() && (file.endsWith('.gguf') || file.endsWith('.bin'))) {
                            const sizeInMB = (stats.size / (1024 * 1024)).toFixed(1);
                            models.push({
                                name: file,
                                path: filePath,
                                size: `${sizeInMB} MB`
                            });
                        }
                    }
                }
                
                res.json({ models });
            } catch (error) {
                console.error('Models list error:', error);
                res.status(500).json({ error: error.message });
            }
        });

        // Change active model
        this.app.post('/api/change-model', async (req, res) => {
            try {
                const { modelPath } = req.body;
                
                if (!modelPath) {
                    return res.status(400).json({ error: 'modelPath is required' });
                }
                
                // Check if model file exists
                if (!await fs.pathExists(modelPath)) {
                    return res.status(400).json({ error: 'Model file not found' });
                }
                
                // Initialize the new model
                const success = this.llm.initialize({
                    modelPath,
                    contextSize: 2048,
                    batchSize: 512,
                    threads: 4,
                    gpuLayers: 0,
                    temperature: 0.7,
                    topP: 0.9,
                    topK: 40,
                    repeatPenalty: 1.1,
                    seed: 42
                });
                
                this.isInitialized = success;
                
                if (success) {
                    res.json({ 
                        success: true, 
                        message: 'Model changed successfully',
                        modelInfo: this.llm.getModelInfo()
                    });
                } else {
                    res.status(500).json({ error: 'Failed to change model' });
                }
            } catch (error) {
                console.error('Model change error:', error);
                res.status(500).json({ error: error.message });
            }
        });

        // Download model
        this.app.post('/api/download-model', async (req, res) => {
            try {
                const { modelUrl, filename } = req.body;
                
                if (!modelUrl || !filename) {
                    return res.status(400).json({ error: 'modelUrl and filename are required' });
                }
                
                const modelsDir = '/home/kilodin/local_llm_edge_pi5/models';
                const filePath = path.join(modelsDir, filename);
                
                // Check if file already exists
                if (await fs.pathExists(filePath)) {
                    return res.status(400).json({ error: 'Model file already exists' });
                }
                
                // Start download in background
                this.downloadModel(modelUrl, filePath, filename);
                
                res.json({ success: true, message: 'Download started' });
                
            } catch (error) {
                console.error('Download model error:', error);
                res.status(500).json({ error: error.message });
            }
        });

        // Cancel download
        this.app.post('/api/cancel-download', async (req, res) => {
            try {
                const { filename } = req.body;
                
                if (!filename) {
                    return res.status(400).json({ error: 'filename is required' });
                }
                
                const downloadId = filename;
                const download = this.activeDownloads.get(downloadId);
                
                if (!download) {
                    return res.status(404).json({ error: 'Download not found' });
                }
                
                // Cancel the download
                if (download.request) {
                    download.request.destroy();
                }
                if (download.fileStream) {
                    download.fileStream.destroy();
                }
                
                // Remove from active downloads
                this.activeDownloads.delete(downloadId);
                
                // Clean up partial file
                const modelsDir = '/home/kilodin/local_llm_edge_pi5/models';
                const filePath = path.join(modelsDir, filename);
                if (await fs.pathExists(filePath)) {
                    await fs.remove(filePath);
                }
                
                this.io.emit('download-cancelled', { filename });
                
                res.json({ success: true, message: 'Download cancelled' });
                
            } catch (error) {
                console.error('Cancel download error:', error);
                res.status(500).json({ error: error.message });
            }
        });

        // Remove model
        this.app.delete('/api/remove-model', async (req, res) => {
            try {
                const { filename } = req.body;
                
                if (!filename) {
                    return res.status(400).json({ error: 'filename is required' });
                }
                
                const modelsDir = '/home/kilodin/local_llm_edge_pi5/models';
                const filePath = path.join(modelsDir, filename);
                
                // Check if file exists
                if (!await fs.pathExists(filePath)) {
                    return res.status(404).json({ error: 'Model file not found' });
                }
                
                // Check if model is currently active
                if (this.isInitialized) {
                    const modelInfo = this.llm.getModelInfo();
                    if (modelInfo && modelInfo.includes(filePath)) {
                        return res.status(400).json({ error: 'Cannot remove currently active model. Please change to a different model first.' });
                    }
                }
                
                // Remove the file
                await fs.remove(filePath);
                
                res.json({ success: true, message: 'Model removed successfully' });
                
            } catch (error) {
                console.error('Remove model error:', error);
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
                    console.log('=== GENERATION REQUEST RECEIVED ===');
                    console.log('Raw data received:', JSON.stringify(data, null, 2));
                    
                    if (!this.isInitialized) {
                        socket.emit('stream-error', { error: 'Model not initialized' });
                        return;
                    }
                    
                    const { prompt, systemPrompt, maxTokens = 512 } = data;
                    
                    console.log('Received generation request:');
                    console.log('User prompt:', prompt);
                    console.log('System prompt:', systemPrompt || 'None');
                    console.log('Max tokens:', maxTokens);
                    
                    if (!prompt) {
                        socket.emit('stream-error', { error: 'prompt is required' });
                        return;
                    }
                    
                    // Combine system prompt with user prompt if provided
                    let fullPrompt = prompt;
                    if (systemPrompt && systemPrompt.trim()) {
                        fullPrompt = `${systemPrompt.trim()}\n\n${prompt}`;
                        console.log('Combined prompt with system prompt');
                    } else {
                        // Default system prompt to prevent fake conversations
                        fullPrompt = `You are a helpful AI assistant. Answer the user's question directly without simulating conversations or pretending to be the user. Do not generate fake user responses or continue conversations on your own.\n\n${prompt}`;
                        console.log('Using default system prompt to prevent fake conversations');
                    }
                    
                    console.log('Full prompt being sent to model:', fullPrompt);
                    console.log('=== END GENERATION REQUEST ===');
                    
                    // Start streaming generation
                    this.llm.generateStream(fullPrompt, (text) => {
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
            
            // Handle download cancellation
            socket.on('cancel-download', async (data) => {
                try {
                    const { filename } = data;
                    
                    if (!filename) {
                        socket.emit('download-error', { error: 'filename is required' });
                        return;
                    }
                    
                    const downloadId = filename;
                    const download = this.activeDownloads.get(downloadId);
                    
                    if (!download) {
                        socket.emit('download-error', { error: 'Download not found' });
                        return;
                    }
                    
                    // Cancel the download
                    if (download.request) {
                        download.request.destroy();
                    }
                    if (download.fileStream) {
                        download.fileStream.destroy();
                    }
                    
                    // Remove from active downloads
                    this.activeDownloads.delete(downloadId);
                    
                    // Clean up partial file
                    const modelsDir = '/home/kilodin/local_llm_edge_pi5/models';
                    const filePath = path.join(modelsDir, filename);
                    if (await fs.pathExists(filePath)) {
                        await fs.remove(filePath);
                    }
                    
                    this.io.emit('download-cancelled', { filename });
                    
                } catch (error) {
                    console.error('Cancel download error:', error);
                    socket.emit('download-error', { error: 'Failed to cancel download' });
                }
            });
        });
    }

    async downloadModel(modelUrl, filePath, filename) {
        const https = require('https');
        const http = require('http');
        
        try {
            const url = new URL(modelUrl);
            const protocol = url.protocol === 'https:' ? https : http;
            
            const makeRequest = (url, redirectCount = 0) => {
                if (redirectCount > 5) {
                    this.io.emit('download-error', { error: 'Too many redirects' });
                    this.activeDownloads.delete(filename);
                    return;
                }
                
                const request = protocol.get(url, { followRedirect: false }, (response) => {
                    // Handle redirects
                    if (response.statusCode >= 300 && response.statusCode < 400) {
                        const location = response.headers.location;
                        if (location) {
                            console.log(`Following redirect to: ${location}`);
                            makeRequest(location, redirectCount + 1);
                            return;
                        }
                    }
                    
                    if (response.statusCode !== 200) {
                        this.io.emit('download-error', { error: `HTTP ${response.statusCode}: ${response.statusMessage}` });
                        this.activeDownloads.delete(filename);
                        return;
                    }
                    
                    const totalSize = parseInt(response.headers['content-length'], 10);
                    let downloadedSize = 0;
                    let startTime = Date.now();
                    let lastUpdateTime = startTime;
                    let lastDownloadedSize = 0;
                    
                    const fileStream = fs.createWriteStream(filePath);
                    
                    // Store download info for cancellation
                    this.activeDownloads.set(filename, {
                        request: request,
                        fileStream: fileStream,
                        filename: filename,
                        filePath: filePath
                    });
                    
                    response.on('data', (chunk) => {
                        downloadedSize += chunk.length;
                        const progress = totalSize ? Math.round((downloadedSize / totalSize) * 100) : 0;
                        
                        const currentTime = Date.now();
                        const timeDiff = (currentTime - lastUpdateTime) / 1000; // seconds
                        
                        if (timeDiff >= 1) { // Update every second
                            const bytesDiff = downloadedSize - lastDownloadedSize;
                            const speedBps = bytesDiff / timeDiff;
                            const speedMBps = (speedBps / (1024 * 1024)).toFixed(2);
                            
                            let eta = 'Calculating...';
                            if (speedBps > 0 && totalSize) {
                                const remainingBytes = totalSize - downloadedSize;
                                const etaSeconds = remainingBytes / speedBps;
                                if (etaSeconds < 60) {
                                    eta = `${Math.round(etaSeconds)}s`;
                                } else if (etaSeconds < 3600) {
                                    eta = `${Math.round(etaSeconds / 60)}m ${Math.round(etaSeconds % 60)}s`;
                                } else {
                                    const hours = Math.floor(etaSeconds / 3600);
                                    const minutes = Math.round((etaSeconds % 3600) / 60);
                                    eta = `${hours}h ${minutes}m`;
                                }
                            }
                            
                            this.io.emit('download-progress', {
                                progress,
                                status: `Downloading ${filename}...`,
                                downloaded: downloadedSize,
                                total: totalSize,
                                speed: `${speedMBps} MB/s`,
                                eta: eta
                            });
                            
                            lastUpdateTime = currentTime;
                            lastDownloadedSize = downloadedSize;
                        }
                    });
                    
                    fileStream.on('finish', () => {
                        this.activeDownloads.delete(filename);
                        this.io.emit('download-complete', { 
                            filename,
                            filePath,
                            message: 'Download completed successfully'
                        });
                    });
                    
                    fileStream.on('error', (error) => {
                        console.error('File write error:', error);
                        this.activeDownloads.delete(filename);
                        this.io.emit('download-error', { error: 'Failed to write file' });
                    });
                    
                    response.pipe(fileStream);
                });
                
                request.on('error', (error) => {
                    console.error('Download request error:', error);
                    this.activeDownloads.delete(filename);
                    this.io.emit('download-error', { error: 'Download failed' });
                });
            };
            
            makeRequest(modelUrl);
            
        } catch (error) {
            console.error('Download setup error:', error);
            this.activeDownloads.delete(filename);
            this.io.emit('download-error', { error: 'Download setup failed' });
        }
    }
    
    start(port = 3001) {
        this.server.listen(port, () => {
            console.log(`🚀 LLM Server running on port ${port}`);
            console.log(`📊 System Info: ${LLMNodeBinding.getSystemInfo()}`);
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