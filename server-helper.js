const express = require('express');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = 4000;

app.use(express.json());

let serverProcess = null;
let serverStatus = 'stopped'; // 'stopped', 'starting', 'running', 'error'

// Check if the main server is running
async function checkServerStatus() {
    try {
        const response = await fetch('http://localhost:3001/api/status');
        return response.ok;
    } catch (error) {
        return false;
    }
}

// Start the main LLM server
async function startServer() {
    if (serverStatus === 'starting' || serverStatus === 'running') {
        return { success: false, message: 'Server is already starting or running' };
    }

    // First check if server is already running
    const isRunning = await checkServerStatus();
    if (isRunning) {
        serverStatus = 'running';
        console.log('LLM Server already running on port 3001');
        return { success: true, message: 'Server already running' };
    }

    serverStatus = 'starting';
    
    const projectDir = __dirname;
    const env = {
        ...process.env,
        LD_LIBRARY_PATH: `${projectDir}/third_party/llama.cpp/build/bin:${projectDir}/build/Release`
    };

    console.log('Starting LLM server...');
    
    serverProcess = spawn('node', ['src/server/index.js'], {
        cwd: projectDir,
        env: env,
        stdio: ['pipe', 'pipe', 'pipe']
    });

    serverProcess.stdout.on('data', (data) => {
        console.log(`LLM Server stdout: ${data}`);
        if (data.toString().includes('Server running on port 3001')) {
            serverStatus = 'running';
            console.log('LLM Server started successfully');
        }
    });

    serverProcess.stderr.on('data', (data) => {
        console.log(`LLM Server stderr: ${data}`);
        if (data.toString().includes('EADDRINUSE')) {
            serverStatus = 'running';
            console.log('LLM Server already running on port 3001');
        }
    });

    serverProcess.on('close', (code) => {
        console.log(`LLM Server process exited with code ${code}`);
        if (serverStatus === 'starting') {
            serverStatus = 'error';
        }
        serverProcess = null;
    });

    serverProcess.on('error', (error) => {
        console.error('Failed to start LLM Server:', error);
        serverStatus = 'error';
        serverProcess = null;
    });

    return { success: true, message: 'Server starting...' };
}

// Stop the main LLM server
function stopServer() {
    if (serverProcess) {
        serverProcess.kill('SIGTERM');
        serverStatus = 'stopped';
        serverProcess = null;
        return { success: true, message: 'Server stopped' };
    }
    return { success: false, message: 'No server process to stop' };
}

// API Routes
app.get('/status', async (req, res) => {
    const isRunning = await checkServerStatus();
    res.json({
        status: isRunning ? 'running' : serverStatus,
        isRunning: isRunning
    });
});

app.post('/start-server', async (req, res) => {
    const result = await startServer();
    res.json(result);
});

app.post('/stop-server', (req, res) => {
    const result = stopServer();
    res.json(result);
});

app.get('/health', (req, res) => {
    res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Graceful shutdown
process.on('SIGINT', () => {
    console.log('Shutting down server helper...');
    if (serverProcess) {
        serverProcess.kill('SIGTERM');
    }
    process.exit(0);
});

process.on('SIGTERM', () => {
    console.log('Shutting down server helper...');
    if (serverProcess) {
        serverProcess.kill('SIGTERM');
    }
    process.exit(0);
});

// Start the helper server
app.listen(PORT, () => {
    console.log(`Server helper running on port ${PORT}`);
    console.log(`Project directory: ${__dirname}`);
});

module.exports = app; 