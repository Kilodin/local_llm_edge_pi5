import React, { useState, useEffect, useRef } from 'react';
import io from 'socket.io-client';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import { Send, Settings, Download, Upload, Bot, User, X, Play, Square } from 'lucide-react';
import './App.css';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:3001';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [socket, setSocket] = useState(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [modelInfo, setModelInfo] = useState('');
  const [systemInfo, setSystemInfo] = useState('');
  const [showSettings, setShowSettings] = useState(false);
  const [showInit, setShowInit] = useState(false);
  const [messageIdCounter, setMessageIdCounter] = useState(0);
  const isInitializingRef = useRef(false);
  const currentStreamingContentRef = useRef('');
  const streamingTimeoutRef = useRef(null);
  const [initConfig, setInitConfig] = useState({
    modelPath: '',
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
  const [parameters, setParameters] = useState({
    temperature: 0.7,
    topP: 0.9,
    topK: 40,
    repeatPenalty: 1.1
  });
  const messagesEndRef = useRef(null);

  useEffect(() => {
    console.log('App useEffect called - setting up socket and checking initialization');
    // Initialize socket connection
    const newSocket = io(API_BASE);
    setSocket(newSocket);

    // Socket event handlers
    newSocket.on('stream-chunk', (data) => {
      console.log('Received stream chunk:', data);
      
      // Check if this is the completion signal
      if (data.text === '[DONE]') {
        console.log('Received [DONE] signal, stopping streaming');
        console.log('Current isStreaming state before update:', isStreaming);
        console.log('Current isLoading state before update:', isLoading);
        
        // Clear the timeout
        if (streamingTimeoutRef.current) {
          clearTimeout(streamingTimeoutRef.current);
          streamingTimeoutRef.current = null;
        }
        
        // Always force the states to false when [DONE] is received
        setIsStreaming(false);
        setIsLoading(false);
        console.log('Forced isStreaming and isLoading to false.');
        
        // Reset streaming content
        currentStreamingContentRef.current = '';
        return;
      }
      
      // Build the complete response by appending new text
      const newContent = currentStreamingContentRef.current + data.text;
      
      // Check if this would create duplicate content
      if (newContent === currentStreamingContentRef.current) {
        console.log('No new content to add, ignoring chunk');
        return;
      }
      
      // Update tracking content
      currentStreamingContentRef.current = newContent;
      
      setMessages(prev => {
        const newMessages = [...prev];
        
        // Check if we need to create a new assistant message
        if (newMessages.length === 0 || newMessages[newMessages.length - 1].type !== 'assistant') {
          // Create new assistant message with the complete content
          const newMessage = {
            id: `${Date.now()}-${messageIdCounter}-${Math.random()}`,
            type: 'assistant',
            content: newContent,
            timestamp: new Date().toLocaleTimeString(),
            isError: false
          };
          setMessageIdCounter(prev => prev + 1);
          return [...newMessages, newMessage];
        } else {
          // Replace the entire content of the existing assistant message
          console.log('Updating content to:', newContent);
          newMessages[newMessages.length - 1].content = newContent;
          return newMessages;
        }
      });
    });

    newSocket.on('stream-error', (data) => {
      setIsStreaming(false);
      setIsLoading(false);
      addMessage('assistant', `Error: ${data.error}`, true);
    });

    newSocket.on('generation-stopped', () => {
      setIsStreaming(false);
      setIsLoading(false);
    });

    // Check server status and auto-initialize model if needed
    const checkAndInit = async () => {
      if (isInitializingRef.current) {
        console.log('Initialization already in progress, skipping checkAndInit.');
        return;
      }
      try {
        console.log('Checking backend health...');
        const response = await axios.get(`${API_BASE}/api/health`);
        console.log('Health response:', response.data);
        setIsInitialized(response.data.initialized);
        setSystemInfo(response.data.systemInfo);
        if (!response.data.initialized) {
          console.log('Backend not initialized, calling /api/initialize...');
          setIsLoading(true);
          isInitializingRef.current = true;
          addMessage('system', 'Initializing default model...');
          const initConfig = {
            modelPath: '/home/kilodin/local_llm_edge_pi5/models/tinyllama-1.1b-chat.gguf',
            contextSize: 2048,
            batchSize: 512,
            threads: 4,
            gpuLayers: 0,
            temperature: 0.7,
            topP: 0.9,
            topK: 40,
            repeatPenalty: 1.1,
            seed: 42
          };
          console.log('Calling /api/initialize with', initConfig);
          const initResp = await axios.post(`${API_BASE}/api/initialize`, initConfig);
          console.log('Init response:', initResp.data);
          if (initResp.data.success) {
            setIsInitialized(true);
            setModelInfo(initResp.data.modelInfo);
            addMessage('system', 'Default model initialized successfully! 🎉');
          } else {
            addMessage('system', 'Failed to initialize default model', true);
          }
        } else {
          if (response.data.initialized) {
            const modelResponse = await axios.get(`${API_BASE}/api/model-info`);
            setModelInfo(modelResponse.data.info);
          }
        }
      } catch (error) {
        console.error('Initialization error:', error);
        addMessage('system', `Initialization error: ${error.response?.data?.error || error.message}`, true);
      } finally {
        setIsLoading(false);
        isInitializingRef.current = false;
      }
    };
    console.log('About to call checkAndInit()');
    checkAndInit();
    return () => newSocket.close();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Debug useEffect to monitor state changes
  useEffect(() => {
    console.log('State changed - isStreaming:', isStreaming, 'isLoading:', isLoading);
  }, [isStreaming, isLoading]);

  const checkServerStatus = async () => {
    try {
      const response = await axios.get(`${API_BASE}/api/health`);
      setIsInitialized(response.data.initialized);
      setSystemInfo(response.data.systemInfo);
      
      if (response.data.initialized) {
        const modelResponse = await axios.get(`${API_BASE}/api/model-info`);
        setModelInfo(modelResponse.data.info);
      }
    } catch (error) {
      console.error('Server not available:', error);
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const addMessage = (type, content, isError = false) => {
    const newMessage = {
      id: `${Date.now()}-${messageIdCounter}`,
      type,
      content,
      timestamp: new Date().toLocaleTimeString(),
      isError
    };
    setMessageIdCounter(prev => prev + 1);
    setMessages(prev => [...prev, newMessage]);
  };

  const initializeModel = async () => {
    if (isInitializingRef.current) {
      console.log('Initialization already in progress, ignoring duplicate request.');
      return;
    }
    try {
      setIsLoading(true);
      isInitializingRef.current = true; // Set flag to true
      const response = await axios.post(`${API_BASE}/api/initialize`, initConfig);
      
      if (response.data.success) {
        setIsInitialized(true);
        setModelInfo(response.data.modelInfo);
        setShowInit(false);
        addMessage('system', 'Model initialized successfully! 🎉');
      } else {
        addMessage('system', 'Failed to initialize model', true);
      }
    } catch (error) {
      addMessage('system', `Initialization error: ${error.response?.data?.error || error.message}`, true);
    } finally {
      setIsLoading(false);
      isInitializingRef.current = false; // Reset flag
    }
  };

  const updateParameters = async () => {
    try {
      await axios.post(`${API_BASE}/api/parameters`, parameters);
      setShowSettings(false);
      addMessage('system', 'Parameters updated successfully!');
    } catch (error) {
      addMessage('system', `Parameter update error: ${error.response?.data?.error || error.message}`, true);
    }
  };

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput('');
    addMessage('user', userMessage);

    if (!isInitialized) {
      addMessage('assistant', 'Please initialize a model first using the settings panel.', true);
      return;
    }

    setIsLoading(true);
    setIsStreaming(true);

    // Reset streaming content for new generation
    currentStreamingContentRef.current = '';

    // Add a timeout fallback to ensure states are reset
    streamingTimeoutRef.current = setTimeout(() => {
      console.log('Timeout fallback: Resetting streaming states');
      setIsStreaming(false);
      setIsLoading(false);
    }, 30000); // 30 seconds timeout

    // Don't create empty assistant message here - let streaming create it

    try {
      // Use streaming for better UX
      socket.emit('generate-stream', {
        prompt: userMessage,
        maxTokens: 256
      });
    } catch (error) {
      if (streamingTimeoutRef.current) {
        clearTimeout(streamingTimeoutRef.current);
        streamingTimeoutRef.current = null;
      }
      setIsStreaming(false);
      setIsLoading(false);
      addMessage('assistant', `Error: ${error.message}`, true);
    }
  };

  const stopGeneration = () => {
    if (socket) {
      socket.emit('stop-generation');
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <h1>🤖 Local LLM Inference</h1>
          <div className="header-actions">
            <button 
              className="btn btn-secondary"
              onClick={() => setShowInit(true)}
              disabled={isInitialized}
            >
              <Download size={16} />
              Initialize Model
            </button>
            <button 
              className="btn btn-secondary"
              onClick={() => setShowSettings(true)}
            >
              <Settings size={16} />
              Settings
            </button>
          </div>
        </div>
      </header>

      <main className="main">
        <div className="chat-container">
          <div className="messages">
            {messages.length === 0 ? (
              <div className="welcome">
                <Bot size={48} />
                <h2>Welcome to Local LLM Inference</h2>
                <p>Initialize a model and start chatting with your local AI!</p>
                {!isInitialized && (
                  <button 
                    className="btn btn-primary"
                    onClick={() => setShowInit(true)}
                  >
                    <Download size={16} />
                    Initialize Model
                  </button>
                )}
                {systemInfo && (
                  <div className="system-info">
                    <h3>System Information</h3>
                    <pre>{systemInfo}</pre>
                  </div>
                )}
              </div>
            ) : (
              messages.map((message) => (
                <div 
                  key={message.id} 
                  className={`message ${message.type} ${message.isError ? 'error' : ''}`}
                >
                  <div className="message-avatar">
                    {message.type === 'user' ? <User size={20} /> : <Bot size={20} />}
                  </div>
                  <div className="message-content">
                    {message.type === 'assistant' ? (
                      <ReactMarkdown>{message.content}</ReactMarkdown>
                    ) : (
                      <p>{message.content}</p>
                    )}
                    <span className="message-time">{message.timestamp}</span>
                  </div>
                </div>
              ))
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="input-container">
            <div className="input-wrapper">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Type your message..."
                disabled={isLoading || !isInitialized}
                rows={1}
              />
              <div className="input-actions">
                {isStreaming ? (
                  <button 
                    className="btn btn-danger"
                    onClick={stopGeneration}
                    title="Stop generation"
                  >
                    <Square size={16} />
                  </button>
                ) : (
                  <button 
                    className="btn btn-primary"
                    onClick={sendMessage}
                    disabled={!input.trim() || isLoading || !isInitialized}
                    title="Send message"
                  >
                    <Send size={16} />
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Initialize Model Modal */}
      {showInit && (
        <div className="modal-overlay">
          <div className="modal">
            <div className="modal-header">
              <h2>Initialize Model</h2>
              <button 
                className="btn-close"
                onClick={() => setShowInit(false)}
              >
                <X size={20} />
              </button>
            </div>
            <div className="modal-content">
              <div className="form-group">
                <label>Model Path:</label>
                <input
                  type="text"
                  value={initConfig.modelPath}
                  onChange={(e) => setInitConfig(prev => ({...prev, modelPath: e.target.value}))}
                  placeholder="/path/to/your/model.gguf"
                />
              </div>
              <div className="form-row">
                <div className="form-group">
                  <label>Context Size:</label>
                  <input
                    type="number"
                    value={initConfig.contextSize}
                    onChange={(e) => setInitConfig(prev => ({...prev, contextSize: parseInt(e.target.value)}))}
                  />
                </div>
                <div className="form-group">
                  <label>Threads:</label>
                  <input
                    type="number"
                    value={initConfig.threads}
                    onChange={(e) => setInitConfig(prev => ({...prev, threads: parseInt(e.target.value)}))}
                  />
                </div>
              </div>
              <div className="form-row">
                <div className="form-group">
                  <label>Temperature:</label>
                  <input
                    type="number"
                    step="0.1"
                    value={initConfig.temperature}
                    onChange={(e) => setInitConfig(prev => ({...prev, temperature: parseFloat(e.target.value)}))}
                  />
                </div>
                <div className="form-group">
                  <label>Top-P:</label>
                  <input
                    type="number"
                    step="0.1"
                    value={initConfig.topP}
                    onChange={(e) => setInitConfig(prev => ({...prev, topP: parseFloat(e.target.value)}))}
                  />
                </div>
              </div>
            </div>
            <div className="modal-footer">
              <button 
                className="btn btn-secondary"
                onClick={() => setShowInit(false)}
              >
                Cancel
              </button>
              <button 
                className="btn btn-primary"
                onClick={initializeModel}
                disabled={isLoading || !initConfig.modelPath}
              >
                {isLoading ? 'Initializing...' : 'Initialize'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Settings Modal */}
      {showSettings && (
        <div className="modal-overlay">
          <div className="modal">
            <div className="modal-header">
              <h2>Settings</h2>
              <button 
                className="btn-close"
                onClick={() => setShowSettings(false)}
              >
                <X size={20} />
              </button>
            </div>
            <div className="modal-content">
              <div className="form-row">
                <div className="form-group">
                  <label>Temperature:</label>
                  <input
                    type="number"
                    step="0.1"
                    value={parameters.temperature}
                    onChange={(e) => setParameters(prev => ({...prev, temperature: parseFloat(e.target.value)}))}
                  />
                </div>
                <div className="form-group">
                  <label>Top-P:</label>
                  <input
                    type="number"
                    step="0.1"
                    value={parameters.topP}
                    onChange={(e) => setParameters(prev => ({...prev, topP: parseFloat(e.target.value)}))}
                  />
                </div>
              </div>
              <div className="form-row">
                <div className="form-group">
                  <label>Top-K:</label>
                  <input
                    type="number"
                    value={parameters.topK}
                    onChange={(e) => setParameters(prev => ({...prev, topK: parseInt(e.target.value)}))}
                  />
                </div>
                <div className="form-group">
                  <label>Repeat Penalty:</label>
                  <input
                    type="number"
                    step="0.1"
                    value={parameters.repeatPenalty}
                    onChange={(e) => setParameters(prev => ({...prev, repeatPenalty: parseFloat(e.target.value)}))}
                  />
                </div>
              </div>
              
              {modelInfo && (
                <div className="info-section">
                  <h3>Model Information</h3>
                  <pre>{modelInfo}</pre>
                </div>
              )}
              
              {systemInfo && (
                <div className="info-section">
                  <h3>System Information</h3>
                  <pre>{systemInfo}</pre>
                </div>
              )}
            </div>
            <div className="modal-footer">
              <button 
                className="btn btn-secondary"
                onClick={() => setShowSettings(false)}
              >
                Cancel
              </button>
              <button 
                className="btn btn-primary"
                onClick={updateParameters}
              >
                Update Parameters
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App; 