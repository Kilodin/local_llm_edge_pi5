import React, { useState, useEffect, useRef } from 'react';
import { flushSync } from 'react-dom';
import io from 'socket.io-client';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import { Send, Settings, Download, Upload, Bot, User, X, Play, Square, Plus, RotateCcw, Power } from 'lucide-react';
import './App.css';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:3001';

// GenerationMetrics component
function GenerationMetrics({ metrics }) {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!metrics) return null;

  return (
    <div className="generation-metrics">
      {/* Compact Header - Always Visible */}
      <div className="metrics-header" onClick={() => setIsExpanded(!isExpanded)}>
        <span>⚡ Performance Metrics</span>
        <span className="metrics-summary">
          Input Tokens: {metrics.input_tokens} - Output Tokens: {metrics.output_tokens} - Duration: {metrics.duration_seconds.toFixed(2)}s - Tokens Per Second: {metrics.tokens_per_second.toFixed(1)}/s
        </span>
        <span className="expand-icon">{isExpanded ? '▼' : '▶'}</span>
      </div>

      {/* Expanded Content - Only visible when expanded */}
      {isExpanded && (
        <div className="metrics-expanded">
          {/* Advanced Metrics */}
          {(metrics.context_used || metrics.first_token_latency_ms) && (
            <div className="metrics-section">
              <div className="section-title">Advanced</div>
              <div className="metrics-grid">
                {metrics.first_token_latency_ms && (
                  <div className="metric-item">
                    <span className="metric-label">First Token</span>
                    <span className="metric-value">{metrics.first_token_latency_ms.toFixed(1)}ms</span>
                  </div>
                )}
                {metrics.context_used && (
                  <div className="metric-item">
                    <span className="metric-label">Context</span>
                    <span className="metric-value">{metrics.context_used}/{metrics.context_size}</span>
                  </div>
                )}
                {metrics.context_usage_percent && (
                  <div className="metric-item">
                    <span className="metric-label">Usage</span>
                    <span className="metric-value">{metrics.context_usage_percent.toFixed(1)}%</span>
                  </div>
                )}
                {metrics.eos_hit && (
                  <div className="metric-item">
                    <span className="metric-label">EOS Hit</span>
                    <span className="metric-value">{metrics.eos_hit === "true" ? "Yes" : "No"}</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Configuration */}
          <div className="metrics-section">
            <div className="section-title">Configuration</div>
            <div className="metrics-grid">
              <div className="metric-item">
                <span className="metric-label">Temperature</span>
                <span className="metric-value">{metrics.temperature?.toFixed(2) || "N/A"}</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Top-P</span>
                <span className="metric-value">{metrics.top_p?.toFixed(2) || "N/A"}</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Top-K</span>
                <span className="metric-value">{metrics.top_k || "N/A"}</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Threads</span>
                <span className="metric-value">{metrics.threads || "N/A"}</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Helper to parse system info string into key-value pairs
function parseSystemInfo(info) {
  if (!info) return [];
  return info.split('\n')
    .map(line => {
      const idx = line.indexOf(':');
      if (idx !== -1) {
        return {
          key: line.slice(0, idx).trim(),
          value: line.slice(idx + 1).trim(),
        };
      }
      return null;
    })
    .filter(Boolean)
    .filter(row => row.value && row.key);
}

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [socket, setSocket] = useState(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [modelInfo, setModelInfo] = useState('');
  const [systemInfo, setSystemInfo] = useState('');
  const [availableModels, setAvailableModels] = useState([]);
  const [showSettings, setShowSettings] = useState(false);
  const [showInit, setShowInit] = useState(false);
  const [showModelDownload, setShowModelDownload] = useState(false);
  const [showDownloadProgress, setShowDownloadProgress] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState({});
  const [selectedModel, setSelectedModel] = useState(null);
  const [messageIdCounter, setMessageIdCounter] = useState(0);
  const [serverConnected, setServerConnected] = useState(false);
  const [serverStatus, setServerStatus] = useState('checking'); // 'checking', 'starting', 'running', 'error'
  const [showRetryButton, setShowRetryButton] = useState(false);
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
    // Basic sampling parameters
    temperature: 0.7,
    topP: 0.9,
    topK: 40,
    minP: 0.0,
    typicalP: 1.0,
    tfsZ: 1.0,
    topA: 0.0,

    // Penalty parameters
    repeatPenalty: 1.1,
    repeatPenaltyLastN: 64,
    frequencyPenalty: 0.0,
    presencePenalty: 0.0,

    // Mirostat parameters
    mirostatTau: 5.0,
    mirostatEta: 0.1,
    mirostatM: 100,

    // RoPE parameters
    ropeFreqBase: 0.0,
    ropeFreqScale: 0.0,

    // YaRN parameters
    yarnExtFactor: -1.0,
    yarnAttnFactor: 1.0,
    yarnBetaFast: 32.0,
    yarnBetaSlow: 1.0,
    yarnOrigCtx: 0,

    // Memory and optimization parameters
    defragThold: 0.0,
    flashAttn: false,
    offloadKqv: false,
    embeddings: false,
    threadsBatch: 4,
    ubatchSize: 512
  });

  // System prompt for controlling model personality
  const [systemPrompt, setSystemPrompt] = useState('You are a helpful AI assistant.');
  const messagesEndRef = useRef(null);
  // Add a dummy state for forced re-render
  const [forceRender, setForceRender] = useState(0);
  // Separate state for streaming content to force re-renders
  const [streamingContent, setStreamingContent] = useState('');
  // Add state to track cursor position for streaming
  const [cursorPosition, setCursorPosition] = useState(0);

  // Function to get the current word and position the cursor
  const getStreamingDisplay = (content) => {
    if (!content || !content.trim()) {
      return { displayText: '', cursorPos: 0 };
    }

    // For streaming, we want the cursor to appear at the very end of the content
    // This gives the effect of the cursor following the text as it's being typed
    return { 
      displayText: content, 
      cursorPos: content.length 
    };
  };

  // Supported models for Raspberry Pi
  const supportedModels = [
    {
      name: "TinyLlama 1.1B Chat",
      filename: "tinyllama-1.1b-chat.gguf",
      url: "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
      size: "637.8 MB",
      description: "Lightweight chat model, great for Raspberry Pi"
    },
    {
      name: "Phi-2",
      filename: "phi-2.gguf",
      url: "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf",
      size: "1706.4 MB",
      description: "Microsoft's Phi-2 model, good performance/size ratio"
    },
    {
      name: "Llama-2-7B-Chat",
      filename: "llama-2-7b-chat.gguf",
      url: "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf",
      size: "4.37 GB",
      description: "Larger model, requires more RAM but better quality"
    },
    {
      name: "Mistral 7B Instruct",
      filename: "mistral-7b-instruct.gguf",
      url: "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
      size: "4.37 GB",
      description: "High quality instruction model"
    },
    {
      name: "CodeLlama 7B",
      filename: "codellama-7b.gguf",
      url: "https://huggingface.co/TheBloke/CodeLlama-7B-GGUF/resolve/main/codellama-7b.Q4_K_M.gguf",
      size: "4.37 GB",
      description: "Specialized for code generation"
    }
  ];

  // Function to extract model name from model info
  const getModelName = (modelInfo) => {
    if (!modelInfo) return null;

    // Try to extract model name from the path
    const lines = modelInfo.split('\n');
    for (const line of lines) {
      if (line.startsWith('Model:')) {
        const path = line.replace('Model:', '').trim();
        // Extract filename from path
        const filename = path.split('/').pop();
        if (filename) {
          // Remove .gguf extension and clean up the name
          return filename.replace('.gguf', '').replace(/[-_]/g, ' ');
        }
      }
    }
    return null;
  };

  useEffect(() => {
    console.log('App useEffect called - setting up socket and checking initialization');
    // Initialize socket connection
    const newSocket = io(API_BASE);
    setSocket(newSocket);

    // Track socket connection status
    newSocket.on('connect', () => {
      console.log('Socket connected');
      setServerConnected(true);
    });

    newSocket.on('disconnect', () => {
      console.log('Socket disconnected');
      setServerConnected(false);
    });

    newSocket.on('connect_error', () => {
      console.log('Socket connection error');
      setServerConnected(false);
    });

    // Socket event handlers
    newSocket.on('stream-chunk', (data) => {
      console.log('Received stream chunk:', data);

      // Reset the timeout on each chunk received
      if (streamingTimeoutRef.current) {
        clearTimeout(streamingTimeoutRef.current);
      }

      // Set a timeout to handle cases where [DONE] is never received
      streamingTimeoutRef.current = setTimeout(() => {
        console.log('Streaming timeout reached, forcing completion');
        console.log('Timeout content:', currentStreamingContentRef.current);

        // Capture the content before clearing
        const timeoutContent = currentStreamingContentRef.current;

        setIsStreaming(false);
        setIsLoading(false);
        setMessages(prev => {
          const newMessages = [...prev];
          if (newMessages.length > 0 && newMessages[newMessages.length - 1].type === 'assistant') {
            newMessages[newMessages.length - 1].content = timeoutContent;
            newMessages[newMessages.length - 1].isStreaming = false;
          }
          return newMessages;
        });
        setStreamingContent('');
        currentStreamingContentRef.current = '';
      }, 15000); // Increased to 15 seconds to give more time

      // Check if this is the completion signal with metrics
      if (data.text.startsWith('[DONE]')) {
        console.log('Received [DONE] signal, stopping streaming');
        console.log('Current isStreaming state before update:', isStreaming);
        console.log('Current isLoading state before update:', isLoading);

        // Capture the final content BEFORE clearing the ref
        const finalContent = currentStreamingContentRef.current;
        console.log('Final streaming content:', finalContent);

        // Clear the timeout since we received [DONE]
        if (streamingTimeoutRef.current) {
          clearTimeout(streamingTimeoutRef.current);
          streamingTimeoutRef.current = null;
        }

        // Always force the states to false when [DONE] is received
        setIsStreaming(false);
        setIsLoading(false);
        console.log('Forced isStreaming and isLoading to false.');

        // Parse metrics if available and valid
        const metricsPart = data.text.substring(6); // after '[DONE]'
        if (metricsPart && metricsPart.trim().startsWith('{')) {
          try {
            const metrics = JSON.parse(metricsPart.trim());
            console.log('Generation metrics:', metrics);
            // Add metrics to the last assistant message and mark as not streaming
            setMessages(prev => {
              const newMessages = [...prev];
              if (newMessages.length > 0 && newMessages[newMessages.length - 1].type === 'assistant') {
                const lastMessage = newMessages[newMessages.length - 1];
                // Update with final content
                lastMessage.content = finalContent;
                // Only set metrics if not already present
                if (!lastMessage.metrics) {
                  lastMessage.metrics = metrics;
                }
                // Mark as not streaming
                lastMessage.isStreaming = false;
              }
              return newMessages;
            });
          } catch (error) {
            console.error('Error parsing metrics:', error);
          }
        } else {
          // No metrics, just mark as not streaming
          setMessages(prev => {
            const newMessages = [...prev];
            if (newMessages.length > 0 && newMessages[newMessages.length - 1].type === 'assistant') {
              // Update with final content
              newMessages[newMessages.length - 1].content = finalContent;
              newMessages[newMessages.length - 1].isStreaming = false;
            }
            return newMessages;
          });
        }

        // Reset streaming content after updating the message
        currentStreamingContentRef.current = '';
        setStreamingContent('');
        return;
      }

      // Update the current streaming content
      console.log('Received chunk:', JSON.stringify(data.text), '| Length:', data.text.length);
      console.log('Previous content:', JSON.stringify(currentStreamingContentRef.current));
      currentStreamingContentRef.current += data.text;
      console.log('Updated streaming content:', JSON.stringify(currentStreamingContentRef.current));
      console.log('---');

            // Force immediate UI update with flushSync to bypass React batching
      flushSync(() => {
        setStreamingContent(currentStreamingContentRef.current);
        
        // Update cursor position based on current content
        const { cursorPos } = getStreamingDisplay(currentStreamingContentRef.current);
        setCursorPosition(cursorPos);
        
        setMessages(prev => {
          const newMessages = [...prev];
          
          // Check if we need to create a new assistant message
          if (newMessages.length === 0 || newMessages[newMessages.length - 1].type !== 'assistant') {
            // Create new assistant message with the complete content so far
            const newMessage = {
              id: `${Date.now()}-${messageIdCounter}-${Math.random()}`,
              type: 'assistant',
              content: currentStreamingContentRef.current,
              timestamp: new Date().toLocaleTimeString(),
              isError: false,
              isStreaming: true
            };
            setMessageIdCounter(prev => prev + 1);
            return [...newMessages, newMessage];
          } else {
            // Update the existing assistant message with the complete content
            return newMessages.map((msg, index) => 
              index === newMessages.length - 1 
                ? {
                    ...msg,
                    content: currentStreamingContentRef.current,
                    isStreaming: true,
                    // Force new object reference to ensure re-render
                    _updateTime: Date.now(),
                    _updateCount: (msg._updateCount || 0) + 1
                  }
                : msg
            );
          }
        });
      });
    });

    newSocket.on('stream-error', (data) => {
      // Clear the timeout
      if (streamingTimeoutRef.current) {
        clearTimeout(streamingTimeoutRef.current);
        streamingTimeoutRef.current = null;
      }

      setIsStreaming(false);
      setIsLoading(false);
      // Mark the last assistant message as not streaming if it exists
      setMessages(prev => {
        const newMessages = [...prev];
        if (newMessages.length > 0 && newMessages[newMessages.length - 1].type === 'assistant') {
          newMessages[newMessages.length - 1].isStreaming = false;
        }
        return newMessages;
      });
      addMessage('assistant', `Error: ${data.error}`, true);
    });

    newSocket.on('generation-stopped', () => {
      // Clear the timeout
      if (streamingTimeoutRef.current) {
        clearTimeout(streamingTimeoutRef.current);
        streamingTimeoutRef.current = null;
      }

      setIsStreaming(false);
      setIsLoading(false);
      // Mark the last assistant message as not streaming if it exists
      setMessages(prev => {
        const newMessages = [...prev];
        if (newMessages.length > 0 && newMessages[newMessages.length - 1].type === 'assistant') {
          newMessages[newMessages.length - 1].isStreaming = false;
        }
        return newMessages;
      });
    });

    // Download progress events
    newSocket.on('download-progress', (data) => {
      setDownloadProgress(data);
    });

    newSocket.on('download-complete', (data) => {
      setDownloadProgress({ progress: 100, status: 'Download complete!' });
      setTimeout(() => {
        setShowDownloadProgress(false);
        fetchAvailableModels(); // Refresh the models list
      }, 2000);
    });

    newSocket.on('download-error', (data) => {
      setDownloadProgress({ progress: 0, status: `Error: ${data.error}` });
      setTimeout(() => {
        setShowDownloadProgress(false);
      }, 3000);
    });

    newSocket.on('download-cancelled', (data) => {
      setDownloadProgress({ progress: 0, status: 'Download cancelled' });
      setTimeout(() => {
        setShowDownloadProgress(false);
      }, 2000);
    });

    // Check server status and auto-start if needed
    const checkAndStartServer = async () => {
      try {
        console.log('Checking server status...');
        const isRunning = await checkServerStatus();

        if (!isRunning) {
          console.log('Server not running, attempting to start...');
          await startServer();
        } else {
          // Server is running, check if model is initialized
          if (!isInitialized) {
            console.log('Server running but model not initialized, attempting auto-initialization...');
            await autoInitializeModel();
          }
        }
      } catch (error) {
        console.error('Server check/start error:', error);
      }
    };

    const autoInitializeModel = async () => {
      if (isInitializingRef.current) {
        console.log('Initialization already in progress, skipping auto-initialization.');
        return;
      }
      try {
        console.log('Auto-initializing default model...');
        setIsLoading(true);
        isInitializingRef.current = true;
        const initConfig = {
          modelPath: '/home/kilodin/local_llm_edge_pi5/models/tinyllama-1.1b-chat.gguf',
          contextSize: 2048,
          batchSize: 512,
          threads: 4,
          gpuLayers: 0,
          temperature: 0.7,  // Back to original temperature
          topP: 0.9,         // Back to original topP
          topK: 40,          // Back to original topK
          repeatPenalty: 1.1, // Slightly higher than original but not too aggressive
          seed: 42
        };
        console.log('Calling /api/initialize with', initConfig);
        const initResp = await axios.post(`${API_BASE}/api/initialize`, initConfig);
        console.log('Init response:', initResp.data);
        if (initResp.data.success) {
          setIsInitialized(true);
          setModelInfo(initResp.data.modelInfo);
        } else {
          console.error('Failed to initialize default model');
        }
      } catch (error) {
        console.error('Auto-initialization error:', error);
      } finally {
        setIsLoading(false);
        isInitializingRef.current = false;
      }
    };

    console.log('About to call checkAndStartServer()');
    checkAndStartServer();
    return () => {
      // Clear any pending timeout
      if (streamingTimeoutRef.current) {
        clearTimeout(streamingTimeoutRef.current);
        streamingTimeoutRef.current = null;
      }
      newSocket.close();
    };
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Debug useEffect to monitor state changes
  useEffect(() => {
    console.log('State changed - isStreaming:', isStreaming, 'isLoading:', isLoading);
  }, [isStreaming, isLoading]);

  // Periodic status check every 5 seconds
  useEffect(() => {
    const statusInterval = setInterval(async () => {
      const isRunning = await checkServerStatus();
      if (!isRunning && serverStatus === 'error' && !showRetryButton) {
        // Only attempt to start if we're in error state and no retry button is shown
        // This prevents constant restart attempts
        console.log('Periodic check: Server not running, attempting restart...');
        await startServer();
      }
    }, 500000);

    return () => clearInterval(statusInterval);
  }, [serverStatus, showRetryButton]);

  // Fetch available models when settings modal opens
  useEffect(() => {
    if (showSettings) {
      fetchAvailableModels();
    }
  }, [showSettings]);

  const checkServerStatus = async () => {
    try {
      const response = await axios.get(`${API_BASE}/api/health`);
      setServerConnected(true);
      setServerStatus('running');
      setShowRetryButton(false);
      setIsInitialized(response.data.initialized);
      setSystemInfo(response.data.systemInfo);

      if (response.data.initialized) {
        const modelResponse = await axios.get(`${API_BASE}/api/model-info`);
        setModelInfo(modelResponse.data.info);
      }
      return true;
    } catch (error) {
      console.error('Server not available:', error);
      setServerConnected(false);
      setServerStatus('error');
      setIsInitialized(false);
      return false;
    }
  };

  const startServer = async () => {
    try {
      setServerStatus('starting');
      setShowRetryButton(false);

      // Try to start the server via the helper
      const response = await axios.post('http://localhost:4000/start-server');

      if (response.data.success) {
        // Wait for server to start
        let attempts = 0;
        const maxAttempts = 30; // 30 seconds timeout

        while (attempts < maxAttempts) {
          await new Promise(resolve => setTimeout(resolve, 1000));

          try {
            const healthCheck = await axios.get(`${API_BASE}/api/health`);
            if (healthCheck.status === 200) {
              setServerStatus('running');
              setServerConnected(true);
              setIsInitialized(healthCheck.data.initialized);
              setSystemInfo(healthCheck.data.systemInfo);

              if (healthCheck.data.initialized) {
                const modelResponse = await axios.get(`${API_BASE}/api/model-info`);
                setModelInfo(modelResponse.data.info);
              }
              return true;
            }
          } catch (error) {
            // Server not ready yet
          }

          attempts++;
        }

        // Timeout
        setServerStatus('error');
        setShowRetryButton(true);
        return false;
      } else {
        setServerStatus('error');
        setShowRetryButton(true);
        return false;
      }
    } catch (error) {
      console.error('Failed to start server:', error);
      setServerStatus('error');
      setShowRetryButton(true);
      return false;
    }
  };

  const retryServerStart = async () => {
    await startServer();
  };

  const restartServer = async () => {
    try {
      // Stop the server first
      await axios.post('http://localhost:4000/stop-server');

      // Wait a moment for it to stop
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Start it again
      await startServer();
    } catch (error) {
      console.error('Failed to restart server:', error);
    }
  };

  const shutdownServer = async () => {
    try {
      await axios.post('http://localhost:4000/stop-server');
      setServerConnected(false);
      setIsInitialized(false);
      setServerStatus('stopped');
    } catch (error) {
      console.error('Failed to shutdown server:', error);
    }
  };

  const fetchAvailableModels = async () => {
    try {
      const response = await axios.get(`${API_BASE}/api/models`);
      setAvailableModels(response.data.models);
    } catch (error) {
      console.error('Failed to fetch available models:', error);
    }
  };

  const changeModel = async (modelPath) => {
    try {
      setIsLoading(true);
      const response = await axios.post(`${API_BASE}/api/change-model`, { modelPath });

      if (response.data.success) {
        setIsInitialized(true);
        setModelInfo(response.data.modelInfo);
        setShowSettings(false);
      } else {
        console.error('Failed to change model');
      }
    } catch (error) {
      console.error('Model change error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const startModelDownload = async (model) => {
    try {
      setSelectedModel(model);
      setShowModelDownload(false);
      setShowDownloadProgress(true);
      setDownloadProgress({ progress: 0, status: 'Starting download...' });

      const response = await axios.post(`${API_BASE}/api/download-model`, {
        modelUrl: model.url,
        filename: model.filename
      });

      if (response.data.success) {
        // Download started successfully, progress will be updated via WebSocket
        console.log('Download started');
      } else {
        console.error('Failed to start download');
        setShowDownloadProgress(false);
      }
    } catch (error) {
      console.error('Download error:', error);
      setShowDownloadProgress(false);
    }
  };

  const cancelDownload = async () => {
    if (!selectedModel) return;

    try {
      // Send cancel request via WebSocket
      if (socket.current) {
        socket.current.emit('cancel-download', { filename: selectedModel.filename });
      }

      // Also try HTTP endpoint as backup
      try {
        await axios.post(`${API_BASE}/api/cancel-download`, {
          filename: selectedModel.filename
        });
      } catch (error) {
        console.log('HTTP cancel failed, WebSocket cancel should work');
      }

    } catch (error) {
      console.error('Cancel download error:', error);
    }
  };

  const removeModel = async (model) => {
    if (!model || !model.filename) return;

    // Confirm deletion
    if (!window.confirm(`Are you sure you want to remove "${model.name}"? This action cannot be undone.`)) {
      return;
    }

    try {
      const response = await axios.delete(`${API_BASE}/api/remove-model`, {
        data: { filename: model.filename }
      });

      if (response.data.success) {
        // Refresh the models list
        fetchAvailableModels();
        console.log('Model removed successfully');
      } else {
        console.error('Failed to remove model');
      }
    } catch (error) {
      console.error('Remove model error:', error);
      if (error.response?.data?.error) {
        alert(`Error: ${error.response.data.error}`);
      } else {
        alert('Failed to remove model. Please try again.');
      }
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
      } else {
        console.error('Failed to initialize model');
      }
    } catch (error) {
      console.error(`Initialization error: ${error.response?.data?.error || error.message}`);
    } finally {
      setIsLoading(false);
      isInitializingRef.current = false; // Reset flag
    }
  };

  const updateParameters = async () => {
    try {
      await axios.post(`${API_BASE}/api/parameters`, parameters);
      setShowSettings(false);
    } catch (error) {
      console.error(`Parameter update error: ${error.response?.data?.error || error.message}`);
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
      // Mark the last assistant message as not streaming if it exists
      setMessages(prev => {
        const newMessages = [...prev];
        if (newMessages.length > 0 && newMessages[newMessages.length - 1].type === 'assistant') {
          newMessages[newMessages.length - 1].isStreaming = false;
        }
        return newMessages;
      });
    }, 30000); // 30 seconds timeout

    // Don't create empty assistant message here - let streaming create it

    try {
      // Use streaming for better UX
      socket.emit('generate-stream', {
        prompt: userMessage,
        systemPrompt: systemPrompt,
        maxTokens: 256
      });
    } catch (error) {
      if (streamingTimeoutRef.current) {
        clearTimeout(streamingTimeoutRef.current);
        streamingTimeoutRef.current = null;
      }
      setIsStreaming(false);
      setIsLoading(false);
      // Mark the last assistant message as not streaming if it exists
      setMessages(prev => {
        const newMessages = [...prev];
        if (newMessages.length > 0 && newMessages[newMessages.length - 1].type === 'assistant') {
          newMessages[newMessages.length - 1].isStreaming = false;
        }
        return newMessages;
      });
      addMessage('assistant', `Error: ${error.message}`, true);
    }
  };

  const stopGeneration = () => {
    if (socket) {
      socket.emit('stop-generation');
    }
    // Mark the last assistant message as not streaming if it exists
    setMessages(prev => {
      const newMessages = [...prev];
      if (newMessages.length > 0 && newMessages[newMessages.length - 1].type === 'assistant') {
        newMessages[newMessages.length - 1].isStreaming = false;
      }
      return newMessages;
    });
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
            <div className="status-indicators">
              <div className="status-item">
                <div className={`status-light ${serverConnected ? 'connected' : 'disconnected'}`}></div>
                <span className="status-label">{serverConnected ? 'Server Connected' : 'Server Disconnected'}</span>
              </div>
              <div className="status-item">
                <div className={`status-light ${isInitialized ? 'connected' : 'disconnected'}`}></div>
                <span className="status-label">{isInitialized ? 'Model Initialized' : 'Model Uninitialized'}</span>
              </div>
              {serverConnected && (
                <div className="server-controls">
                  <button
                    className="btn btn-secondary btn-small"
                    onClick={restartServer}
                    title="Restart LLM Server"
                  >
                    <RotateCcw size={14} />
                    Restart
                  </button>
                  <button
                    className="btn btn-danger btn-small"
                    onClick={shutdownServer}
                    title="Shutdown LLM Server"
                  >
                    <Power size={14} />
                    Shutdown
                  </button>
                </div>
              )}
            </div>
            {isInitialized && modelInfo && (
              <div className="current-model" onClick={() => setShowSettings(true)}>
                <span className="model-label">Active Model:</span>
                <span className="header-model-name">{getModelName(modelInfo)}</span>
              </div>
            )}
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

                {/* Server Status Display */}
                <div className="server-status">
                  {serverStatus === 'checking' && (
                    <div className="status-loading">
                      <div className="loading-spinner"></div>
                      <span>Checking server status...</span>
                    </div>
                  )}

                  {serverStatus === 'starting' && (
                    <div className="status-loading">
                      <div className="loading-spinner"></div>
                      <span>Starting server...</span>
                    </div>
                  )}

                  {serverStatus === 'running' && !isInitialized && (
                    <div className="status-loading">
                      <div className="loading-spinner"></div>
                      <span>Initializing model...</span>
                    </div>
                  )}

                  {serverStatus === 'error' && showRetryButton && (
                    <div className="status-error">
                      <span>Server connection failed</span>
                      <button
                        className="btn btn-primary"
                        onClick={retryServerStart}
                      >
                        <Play size={16} />
                        Retry Connection
                      </button>
                    </div>
                  )}
                </div>
                {systemInfo && (
                  <div className="system-info">
                    <h3>System Information</h3>
                    <table className="system-info-table">
                      <tbody>
                        {parseSystemInfo(systemInfo).map(({ key, value }) => (
                          <tr key={key}>
                            <td className="sysinfo-key">{key}:</td>
                            <td className="sysinfo-value">{value}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            ) : (
              messages.map((message) => (
                <div
                  key={message.id}
                  className={`message ${message.type} ${message.isError ? 'error' : ''} ${message.isStreaming ? 'streaming' : ''}`}
                >
                  <div className="message-avatar">
                    {message.type === 'user' ? <User size={20} /> : <Bot size={20} />}
                  </div>
                  <div className="message-content">
                    {message.type === 'assistant' ? (
                      <>
                        {message.content && message.content.trim() ? (
                          message.isStreaming ? (
                            // For streaming messages, render with dynamic cursor
                            <div className="streaming-content">
                              {(() => {
                                const { displayText, cursorPos } = getStreamingDisplay(message.content);
                                return (
                                  <>
                                    <span>{displayText.slice(0, cursorPos)}</span>
                                    <span className="streaming-cursor">|</span>
                                    <span>{displayText.slice(cursorPos)}</span>
                                  </>
                                );
                              })()}
                            </div>
                          ) : (
                            // For completed messages, render normally
                            <ReactMarkdown>{message.content}</ReactMarkdown>
                          )
                        ) : (
                          <p style={{ color: 'red' }}>No content available (length: {message.content?.length || 0})</p>
                        )}
                        {message.metrics && <GenerationMetrics metrics={message.metrics} />}
                      </>
                    ) : (
                      <p>{message.content}</p>
                    )}
                    <span className="message-time">{message.timestamp}</span>
                    {message.type === 'assistant' && (
                      <span className="generation-status">
                        {message.isStreaming ? (
                          <span className="generating-text">
                            <span className="status-icon loading-icon">⟳</span>
                            Generating<span className="animated-dots">
                              <span className="dot">.</span>
                              <span className="dot">.</span>
                              <span className="dot">.</span>
                            </span>
                          </span>
                        ) : message.content && message.content.trim() ? (
                          <span className="completed-text">
                            <span className="status-icon completed-icon">✓</span>
                            Generation Completed
                          </span>
                        ) : ''}
                      </span>
                    )}
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
                  onChange={(e) => setInitConfig(prev => ({ ...prev, modelPath: e.target.value }))}
                  placeholder="/path/to/your/model.gguf"
                />
              </div>
              <div className="form-row">
                <div className="form-group">
                  <label>Context Size:</label>
                  <input
                    type="number"
                    value={initConfig.contextSize}
                    onChange={(e) => setInitConfig(prev => ({ ...prev, contextSize: parseInt(e.target.value) }))}
                  />
                </div>
                <div className="form-group">
                  <label>Threads:</label>
                  <input
                    type="number"
                    value={initConfig.threads}
                    onChange={(e) => setInitConfig(prev => ({ ...prev, threads: parseInt(e.target.value) }))}
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
                    onChange={(e) => setInitConfig(prev => ({ ...prev, temperature: parseFloat(e.target.value) }))}
                  />
                </div>
                <div className="form-group">
                  <label>Top-P:</label>
                  <input
                    type="number"
                    step="0.1"
                    value={initConfig.topP}
                    onChange={(e) => setInitConfig(prev => ({ ...prev, topP: parseFloat(e.target.value) }))}
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
              <div className="info-sections-row">
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

              {availableModels.length > 0 && (
                <div className="model-selection-section">
                  <h3>Available Models</h3>
                  <div className="models-grid">
                    {availableModels.map((model) => (
                      <div
                        key={model.path}
                        className={`model-card ${modelInfo && modelInfo.includes(model.path) ? 'active' : ''}`}
                        onClick={() => changeModel(model.path)}
                      >
                        <div className="selector-model-name">{model.name}</div>
                        <div className="model-size">{model.size}</div>
                        {modelInfo && modelInfo.includes(model.path) && (
                          <div className="active-indicator">Active</div>
                        )}
                      </div>
                    ))}
                  </div>
                  <button
                    className="btn btn-secondary download-model-btn"
                    onClick={() => setShowModelDownload(true)}
                  >
                    <Plus size={16} />
                    Download New Model
                  </button>
                </div>
              )}

              <div className="parameters-section">
                <h3>System Prompt</h3>
                <div className="form-group">
                  <label>Personality & Behavior:</label>
                  <textarea
                    value={systemPrompt}
                    onChange={(e) => setSystemPrompt(e.target.value)}
                    placeholder="Enter system prompt to control the model's personality and behavior..."
                    rows={4}
                    style={{ resize: 'vertical', minHeight: '100px', width: '100%' }}
                  />
                  <small style={{ color: '#666', marginTop: '0.5rem', display: 'block' }}>
                    This prompt will be sent to the model before each conversation to control its personality, tone, and behavior.
                  </small>
                </div>
              </div>

              <div className="parameters-section">
                <h3>Sampling Parameters</h3>
                <div className="form-row">
                  <div className="form-group">
                    <label>Temperature:</label>
                    <input
                      type="number"
                      step="0.1"
                      value={parameters.temperature}
                      onChange={(e) => setParameters(prev => ({ ...prev, temperature: parseFloat(e.target.value) }))}
                    />
                  </div>
                  <div className="form-group">
                    <label>Top-P:</label>
                    <input
                      type="number"
                      step="0.1"
                      value={parameters.topP}
                      onChange={(e) => setParameters(prev => ({ ...prev, topP: parseFloat(e.target.value) }))}
                    />
                  </div>
                </div>
                <div className="form-row">
                  <div className="form-group">
                    <label>Top-K:</label>
                    <input
                      type="number"
                      value={parameters.topK}
                      onChange={(e) => setParameters(prev => ({ ...prev, topK: parseInt(e.target.value) }))}
                    />
                  </div>
                  <div className="form-group">
                    <label>Min-P:</label>
                    <input
                      type="number"
                      step="0.1"
                      value={parameters.minP}
                      onChange={(e) => setParameters(prev => ({ ...prev, minP: parseFloat(e.target.value) }))}
                    />
                  </div>
                </div>
                <div className="form-row">
                  <div className="form-group">
                    <label>Typical-P:</label>
                    <input
                      type="number"
                      step="0.1"
                      value={parameters.typicalP}
                      onChange={(e) => setParameters(prev => ({ ...prev, typicalP: parseFloat(e.target.value) }))}
                    />
                  </div>
                  <div className="form-group">
                    <label>TFS-Z:</label>
                    <input
                      type="number"
                      step="0.1"
                      value={parameters.tfsZ}
                      onChange={(e) => setParameters(prev => ({ ...prev, tfsZ: parseFloat(e.target.value) }))}
                    />
                  </div>
                </div>
                <div className="form-row">
                  <div className="form-group">
                    <label>Top-A:</label>
                    <input
                      type="number"
                      step="0.1"
                      value={parameters.topA}
                      onChange={(e) => setParameters(prev => ({ ...prev, topA: parseFloat(e.target.value) }))}
                    />
                  </div>
                </div>
              </div>

              <div className="parameters-section">
                <h3>Penalty Parameters</h3>
                <div className="form-row">
                  <div className="form-group">
                    <label>Repeat Penalty:</label>
                    <input
                      type="number"
                      step="0.1"
                      value={parameters.repeatPenalty}
                      onChange={(e) => setParameters(prev => ({ ...prev, repeatPenalty: parseFloat(e.target.value) }))}
                    />
                  </div>
                  <div className="form-group">
                    <label>Repeat Penalty Last-N:</label>
                    <input
                      type="number"
                      value={parameters.repeatPenaltyLastN}
                      onChange={(e) => setParameters(prev => ({ ...prev, repeatPenaltyLastN: parseInt(e.target.value) }))}
                    />
                  </div>
                </div>
                <div className="form-row">
                  <div className="form-group">
                    <label>Frequency Penalty:</label>
                    <input
                      type="number"
                      step="0.1"
                      value={parameters.frequencyPenalty}
                      onChange={(e) => setParameters(prev => ({ ...prev, frequencyPenalty: parseFloat(e.target.value) }))}
                    />
                  </div>
                  <div className="form-group">
                    <label>Presence Penalty:</label>
                    <input
                      type="number"
                      step="0.1"
                      value={parameters.presencePenalty}
                      onChange={(e) => setParameters(prev => ({ ...prev, presencePenalty: parseFloat(e.target.value) }))}
                    />
                  </div>
                </div>
              </div>

              <div className="parameters-section">
                <h3>Mirostat Parameters</h3>
                <div className="form-row">
                  <div className="form-group">
                    <label>Mirostat Tau:</label>
                    <input
                      type="number"
                      step="0.1"
                      value={parameters.mirostatTau}
                      onChange={(e) => setParameters(prev => ({ ...prev, mirostatTau: parseFloat(e.target.value) }))}
                    />
                  </div>
                  <div className="form-group">
                    <label>Mirostat Eta:</label>
                    <input
                      type="number"
                      step="0.1"
                      value={parameters.mirostatEta}
                      onChange={(e) => setParameters(prev => ({ ...prev, mirostatEta: parseFloat(e.target.value) }))}
                    />
                  </div>
                </div>
                <div className="form-row">
                  <div className="form-group">
                    <label>Mirostat M:</label>
                    <input
                      type="number"
                      value={parameters.mirostatM}
                      onChange={(e) => setParameters(prev => ({ ...prev, mirostatM: parseInt(e.target.value) }))}
                    />
                  </div>
                </div>
              </div>

              <div className="parameters-section">
                <h3>RoPE Parameters</h3>
                <div className="form-row">
                  <div className="form-group">
                    <label>RoPE Freq Base:</label>
                    <input
                      type="number"
                      step="0.1"
                      value={parameters.ropeFreqBase}
                      onChange={(e) => setParameters(prev => ({ ...prev, ropeFreqBase: parseFloat(e.target.value) }))}
                    />
                  </div>
                  <div className="form-group">
                    <label>RoPE Freq Scale:</label>
                    <input
                      type="number"
                      step="0.1"
                      value={parameters.ropeFreqScale}
                      onChange={(e) => setParameters(prev => ({ ...prev, ropeFreqScale: parseFloat(e.target.value) }))}
                    />
                  </div>
                </div>
              </div>

              <div className="parameters-section">
                <h3>YaRN Parameters</h3>
                <div className="form-row">
                  <div className="form-group">
                    <label>YaRN Ext Factor:</label>
                    <input
                      type="number"
                      step="0.1"
                      value={parameters.yarnExtFactor}
                      onChange={(e) => setParameters(prev => ({ ...prev, yarnExtFactor: parseFloat(e.target.value) }))}
                    />
                  </div>
                  <div className="form-group">
                    <label>YaRN Attn Factor:</label>
                    <input
                      type="number"
                      step="0.1"
                      value={parameters.yarnAttnFactor}
                      onChange={(e) => setParameters(prev => ({ ...prev, yarnAttnFactor: parseFloat(e.target.value) }))}
                    />
                  </div>
                </div>
                <div className="form-row">
                  <div className="form-group">
                    <label>YaRN Beta Fast:</label>
                    <input
                      type="number"
                      step="0.1"
                      value={parameters.yarnBetaFast}
                      onChange={(e) => setParameters(prev => ({ ...prev, yarnBetaFast: parseFloat(e.target.value) }))}
                    />
                  </div>
                  <div className="form-group">
                    <label>YaRN Beta Slow:</label>
                    <input
                      type="number"
                      step="0.1"
                      value={parameters.yarnBetaSlow}
                      onChange={(e) => setParameters(prev => ({ ...prev, yarnBetaSlow: parseFloat(e.target.value) }))}
                    />
                  </div>
                </div>
                <div className="form-row">
                  <div className="form-group">
                    <label>YaRN Orig Ctx:</label>
                    <input
                      type="number"
                      value={parameters.yarnOrigCtx}
                      onChange={(e) => setParameters(prev => ({ ...prev, yarnOrigCtx: parseInt(e.target.value) }))}
                    />
                  </div>
                </div>
              </div>

              <div className="parameters-section">
                <h3>Performance Parameters</h3>
                <div className="form-row">
                  <div className="form-group">
                    <label>Defrag Threshold:</label>
                    <input
                      type="number"
                      step="0.1"
                      value={parameters.defragThold}
                      onChange={(e) => setParameters(prev => ({ ...prev, defragThold: parseFloat(e.target.value) }))}
                    />
                  </div>
                  <div className="form-group">
                    <label>Threads Batch:</label>
                    <input
                      type="number"
                      value={parameters.threadsBatch}
                      onChange={(e) => setParameters(prev => ({ ...prev, threadsBatch: parseInt(e.target.value) }))}
                    />
                  </div>
                </div>
                <div className="form-row">
                  <div className="form-group">
                    <label>Ubatch Size:</label>
                    <input
                      type="number"
                      value={parameters.ubatchSize}
                      onChange={(e) => setParameters(prev => ({ ...prev, ubatchSize: parseInt(e.target.value) }))}
                    />
                  </div>
                </div>
                <div className="form-row">
                  <div className="form-group">
                    <label>
                      <input
                        type="checkbox"
                        checked={parameters.flashAttn}
                        onChange={(e) => setParameters(prev => ({ ...prev, flashAttn: e.target.checked }))}
                      />
                      Flash Attention
                    </label>
                  </div>
                  <div className="form-group">
                    <label>
                      <input
                        type="checkbox"
                        checked={parameters.offloadKqv}
                        onChange={(e) => setParameters(prev => ({ ...prev, offloadKqv: e.target.checked }))}
                      />
                      Offload KQV
                    </label>
                  </div>
                </div>
                <div className="form-row">
                  <div className="form-group">
                    <label>
                      <input
                        type="checkbox"
                        checked={parameters.embeddings}
                        onChange={(e) => setParameters(prev => ({ ...prev, embeddings: e.target.checked }))}
                      />
                      Extract Embeddings
                    </label>
                  </div>
                </div>
              </div>
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

      {/* Model Download Selection Modal */}
      {showModelDownload && (
        <div className="modal-overlay">
          <div className="modal">
            <div className="modal-header">
              <h2>Download New Model</h2>
              <button
                className="btn-close"
                onClick={() => setShowModelDownload(false)}
              >
                <X size={20} />
              </button>
            </div>
            <div className="modal-content">
              <div className="models-download-grid">
                {supportedModels.map((model) => {
                  const isDownloaded = availableModels.some(available =>
                    available.name === model.filename
                  );

                  return (
                    <div
                      key={model.filename}
                      className={`download-model-card ${isDownloaded ? 'downloaded' : ''}`}
                      onClick={!isDownloaded ? () => startModelDownload(model) : undefined}
                    >
                      <div className="download-model-name">{model.name}</div>
                      <div className="download-model-size">{model.size}</div>
                      <div className="download-model-description">{model.description}</div>
                      {isDownloaded && (
                        <>
                          <div className="downloaded-indicator">Already in Model DB</div>
                          <button
                            className="btn btn-danger btn-remove-model"
                            onClick={(e) => {
                              e.stopPropagation();
                              removeModel(model);
                            }}
                            title="Remove model"
                          >
                            Remove
                          </button>
                        </>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
            <div className="modal-footer">
              <button
                className="btn btn-secondary"
                onClick={() => setShowModelDownload(false)}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Download Progress Modal */}
      {showDownloadProgress && (
        <div className="modal-overlay">
          <div className="modal download-progress-modal">
            <div className="modal-header">
              <h2>Downloading Model</h2>
            </div>
            <div className="modal-content">
              {selectedModel && (
                <div className="download-info">
                  <h3>{selectedModel.name}</h3>
                  <p>Size: {selectedModel.size}</p>
                </div>
              )}
              <div className="download-progress">
                <div className="progress-bar">
                  <div
                    className="progress-fill"
                    style={{ width: `${downloadProgress.progress || 0}%` }}
                  ></div>
                </div>
                <div className="progress-text">
                  {downloadProgress.status || 'Preparing download...'}
                </div>
                <div className="progress-details">
                  <div className="progress-percentage">
                    {Math.round(downloadProgress.progress || 0)}%
                  </div>
                  {downloadProgress.speed && (
                    <div className="progress-speed">
                      Speed: {downloadProgress.speed}
                    </div>
                  )}
                  {downloadProgress.eta && (
                    <div className="progress-eta">
                      ETA: {downloadProgress.eta}
                    </div>
                  )}
                </div>
              </div>
            </div>
            <div className="modal-footer">
              <button
                className="btn btn-danger"
                onClick={cancelDownload}
                disabled={downloadProgress.status?.includes('complete') || downloadProgress.status?.includes('cancelled') || downloadProgress.status?.includes('Error')}
              >
                Cancel Download
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App; 