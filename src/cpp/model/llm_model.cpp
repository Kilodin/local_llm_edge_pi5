#include "llm_model.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <random>
#include <cmath>

namespace local_llm {

LLMModel::LLMModel() : ctx_(nullptr), model_(nullptr) {}

LLMModel::~LLMModel() {
    if (ctx_) {
        llama_free(ctx_);
    }
    if (model_) {
        llama_model_free(model_);
    }
    llama_backend_free();
}

bool LLMModel::initialize(const ModelConfig& config) {
    config_ = config;
    
    // Initialize llama.cpp backend
    llama_backend_init();
    
    // Load model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = config.gpu_layers;
    
    model_ = llama_model_load_from_file(config.model_path.c_str(), model_params);
    if (!model_) {
        std::cerr << "Failed to load model: " << config.model_path << std::endl;
        return false;
    }
    
    // Create context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = config.context_size;
    ctx_params.n_batch = config.batch_size;
    ctx_params.n_threads = config.threads;
    ctx_params.n_threads_batch = config.threads;
    
    ctx_ = llama_init_from_model(model_, ctx_params);
    if (!ctx_) {
        std::cerr << "Failed to create context" << std::endl;
        return false;
    }
    
    std::cout << "Model loaded successfully: " << config.model_path << std::endl;
    return true;
}

std::string LLMModel::generate(const std::string& prompt, int max_tokens) {
    return generate_internal(prompt, max_tokens);
}

void LLMModel::generate_stream(const std::string& prompt, 
                              std::function<void(const std::string&)> callback,
                              int max_tokens) {
    // For now, use simple generation without streaming
    std::string result = generate_internal(prompt, max_tokens);
    callback(result);
}

std::string LLMModel::generate_internal(const std::string& prompt, int max_tokens) {
    if (!ctx_ || !model_) return "Model not loaded";
    
    // Reset context state for new generation
    llama_kv_self_clear(ctx_);
    
    // Reset any internal state
    llama_reset_timings(ctx_);
    
    // Tokenize prompt
    std::vector<llama_token> input_tokens = tokenize(prompt);
    if (input_tokens.empty()) return "Tokenization failed";

    // Add BOS if needed
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    if (llama_vocab_get_add_bos(vocab)) {
        input_tokens.insert(input_tokens.begin(), llama_vocab_bos(vocab));
    }

    // Simple approach: evaluate all input tokens at once
    std::vector<llama_token> output_tokens;
    
    try {
        // Process input tokens in batches
        for (size_t i = 0; i < input_tokens.size(); i += config_.batch_size) {
            int n_eval = std::min((int)input_tokens.size() - (int)i, config_.batch_size);
            llama_batch batch = llama_batch_get_one(input_tokens.data() + i, n_eval);
            int ret = llama_decode(ctx_, batch);
            if (ret != 0) {
                return "Failed to decode input tokens";
            }
        }

        // Generate new tokens
        for (int i = 0; i < max_tokens; ++i) {
            // Get logits for the last token
            float* logits = llama_get_logits(ctx_);
            if (!logits) {
                return "Failed to get logits";
            }
            
            int vocab_size = llama_vocab_n_tokens(vocab);
            std::vector<float> logits_vec(logits, logits + vocab_size);
            
            // Sample next token
            llama_token next_token = sample_next_token(logits_vec);
            
            // Check for EOS
            if (next_token == llama_vocab_eos(vocab)) {
                break;
            }
            
            output_tokens.push_back(next_token);
            
            // Feed the new token back to the model
            llama_batch batch = llama_batch_get_one(&next_token, 1);
            int ret = llama_decode(ctx_, batch);
            if (ret != 0) {
                return "Failed to decode generated token";
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception during generation: " << e.what() << std::endl;
        return "Error during generation: " + std::string(e.what());
    } catch (...) {
        std::cerr << "Unknown exception during generation" << std::endl;
        return "Unknown error during generation";
    }

    // Detokenize the generated tokens
    return detokenize(output_tokens);
}

std::vector<llama_token> LLMModel::tokenize(const std::string& text) {
    std::vector<llama_token> tokens;
    tokens.resize(text.size() + 1);
    
    // Get vocabulary from model
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    
    int n_tokens = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), true, false);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), true, false);
    }
    
    tokens.resize(n_tokens);
    return tokens;
}

std::string LLMModel::detokenize(const std::vector<llama_token>& tokens) {
    std::string result;
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    
    for (llama_token token : tokens) {
        char piece[8];
        int n_piece = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, false);
        if (n_piece > 0) {
            result += std::string(piece, n_piece);
        }
    }
    return result;
}

llama_token LLMModel::sample_next_token(const std::vector<float>& logits) {
    if (logits.empty()) return 0;
    
    // Apply temperature
    std::vector<float> logits_copy = logits;
    float temp = config_.temperature;
    if (temp > 0.0f) {
        for (float& logit : logits_copy) {
            logit /= temp;
        }
    }
    
    // Apply top-k sampling
    int top_k = config_.top_k;
    if (top_k > 0 && top_k < (int)logits_copy.size()) {
        std::vector<std::pair<float, int>> logits_with_indices;
        for (int i = 0; i < (int)logits_copy.size(); ++i) {
            logits_with_indices.emplace_back(logits_copy[i], i);
        }
        
        // Sort by logit value (descending)
        std::partial_sort(logits_with_indices.begin(), 
                         logits_with_indices.begin() + top_k,
                         logits_with_indices.end(),
                         [](const auto& a, const auto& b) {
                             return a.first > b.first;
                         });
        
        // Keep only top-k
        logits_with_indices.resize(top_k);
        
        // Convert back to logits vector (zero out non-top-k)
        std::fill(logits_copy.begin(), logits_copy.end(), -INFINITY);
        for (const auto& pair : logits_with_indices) {
            logits_copy[pair.second] = pair.first;
        }
    }
    
    // Convert to probabilities using softmax
    float max_logit = *std::max_element(logits_copy.begin(), logits_copy.end());
    std::vector<float> probs(logits_copy.size());
    float sum = 0.0f;
    
    for (size_t i = 0; i < logits_copy.size(); ++i) {
        probs[i] = expf(logits_copy[i] - max_logit);
        sum += probs[i];
    }
    
    if (sum > 0.0f) {
        for (float& prob : probs) {
            prob /= sum;
        }
    }
    
    // Sample from the distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(gen);
    
    float cumsum = 0.0f;
    for (int i = 0; i < (int)probs.size(); ++i) {
        cumsum += probs[i];
        if (r <= cumsum) {
            return i;
        }
    }
    
    // Fallback to argmax
    return std::max_element(logits_copy.begin(), logits_copy.end()) - logits_copy.begin();
}

std::string LLMModel::get_model_info() const {
    if (!model_) {
        return "No model loaded";
    }
    
    std::ostringstream oss;
    oss << "Model: " << config_.model_path << "\n";
    oss << "Context size: " << config_.context_size << "\n";
    oss << "Batch size: " << config_.batch_size << "\n";
    oss << "Threads: " << config_.threads << "\n";
    oss << "GPU layers: " << config_.gpu_layers << "\n";
    oss << "Temperature: " << config_.temperature << "\n";
    oss << "Top-p: " << config_.top_p << "\n";
    oss << "Top-k: " << config_.top_k << "\n";
    oss << "Repeat penalty: " << config_.repeat_penalty << "\n";
    
    return oss.str();
}

} // namespace local_llm