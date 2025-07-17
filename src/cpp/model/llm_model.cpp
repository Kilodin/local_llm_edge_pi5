#include "llm_model.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <random>
#include <cmath>

namespace local_llm {

// Static flag to ensure backend is initialized only once
static bool backend_initialized = false;

LLMModel::LLMModel() : ctx_(nullptr), model_(nullptr) {
    std::cerr << "[LLMModel] Constructor called, this=" << this << std::endl;
}

LLMModel::~LLMModel() {
    std::cerr << "[LLMModel] Destructor called, this=" << this << std::endl;
    if (ctx_) {
        std::cerr << "[LLMModel] Freeing context in destructor" << std::endl;
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_) {
        std::cerr << "[LLMModel] Freeing model in destructor" << std::endl;
        llama_model_free(model_);
        model_ = nullptr;
    }
    // Don't call llama_backend_free() here - it should only be called once at application shutdown
}

void LLMModel::cleanup_backend() {
    if (backend_initialized) {
        std::cerr << "[LLMModel] Cleaning up llama.cpp backend" << std::endl;
        llama_backend_free();
        backend_initialized = false;
    }
}

bool LLMModel::initialize(const ModelConfig& config) {
    std::cerr << "[LLMModel] initialize called, this=" << this << std::endl;
    config_ = config;
    
    // Initialize llama.cpp backend only once
    if (!backend_initialized) {
        std::cerr << "[LLMModel] Initializing llama.cpp backend" << std::endl;
        llama_backend_init();
        backend_initialized = true;
    }
    
    // Load model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = config.gpu_layers;
    
    model_ = llama_model_load_from_file(config.model_path.c_str(), model_params);
    if (model_) {
        std::cerr << "[LLMModel] Model loaded, model_=" << model_ << std::endl;
    }
    if (!model_) {
        std::cerr << "Failed to load model: " << config.model_path << std::endl;
        return false;
    }
    
    // Do NOT create context here anymore
    std::cout << "Model loaded successfully: " << config.model_path << std::endl;
    return true;
}

std::string LLMModel::generate(const std::string& prompt, int max_tokens) {
    return generate_internal(prompt, max_tokens);
}

void LLMModel::generate_stream(const std::string& prompt, 
                              std::function<void(const std::string&)> callback,
                              int max_tokens) {
    std::cerr << "[LLMModel] generate_stream called, this=" << this << ", model_=" << model_ << std::endl;
    if (!model_) {
        callback("Model not loaded");
        return;
    }
    
    // Ensure any existing context is freed before creating a new one
    if (ctx_) {
        std::cerr << "[LLMModel] Freeing existing context before creating new one" << std::endl;
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    
    // Create context for this generation
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = config_.context_size;
    ctx_params.n_batch = config_.batch_size;
    ctx_params.n_threads = config_.threads;
    ctx_params.n_threads_batch = config_.threads;
    ctx_ = llama_init_from_model(model_, ctx_params);
    if (!ctx_) {
        callback("Failed to create context");
        return;
    }
    std::vector<llama_token> input_tokens = tokenize(prompt);
    std::cerr << "[LLMModel] Input tokens size: " << input_tokens.size() << std::endl;
    if (input_tokens.empty()) {
        callback("Tokenization failed");
        if (ctx_) { llama_free(ctx_); ctx_ = nullptr; }
        return;
    }
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    if (llama_vocab_get_add_bos(vocab)) {
        input_tokens.insert(input_tokens.begin(), llama_vocab_bos(vocab));
        std::cerr << "[LLMModel] Added BOS token, total tokens: " << input_tokens.size() << std::endl;
    }
    try {
        for (size_t i = 0; i < input_tokens.size(); i += config_.batch_size) {
            int n_eval = std::min((int)input_tokens.size() - (int)i, config_.batch_size);
            llama_batch batch = llama_batch_get_one(input_tokens.data() + i, n_eval);
            int ret = llama_decode(ctx_, batch);
            if (ret != 0) {
                callback("Failed to decode input tokens");
                if (ctx_) { llama_free(ctx_); ctx_ = nullptr; }
                return;
            }
        }
        std::string full_result;
        std::cerr << "[LLMModel] Starting generation loop, max_tokens: " << max_tokens << std::endl;
        for (int i = 0; i < max_tokens; ++i) {
            float* logits = llama_get_logits(ctx_);
            if (!logits) {
                callback("Failed to get logits");
                if (ctx_) { llama_free(ctx_); ctx_ = nullptr; }
                return;
            }
            int vocab_size = llama_vocab_n_tokens(vocab);
            std::vector<float> logits_vec(logits, logits + vocab_size);
            llama_token next_token = sample_next_token(logits_vec);
            std::cerr << "[LLMModel] Generated token " << i << ": " << next_token << " (EOS: " << llama_vocab_eos(vocab) << ")" << std::endl;
            if (next_token == llama_vocab_eos(vocab)) {
                std::cerr << "[LLMModel] Hit EOS token, stopping generation" << std::endl;
                break;
            }
            char piece[8];
            int n_piece = llama_token_to_piece(vocab, next_token, piece, sizeof(piece), 0, false);
            if (n_piece > 0) {
                std::string token_text(piece, n_piece);
                full_result += token_text;
                callback(token_text); // Stream the token
            }
            llama_batch batch = llama_batch_get_one(&next_token, 1);
            int ret = llama_decode(ctx_, batch);
            if (ret != 0) {
                callback("Failed to decode generated token");
                if (ctx_) { llama_free(ctx_); ctx_ = nullptr; }
                return;
            }
        }
        std::cerr << "[LLMModel] generate_stream result: " << full_result << std::endl;
        callback("[DONE]");
        if (ctx_) { llama_free(ctx_); ctx_ = nullptr; }
    } catch (const std::exception& e) {
        std::cerr << "Exception during streaming generation: " << e.what() << std::endl;
        callback("Error during generation: " + std::string(e.what()));
        if (ctx_) { llama_free(ctx_); ctx_ = nullptr; }
    } catch (...) {
        std::cerr << "Unknown exception during streaming generation" << std::endl;
        callback("Unknown error during generation");
        if (ctx_) { llama_free(ctx_); ctx_ = nullptr; }
    }
}

std::string LLMModel::generate_internal(const std::string& prompt, int max_tokens) {
    std::cerr << "[LLMModel] generate_internal called, this=" << this << ", model_=" << model_ << std::endl;
    if (!model_) return "Model not loaded";
    
    // Ensure any existing context is freed before creating a new one
    if (ctx_) {
        std::cerr << "[LLMModel] Freeing existing context before creating new one" << std::endl;
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    
    // Create context for this generation
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = config_.context_size;
    ctx_params.n_batch = config_.batch_size;
    ctx_params.n_threads = config_.threads;
    ctx_params.n_threads_batch = config_.threads;
    ctx_ = llama_init_from_model(model_, ctx_params);
    if (!ctx_) {
        return "Failed to create context";
    }
    
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
    std::string result;
    try {
        // Process input tokens in batches
        for (size_t i = 0; i < input_tokens.size(); i += config_.batch_size) {
            int n_eval = std::min((int)input_tokens.size() - (int)i, config_.batch_size);
            llama_batch batch = llama_batch_get_one(input_tokens.data() + i, n_eval);
            int ret = llama_decode(ctx_, batch);
            if (ret != 0) {
                if (ctx_) { llama_free(ctx_); ctx_ = nullptr; }
                return "Failed to decode input tokens";
            }
        }

        // Generate new tokens
        for (int i = 0; i < max_tokens; ++i) {
            float* logits = llama_get_logits(ctx_);
            if (!logits) {
                if (ctx_) { llama_free(ctx_); ctx_ = nullptr; }
                return "Failed to get logits";
            }
            int vocab_size = llama_vocab_n_tokens(vocab);
            std::vector<float> logits_vec(logits, logits + vocab_size);
            llama_token next_token = sample_next_token(logits_vec);
            if (next_token == llama_vocab_eos(vocab)) {
                break;
            }
            output_tokens.push_back(next_token);
            llama_batch batch = llama_batch_get_one(&next_token, 1);
            int ret = llama_decode(ctx_, batch);
            if (ret != 0) {
                if (ctx_) { llama_free(ctx_); ctx_ = nullptr; }
                return "Failed to decode generated token";
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception during generation: " << e.what() << std::endl;
        if (ctx_) { llama_free(ctx_); ctx_ = nullptr; }
        return "Error during generation: " + std::string(e.what());
    } catch (...) {
        std::cerr << "Unknown exception during generation" << std::endl;
        if (ctx_) { llama_free(ctx_); ctx_ = nullptr; }
        return "Unknown error during generation";
    }
    result = detokenize(output_tokens);
    std::cerr << "[LLMModel] generate_internal result: " << result << std::endl;
    if (ctx_) { llama_free(ctx_); ctx_ = nullptr; }
    return result;
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