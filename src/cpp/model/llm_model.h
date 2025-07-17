#pragma once

#include <string>
#include <memory>
#include <vector>
#include <functional>
#include "llama.h"

namespace local_llm {

struct ModelConfig {
    std::string model_path;
    int context_size = 2048;
    int batch_size = 512;
    int threads = 4;
    int gpu_layers = 0;  // CPU-only by default for edge devices
    float temperature = 0.7f;
    float top_p = 0.9f;
    int top_k = 40;
    float repeat_penalty = 1.1f;
    int seed = 42;
};

class LLMModel {
public:
    LLMModel();
    ~LLMModel();
    
    // Initialize model with configuration
    bool initialize(const ModelConfig& config);
    
    // Generate text from prompt
    std::string generate(const std::string& prompt, int max_tokens = 256);
    
    // Generate with streaming callback
    void generate_stream(const std::string& prompt, 
                        std::function<void(const std::string&)> callback,
                        int max_tokens = 256);
    
    // Check if model is loaded
    bool is_loaded() const { return model_ != nullptr; }
    
    // Get model info
    std::string get_model_info() const;
    
    // Update generation parameters
    void set_temperature(float temp) { config_.temperature = temp; }
    void set_top_p(float top_p) { config_.top_p = top_p; }
    void set_top_k(int top_k) { config_.top_k = top_k; }
    void set_repeat_penalty(float penalty) { config_.repeat_penalty = penalty; }
    
    // Cleanup backend (call at application shutdown)
    static void cleanup_backend();

private:
    llama_context* ctx_;
    llama_model* model_;
    ModelConfig config_;
    
    // Internal generation helper
    std::string generate_internal(const std::string& prompt, int max_tokens);
    
    // Tokenize text
    std::vector<llama_token> tokenize(const std::string& text);
    
    // Detokenize tokens
    std::string detokenize(const std::vector<llama_token>& tokens);
    
    // Apply sampling
    llama_token sample_next_token(const std::vector<float>& logits);
};

} // namespace local_llm 