#pragma once

#include "../model/llm_model.h"
#include <memory>
#include <string>
#include <functional>
#include <thread>
#include <mutex>

namespace local_llm {

class InferenceEngine {
public:
    InferenceEngine();
    ~InferenceEngine();
    
    // Initialize the inference engine
    bool initialize(const ModelConfig& config);
    
    // Generate text (synchronous)
    std::string generate_text(const std::string& prompt, int max_tokens = 256);
    
    // Generate text with streaming (asynchronous)
    void generate_text_stream(const std::string& prompt,
                             std::function<void(const std::string&)> callback,
                             int max_tokens = 256);
    
    // Check if engine is ready
    bool is_ready() const;
    
    // Get model information
    std::string get_model_info() const;
    
    // Update generation parameters
    void set_temperature(float temp);
    void set_top_p(float top_p);
    void set_top_k(int top_k);
    void set_repeat_penalty(float penalty);
    
    // Stop current generation
    void stop_generation();
    
    // Get system information
    static std::string get_system_info();

private:
    std::unique_ptr<LLMModel> model_;
    std::thread generation_thread_;
    mutable std::mutex model_mutex_;  // Changed to mutable
    bool stop_generation_;
    
    // Thread-safe model access
    LLMModel* get_model() const;
};

} // namespace local_llm