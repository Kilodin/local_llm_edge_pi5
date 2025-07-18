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
    void set_min_p(float min_p);
    void set_typical_p(float typical_p);
    void set_tfs_z(float tfs_z);
    void set_top_a(float top_a);
    void set_repeat_penalty(float penalty);
    void set_repeat_penalty_last_n(int last_n);
    void set_frequency_penalty(float penalty);
    void set_presence_penalty(float penalty);
    void set_mirostat_tau(float tau);
    void set_mirostat_eta(float eta);
    void set_mirostat_m(int m);
    void set_rope_freq_base(float freq_base);
    void set_rope_freq_scale(float freq_scale);
    void set_yarn_ext_factor(float factor);
    void set_yarn_attn_factor(float factor);
    void set_yarn_beta_fast(float beta);
    void set_yarn_beta_slow(float beta);
    void set_yarn_orig_ctx(uint32_t ctx);
    void set_defrag_thold(float thold);
    void set_flash_attn(bool enabled);
    void set_offload_kqv(bool enabled);
    void set_embeddings(bool enabled);
    void set_threads_batch(int threads);
    void set_ubatch_size(int size);
    
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