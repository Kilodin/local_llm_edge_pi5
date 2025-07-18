#pragma once

#include <string>
#include <memory>
#include <vector>
#include <functional>
#include "llama.h"

namespace local_llm {

struct ModelConfig {
    std::string model_path;
    
    // Context and memory settings
    int context_size = 2048;
    int batch_size = 512;
    int ubatch_size = 512;  // physical maximum batch size
    
    // Threading and performance
    int threads = 4;
    int threads_batch = 4;  // threads for batch processing
    int gpu_layers = 0;     // CPU-only by default for edge devices
    
    // Sampling parameters
    float temperature = 0.7f;
    float top_p = 0.9f;
    int top_k = 40;
    float min_p = 0.0f;     // minimum probability threshold
    float typical_p = 1.0f; // typical sampling
    float tfs_z = 1.0f;     // tail free sampling
    float top_a = 0.0f;     // top-a sampling
    
    // Penalty parameters
    float repeat_penalty = 1.1f;
    int repeat_penalty_last_n = 64;  // last n tokens to penalize
    float frequency_penalty = 0.0f;  // frequency penalty
    float presence_penalty = 0.0f;   // presence penalty
    
    // Advanced sampling
    float mirostat_tau = 5.0f;       // mirostat target entropy
    float mirostat_eta = 0.1f;       // mirostat learning rate
    int mirostat_m = 100;            // mirostat number of tokens
    
    // RoPE (Rotary Position Embedding) settings
    float rope_freq_base = 0.0f;     // RoPE base frequency (0 = from model)
    float rope_freq_scale = 0.0f;    // RoPE frequency scaling (0 = from model)
    
    // YaRN (Yet another RoPE extension) settings
    float yarn_ext_factor = -1.0f;   // YaRN extrapolation mix factor
    float yarn_attn_factor = 1.0f;   // YaRN magnitude scaling factor
    float yarn_beta_fast = 32.0f;    // YaRN low correction dim
    float yarn_beta_slow = 1.0f;     // YaRN high correction dim
    uint32_t yarn_orig_ctx = 0;      // YaRN original context size
    
    // Memory and optimization
    float defrag_thold = 0.0f;       // KV cache defrag threshold
    bool flash_attn = false;         // use flash attention
    bool offload_kqv = false;        // offload KQV to GPU
    bool embeddings = false;         // extract embeddings
    
    // Random seed
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
    void set_min_p(float min_p) { config_.min_p = min_p; }
    void set_typical_p(float typical_p) { config_.typical_p = typical_p; }
    void set_tfs_z(float tfs_z) { config_.tfs_z = tfs_z; }
    void set_top_a(float top_a) { config_.top_a = top_a; }
    void set_repeat_penalty(float penalty) { config_.repeat_penalty = penalty; }
    void set_repeat_penalty_last_n(int last_n) { config_.repeat_penalty_last_n = last_n; }
    void set_frequency_penalty(float penalty) { config_.frequency_penalty = penalty; }
    void set_presence_penalty(float penalty) { config_.presence_penalty = penalty; }
    void set_mirostat_tau(float tau) { config_.mirostat_tau = tau; }
    void set_mirostat_eta(float eta) { config_.mirostat_eta = eta; }
    void set_mirostat_m(int m) { config_.mirostat_m = m; }
    void set_rope_freq_base(float freq_base) { config_.rope_freq_base = freq_base; }
    void set_rope_freq_scale(float freq_scale) { config_.rope_freq_scale = freq_scale; }
    void set_yarn_ext_factor(float factor) { config_.yarn_ext_factor = factor; }
    void set_yarn_attn_factor(float factor) { config_.yarn_attn_factor = factor; }
    void set_yarn_beta_fast(float beta) { config_.yarn_beta_fast = beta; }
    void set_yarn_beta_slow(float beta) { config_.yarn_beta_slow = beta; }
    void set_yarn_orig_ctx(uint32_t ctx) { config_.yarn_orig_ctx = ctx; }
    void set_defrag_thold(float thold) { config_.defrag_thold = thold; }
    void set_flash_attn(bool enabled) { config_.flash_attn = enabled; }
    void set_offload_kqv(bool enabled) { config_.offload_kqv = enabled; }
    void set_embeddings(bool enabled) { config_.embeddings = enabled; }
    void set_threads_batch(int threads) { config_.threads_batch = threads; }
    void set_ubatch_size(int size) { config_.ubatch_size = size; }
    
    // Cleanup backend (call at application shutdown)
    static void cleanup_backend();

private:
    llama_context* ctx_;
    llama_model* model_;
    ModelConfig config_;
    std::vector<llama_token> recent_tokens_;  // For repeat penalty
    
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