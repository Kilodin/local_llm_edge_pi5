#include "inference_engine.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <unistd.h>
#include <sys/sysinfo.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <cstring>

#ifdef __linux__
#include <fstream>
#include <sstream>
#endif

namespace local_llm {

InferenceEngine::InferenceEngine() : stop_generation_(false) {}

InferenceEngine::~InferenceEngine() {
    stop_generation();
    if (generation_thread_.joinable()) {
        generation_thread_.join();
    }
}

bool InferenceEngine::initialize(const ModelConfig& config) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    
    model_ = std::make_unique<LLMModel>();
    bool success = model_->initialize(config);
    
    if (success) {
        std::cout << "Inference engine initialized successfully" << std::endl;
        std::cout << "System info: " << get_system_info() << std::endl;
    } else {
        std::cerr << "Failed to initialize inference engine" << std::endl;
    }
    
    return success;
}

std::string InferenceEngine::generate_text(const std::string& prompt, int max_tokens) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    
    if (!model_ || !model_->is_loaded()) {
        return "Error: Model not loaded";
    }
    
    return model_->generate(prompt, max_tokens);
}

void InferenceEngine::generate_text_stream(const std::string& prompt,
                                         std::function<void(const std::string&)> callback,
                                         int max_tokens) {
    stop_generation();
    if (generation_thread_.joinable()) {
        generation_thread_.join();
    }
    
    stop_generation_ = false;
    
    generation_thread_ = std::thread([this, prompt, callback, max_tokens]() {
        std::lock_guard<std::mutex> lock(model_mutex_);
        
        if (!model_ || !model_->is_loaded()) {
            callback("Error: Model not loaded");
            return;
        }
        
        model_->generate_stream(prompt, [this, callback](const std::string& text) {
            if (stop_generation_) {
                return;
            }
            callback(text);
        }, max_tokens);
    });
}

bool InferenceEngine::is_ready() const {
    std::lock_guard<std::mutex> lock(model_mutex_);
    return model_ && model_->is_loaded();
}

std::string InferenceEngine::get_model_info() const {
    std::lock_guard<std::mutex> lock(model_mutex_);
    if (!model_) {
        return "No model loaded";
    }
    return model_->get_model_info();
}

void InferenceEngine::set_temperature(float temp) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    if (model_) {
        model_->set_temperature(temp);
    }
}

void InferenceEngine::set_top_p(float top_p) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    if (model_) {
        model_->set_top_p(top_p);
    }
}

void InferenceEngine::set_top_k(int top_k) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    if (model_) {
        model_->set_top_k(top_k);
    }
}

void InferenceEngine::set_repeat_penalty(float penalty) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    if (model_) {
        model_->set_repeat_penalty(penalty);
    }
}

void InferenceEngine::set_min_p(float min_p) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    if (model_) {
        model_->set_min_p(min_p);
    }
}

void InferenceEngine::set_typical_p(float typical_p) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    if (model_) {
        model_->set_typical_p(typical_p);
    }
}

void InferenceEngine::set_tfs_z(float tfs_z) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    if (model_) {
        model_->set_tfs_z(tfs_z);
    }
}

void InferenceEngine::set_top_a(float top_a) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    if (model_) {
        model_->set_top_a(top_a);
    }
}

void InferenceEngine::set_repeat_penalty_last_n(int last_n) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    if (model_) {
        model_->set_repeat_penalty_last_n(last_n);
    }
}

void InferenceEngine::set_frequency_penalty(float penalty) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    if (model_) {
        model_->set_frequency_penalty(penalty);
    }
}

void InferenceEngine::set_presence_penalty(float penalty) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    if (model_) {
        model_->set_presence_penalty(penalty);
    }
}

void InferenceEngine::set_mirostat_tau(float tau) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    if (model_) {
        model_->set_mirostat_tau(tau);
    }
}

void InferenceEngine::set_mirostat_eta(float eta) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    if (model_) {
        model_->set_mirostat_eta(eta);
    }
}

void InferenceEngine::set_mirostat_m(int m) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    if (model_) {
        model_->set_mirostat_m(m);
    }
}

void InferenceEngine::set_rope_freq_base(float freq_base) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    if (model_) {
        model_->set_rope_freq_base(freq_base);
    }
}

void InferenceEngine::set_rope_freq_scale(float freq_scale) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    if (model_) {
        model_->set_rope_freq_scale(freq_scale);
    }
}

void InferenceEngine::set_yarn_ext_factor(float factor) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    if (model_) {
        model_->set_yarn_ext_factor(factor);
    }
}

void InferenceEngine::set_yarn_attn_factor(float factor) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    if (model_) {
        model_->set_yarn_attn_factor(factor);
    }
}

void InferenceEngine::set_yarn_beta_fast(float beta) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    if (model_) {
        model_->set_yarn_beta_fast(beta);
    }
}

void InferenceEngine::set_yarn_beta_slow(float beta) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    if (model_) {
        model_->set_yarn_beta_slow(beta);
    }
}

void InferenceEngine::set_yarn_orig_ctx(uint32_t ctx) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    if (model_) {
        model_->set_yarn_orig_ctx(ctx);
    }
}

void InferenceEngine::set_defrag_thold(float thold) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    if (model_) {
        model_->set_defrag_thold(thold);
    }
}

void InferenceEngine::set_flash_attn(bool enabled) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    if (model_) {
        model_->set_flash_attn(enabled);
    }
}

void InferenceEngine::set_offload_kqv(bool enabled) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    if (model_) {
        model_->set_offload_kqv(enabled);
    }
}

void InferenceEngine::set_embeddings(bool enabled) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    if (model_) {
        model_->set_embeddings(enabled);
    }
}

void InferenceEngine::set_threads_batch(int threads) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    if (model_) {
        model_->set_threads_batch(threads);
    }
}

void InferenceEngine::set_ubatch_size(int size) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    if (model_) {
        model_->set_ubatch_size(size);
    }
}

void InferenceEngine::stop_generation() {
    stop_generation_ = true;
}

LLMModel* InferenceEngine::get_model() const {
    return model_.get();
}

std::string InferenceEngine::get_system_info() {
    std::ostringstream oss;
    
    // Detect Raspberry Pi 5
    std::ifstream cpuinfo("/proc/cpuinfo");
    bool is_raspberry_pi = false;
    std::string hardware = "";
    std::string revision = "";
    
    if (cpuinfo.is_open()) {
        std::string line;
        while (std::getline(cpuinfo, line)) {
            if (line.substr(0, 8) == "Hardware") {
                size_t pos = line.find(':');
                if (pos != std::string::npos) {
                    hardware = line.substr(pos + 2);
                    if (hardware.find("BCM") != std::string::npos) {
                        is_raspberry_pi = true;
                    }
                }
            } else if (line.substr(0, 8) == "Revision") {
                size_t pos = line.find(':');
                if (pos != std::string::npos) {
                    revision = line.substr(pos + 2);
                }
            }
        }
        cpuinfo.close();
    }
    
    // System identification
    if (is_raspberry_pi) {
        oss << "Device: Raspberry Pi 5\n";
        oss << "Hardware: " << hardware << "\n";
        if (!revision.empty()) {
            oss << "Revision: " << revision << "\n";
        }
    } else {
        oss << "Device: Linux System\n";
    }
    
    // CPU info
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        oss << "CPU Cores: " << si.procs << "\n";
        oss << "Total RAM: " << (si.totalram / 1024 / 1024) << " MB\n";
        oss << "Free RAM: " << (si.freeram / 1024 / 1024) << " MB\n";
        oss << "Used RAM: " << ((si.totalram - si.freeram) / 1024 / 1024) << " MB\n";
    }
    
#ifdef __linux__
    // Get IP address
    struct ifaddrs *ifaddr, *ifa;
    if (getifaddrs(&ifaddr) == -1) {
        oss << "IP Address: Unable to get\n";
    } else {
        bool found_ip = false;
        for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
            if (ifa->ifa_addr == NULL) continue;
            
            // Only show IPv4 addresses and skip loopback
            if (ifa->ifa_addr->sa_family == AF_INET && 
                strcmp(ifa->ifa_name, "lo") != 0) {
                char host[NI_MAXHOST];
                if (getnameinfo(ifa->ifa_addr, sizeof(struct sockaddr_in),
                               host, NI_MAXHOST, NULL, 0, NI_NUMERICHOST) == 0) {
                    oss << "IP Address: " << host << " (" << ifa->ifa_name << ")\n";
                    found_ip = true;
                    break; // Just show the first non-loopback IP
                }
            }
        }
        if (!found_ip) {
            oss << "IP Address: Not found\n";
        }
        freeifaddrs(ifaddr);
    }
    
    // Load average
    std::ifstream loadavg("/proc/loadavg");
    if (loadavg.is_open()) {
        std::string line;
        std::getline(loadavg, line);
        oss << "Load Average: " << line << "\n";
        loadavg.close();
    }
    
    // CPU temperature (for Raspberry Pi)
    if (is_raspberry_pi) {
        std::ifstream temp("/sys/class/thermal/thermal_zone0/temp");
        if (temp.is_open()) {
            int temp_value;
            temp >> temp_value;
            temp.close();
            float temp_celsius = temp_value / 1000.0f;
            oss << "CPU Temperature: " << std::fixed << std::setprecision(1) << temp_celsius << "°C\n";
        }
    }
#endif
    
    return oss.str();
}

} // namespace local_llm 