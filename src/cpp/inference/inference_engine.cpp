#include "inference_engine.h"
#include <iostream>
#include <sstream>
#include <unistd.h>
#include <sys/sysinfo.h>

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

void InferenceEngine::stop_generation() {
    stop_generation_ = true;
}

LLMModel* InferenceEngine::get_model() const {
    return model_.get();
}

std::string InferenceEngine::get_system_info() {
    std::ostringstream oss;
    
    // CPU info
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        oss << "CPU cores: " << si.procs << "\n";
        oss << "Total RAM: " << (si.totalram / 1024 / 1024) << " MB\n";
        oss << "Free RAM: " << (si.freeram / 1024 / 1024) << " MB\n";
    }
    
#ifdef __linux__
    // More detailed CPU info on Linux
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (cpuinfo.is_open()) {
        std::string line;
        while (std::getline(cpuinfo, line)) {
            if (line.substr(0, 10) == "model name") {
                size_t pos = line.find(':');
                if (pos != std::string::npos) {
                    oss << "CPU: " << line.substr(pos + 2) << "\n";
                    break;
                }
            }
        }
        cpuinfo.close();
    }
    
    // Load average
    std::ifstream loadavg("/proc/loadavg");
    if (loadavg.is_open()) {
        std::string line;
        std::getline(loadavg, line);
        oss << "Load average: " << line << "\n";
        loadavg.close();
    }
#endif
    
    return oss.str();
}

} // namespace local_llm 