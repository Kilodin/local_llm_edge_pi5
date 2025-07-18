#include <napi.h>
#include "../inference/inference_engine.h"
#include <memory>
#include <thread>
#include <functional>
#include <iostream> // Added for logging
#include <cstddef>  // Added for std::flush

class LLMNodeBinding : public Napi::ObjectWrap<LLMNodeBinding> {
private:
    std::unique_ptr<local_llm::InferenceEngine> engine_;
    Napi::ThreadSafeFunction callback_tsfn_;
    std::thread worker_thread_;
    bool callback_tsfn_initialized_ = false;

public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports) {
        Napi::Function func = DefineClass(env, "LLMNodeBinding", {
            InstanceMethod("initialize", &LLMNodeBinding::Initialize),
            InstanceMethod("generate", &LLMNodeBinding::Generate),
            InstanceMethod("generateStream", &LLMNodeBinding::GenerateStream),
            InstanceMethod("isReady", &LLMNodeBinding::IsReady),
            InstanceMethod("getModelInfo", &LLMNodeBinding::GetModelInfo),
            InstanceMethod("setTemperature", &LLMNodeBinding::SetTemperature),
            InstanceMethod("setTopP", &LLMNodeBinding::SetTopP),
            InstanceMethod("setTopK", &LLMNodeBinding::SetTopK),
            InstanceMethod("setMinP", &LLMNodeBinding::SetMinP),
            InstanceMethod("setTypicalP", &LLMNodeBinding::SetTypicalP),
            InstanceMethod("setTfsZ", &LLMNodeBinding::SetTfsZ),
            InstanceMethod("setTopA", &LLMNodeBinding::SetTopA),
            InstanceMethod("setRepeatPenalty", &LLMNodeBinding::SetRepeatPenalty),
            InstanceMethod("setRepeatPenaltyLastN", &LLMNodeBinding::SetRepeatPenaltyLastN),
            InstanceMethod("setFrequencyPenalty", &LLMNodeBinding::SetFrequencyPenalty),
            InstanceMethod("setPresencePenalty", &LLMNodeBinding::SetPresencePenalty),
            InstanceMethod("setMirostatTau", &LLMNodeBinding::SetMirostatTau),
            InstanceMethod("setMirostatEta", &LLMNodeBinding::SetMirostatEta),
            InstanceMethod("setMirostatM", &LLMNodeBinding::SetMirostatM),
            InstanceMethod("setRopeFreqBase", &LLMNodeBinding::SetRopeFreqBase),
            InstanceMethod("setRopeFreqScale", &LLMNodeBinding::SetRopeFreqScale),
            InstanceMethod("setYarnExtFactor", &LLMNodeBinding::SetYarnExtFactor),
            InstanceMethod("setYarnAttnFactor", &LLMNodeBinding::SetYarnAttnFactor),
            InstanceMethod("setYarnBetaFast", &LLMNodeBinding::SetYarnBetaFast),
            InstanceMethod("setYarnBetaSlow", &LLMNodeBinding::SetYarnBetaSlow),
            InstanceMethod("setYarnOrigCtx", &LLMNodeBinding::SetYarnOrigCtx),
            InstanceMethod("setDefragThold", &LLMNodeBinding::SetDefragThold),
            InstanceMethod("setFlashAttn", &LLMNodeBinding::SetFlashAttn),
            InstanceMethod("setOffloadKqv", &LLMNodeBinding::SetOffloadKqv),
            InstanceMethod("setEmbeddings", &LLMNodeBinding::SetEmbeddings),
            InstanceMethod("setThreadsBatch", &LLMNodeBinding::SetThreadsBatch),
            InstanceMethod("setUbatchSize", &LLMNodeBinding::SetUbatchSize),
            InstanceMethod("stopGeneration", &LLMNodeBinding::StopGeneration),
            StaticMethod("getSystemInfo", &LLMNodeBinding::GetSystemInfo)
        });

        exports.Set("LLMNodeBinding", func);
        return exports;
    }

    LLMNodeBinding(const Napi::CallbackInfo& info) : Napi::ObjectWrap<LLMNodeBinding>(info) {
        std::cerr << "[LLMNodeBinding] Constructor called, this=" << this << std::endl << std::flush;
        engine_ = std::make_unique<local_llm::InferenceEngine>();
    }

    ~LLMNodeBinding() {
        std::cerr << "[LLMNodeBinding] Destructor called, this=" << this << std::endl << std::flush;
        engine_->stop_generation();
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
        if (callback_tsfn_initialized_) {
            callback_tsfn_.Release();
            callback_tsfn_initialized_ = false;
        }
    }

    Napi::Value Initialize(const Napi::CallbackInfo& info) {
        std::cerr << "[LLMNodeBinding] Initialize called, this=" << this << std::endl << std::flush;
        Napi::Env env = info.Env();
        
        if (info.Length() < 1 || !info[0].IsObject()) {
            Napi::TypeError::New(env, "Expected object argument").ThrowAsJavaScriptException();
            return env.Null();
        }

        Napi::Object config_obj = info[0].As<Napi::Object>();
        
        local_llm::ModelConfig config;
        
        if (config_obj.Has("modelPath")) {
            config.model_path = config_obj.Get("modelPath").As<Napi::String>().Utf8Value();
        }
        
        if (config_obj.Has("contextSize")) {
            config.context_size = config_obj.Get("contextSize").As<Napi::Number>().Int32Value();
        }
        
        if (config_obj.Has("batchSize")) {
            config.batch_size = config_obj.Get("batchSize").As<Napi::Number>().Int32Value();
        }
        
        if (config_obj.Has("threads")) {
            config.threads = config_obj.Get("threads").As<Napi::Number>().Int32Value();
        }
        
        if (config_obj.Has("gpuLayers")) {
            config.gpu_layers = config_obj.Get("gpuLayers").As<Napi::Number>().Int32Value();
        }
        
        if (config_obj.Has("temperature")) {
            config.temperature = config_obj.Get("temperature").As<Napi::Number>().FloatValue();
        }
        
        if (config_obj.Has("topP")) {
            config.top_p = config_obj.Get("topP").As<Napi::Number>().FloatValue();
        }
        
        if (config_obj.Has("topK")) {
            config.top_k = config_obj.Get("topK").As<Napi::Number>().Int32Value();
        }
        
        if (config_obj.Has("repeatPenalty")) {
            config.repeat_penalty = config_obj.Get("repeatPenalty").As<Napi::Number>().FloatValue();
        }
        
        if (config_obj.Has("seed")) {
            config.seed = config_obj.Get("seed").As<Napi::Number>().Int32Value();
        }

        bool success = engine_->initialize(config);
        return Napi::Boolean::New(env, success);
    }

    Napi::Value Generate(const Napi::CallbackInfo& info) {
        std::cerr << "[LLMNodeBinding] Generate called, this=" << this << ", engine_=" << engine_.get() << std::endl << std::flush;
        Napi::Env env = info.Env();
        
        if (info.Length() < 1 || !info[0].IsString()) {
            Napi::TypeError::New(env, "Expected string argument").ThrowAsJavaScriptException();
            return env.Null();
        }

        std::string prompt = info[0].As<Napi::String>().Utf8Value();
        std::cerr << "[LLMNodeBinding] Prompt received: '" << prompt << "'" << std::endl << std::flush;
        int max_tokens = 256;
        
        if (info.Length() > 1 && info[1].IsNumber()) {
            max_tokens = info[1].As<Napi::Number>().Int32Value();
        }

        std::string result = engine_->generate_text(prompt, max_tokens);
        return Napi::String::New(env, result);
    }

    Napi::Value GenerateStream(const Napi::CallbackInfo& info) {
        std::cerr << "[LLMNodeBinding] GenerateStream called, this=" << this << std::endl << std::flush;
        Napi::Env env = info.Env();
        
        if (info.Length() < 2 || !info[0].IsString() || !info[1].IsFunction()) {
            Napi::TypeError::New(env, "Expected string and function arguments").ThrowAsJavaScriptException();
            return env.Null();
        }

        std::string prompt = info[0].As<Napi::String>().Utf8Value();
        Napi::Function callback = info[1].As<Napi::Function>();
        int max_tokens = 256;
        
        if (info.Length() > 2 && info[2].IsNumber()) {
            max_tokens = info[2].As<Napi::Number>().Int32Value();
        }

        // Stop any previous generation first
        engine_->stop_generation();
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }

        // Release previous ThreadSafeFunction if it exists
        if (callback_tsfn_initialized_) {
            callback_tsfn_.Release();
            callback_tsfn_initialized_ = false;
        }

        // Create thread-safe function for callback with larger queue and better error handling
        callback_tsfn_ = Napi::ThreadSafeFunction::New(
            env,
            callback,
            "LLMStreamCallback",
            2000,  // max_queue_size = 2000 (allow more queued callbacks)
            1
        );
        callback_tsfn_initialized_ = true;

        // Start generation in worker thread
        worker_thread_ = std::thread([this, prompt, max_tokens]() {
            std::cerr << "[LLMNodeBinding] Starting generation thread for prompt: '" << prompt << "'" << std::endl;
            engine_->generate_text_stream(prompt, [this](const std::string& text) {
                std::cerr << "[LLMNodeBinding] Received text from engine: '" << text << "' (length: " << text.length() << ")" << std::endl;
                auto callback = [text](Napi::Env env, Napi::Function js_callback) {
                    try {
                        // std::cerr << "[LLMNodeBinding] Executing JS callback with text: '" << text << "'" << std::endl;
                        js_callback.Call({Napi::String::New(env, text)});
                        // std::cerr << "[LLMNodeBinding] JS callback completed successfully" << std::endl;
                    } catch (const std::exception& e) {
                        std::cerr << "[LLMNodeBinding] Exception in callback: " << e.what() << std::endl;
                    }
                };
                
                // Try NonBlockingCall first, fall back to BlockingCall if queue is full
                napi_status status = callback_tsfn_.NonBlockingCall(callback);
                if (status != napi_ok) {
                    std::cerr << "[LLMNodeBinding] NonBlockingCall failed with status " << status 
                              << ", trying BlockingCall for text: '" << text << "'" << std::endl;
                    
                    // Fall back to blocking call if non-blocking fails
                    status = callback_tsfn_.BlockingCall(callback);
                    if (status != napi_ok) {
                        std::cerr << "[LLMNodeBinding] BlockingCall also failed with status " << status << std::endl;
                    }
                } else {
                    std::cerr << "[LLMNodeBinding] Successfully queued callback for text: '" << text << "'" << std::endl;
                }
            }, max_tokens);
            std::cerr << "[LLMNodeBinding] Generation thread completed" << std::endl;
        });

        return env.Undefined();
    }

    Napi::Value IsReady(const Napi::CallbackInfo& info) {
        std::cerr << "[LLMNodeBinding] IsReady called, this=" << this << std::endl << std::flush;
        Napi::Env env = info.Env();
        return Napi::Boolean::New(env, engine_->is_ready());
    }

    Napi::Value GetModelInfo(const Napi::CallbackInfo& info) {
        std::cerr << "[LLMNodeBinding] GetModelInfo called, this=" << this << std::endl << std::flush;
        Napi::Env env = info.Env();
        std::string info_str = engine_->get_model_info();
        return Napi::String::New(env, info_str);
    }

    Napi::Value SetTemperature(const Napi::CallbackInfo& info) {
        std::cerr << "[LLMNodeBinding] SetTemperature called, this=" << this << std::endl << std::flush;
        Napi::Env env = info.Env();
        
        if (info.Length() < 1 || !info[0].IsNumber()) {
            Napi::TypeError::New(env, "Expected number argument").ThrowAsJavaScriptException();
            return env.Null();
        }

        float temp = info[0].As<Napi::Number>().FloatValue();
        engine_->set_temperature(temp);
        return env.Undefined();
    }

    Napi::Value SetTopP(const Napi::CallbackInfo& info) {
        std::cerr << "[LLMNodeBinding] SetTopP called, this=" << this << std::endl << std::flush;
        Napi::Env env = info.Env();
        
        if (info.Length() < 1 || !info[0].IsNumber()) {
            Napi::TypeError::New(env, "Expected number argument").ThrowAsJavaScriptException();
            return env.Null();
        }

        float top_p = info[0].As<Napi::Number>().FloatValue();
        engine_->set_top_p(top_p);
        return env.Undefined();
    }

    Napi::Value SetTopK(const Napi::CallbackInfo& info) {
        std::cerr << "[LLMNodeBinding] SetTopK called, this=" << this << std::endl << std::flush;
        Napi::Env env = info.Env();
        
        if (info.Length() < 1 || !info[0].IsNumber()) {
            Napi::TypeError::New(env, "Expected number argument").ThrowAsJavaScriptException();
            return env.Null();
        }

        int top_k = info[0].As<Napi::Number>().Int32Value();
        engine_->set_top_k(top_k);
        return env.Undefined();
    }

    Napi::Value SetRepeatPenalty(const Napi::CallbackInfo& info) {
        std::cerr << "[LLMNodeBinding] SetRepeatPenalty called, this=" << this << std::endl << std::flush;
        Napi::Env env = info.Env();
        
        if (info.Length() < 1 || !info[0].IsNumber()) {
            Napi::TypeError::New(env, "Expected number argument").ThrowAsJavaScriptException();
            return env.Null();
        }

        float penalty = info[0].As<Napi::Number>().FloatValue();
        engine_->set_repeat_penalty(penalty);
        return env.Undefined();
    }

    Napi::Value SetMinP(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        if (info.Length() < 1 || !info[0].IsNumber()) {
            Napi::TypeError::New(env, "Expected number argument").ThrowAsJavaScriptException();
            return env.Null();
        }
        float min_p = info[0].As<Napi::Number>().FloatValue();
        engine_->set_min_p(min_p);
        return env.Undefined();
    }

    Napi::Value SetTypicalP(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        if (info.Length() < 1 || !info[0].IsNumber()) {
            Napi::TypeError::New(env, "Expected number argument").ThrowAsJavaScriptException();
            return env.Null();
        }
        float typical_p = info[0].As<Napi::Number>().FloatValue();
        engine_->set_typical_p(typical_p);
        return env.Undefined();
    }

    Napi::Value SetTfsZ(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        if (info.Length() < 1 || !info[0].IsNumber()) {
            Napi::TypeError::New(env, "Expected number argument").ThrowAsJavaScriptException();
            return env.Null();
        }
        float tfs_z = info[0].As<Napi::Number>().FloatValue();
        engine_->set_tfs_z(tfs_z);
        return env.Undefined();
    }

    Napi::Value SetTopA(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        if (info.Length() < 1 || !info[0].IsNumber()) {
            Napi::TypeError::New(env, "Expected number argument").ThrowAsJavaScriptException();
            return env.Null();
        }
        float top_a = info[0].As<Napi::Number>().FloatValue();
        engine_->set_top_a(top_a);
        return env.Undefined();
    }

    Napi::Value SetRepeatPenaltyLastN(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        if (info.Length() < 1 || !info[0].IsNumber()) {
            Napi::TypeError::New(env, "Expected number argument").ThrowAsJavaScriptException();
            return env.Null();
        }
        int last_n = info[0].As<Napi::Number>().Int32Value();
        engine_->set_repeat_penalty_last_n(last_n);
        return env.Undefined();
    }

    Napi::Value SetFrequencyPenalty(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        if (info.Length() < 1 || !info[0].IsNumber()) {
            Napi::TypeError::New(env, "Expected number argument").ThrowAsJavaScriptException();
            return env.Null();
        }
        float penalty = info[0].As<Napi::Number>().FloatValue();
        engine_->set_frequency_penalty(penalty);
        return env.Undefined();
    }

    Napi::Value SetPresencePenalty(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        if (info.Length() < 1 || !info[0].IsNumber()) {
            Napi::TypeError::New(env, "Expected number argument").ThrowAsJavaScriptException();
            return env.Null();
        }
        float penalty = info[0].As<Napi::Number>().FloatValue();
        engine_->set_presence_penalty(penalty);
        return env.Undefined();
    }

    Napi::Value SetMirostatTau(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        if (info.Length() < 1 || !info[0].IsNumber()) {
            Napi::TypeError::New(env, "Expected number argument").ThrowAsJavaScriptException();
            return env.Null();
        }
        float tau = info[0].As<Napi::Number>().FloatValue();
        engine_->set_mirostat_tau(tau);
        return env.Undefined();
    }

    Napi::Value SetMirostatEta(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        if (info.Length() < 1 || !info[0].IsNumber()) {
            Napi::TypeError::New(env, "Expected number argument").ThrowAsJavaScriptException();
            return env.Null();
        }
        float eta = info[0].As<Napi::Number>().FloatValue();
        engine_->set_mirostat_eta(eta);
        return env.Undefined();
    }

    Napi::Value SetMirostatM(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        if (info.Length() < 1 || !info[0].IsNumber()) {
            Napi::TypeError::New(env, "Expected number argument").ThrowAsJavaScriptException();
            return env.Null();
        }
        int m = info[0].As<Napi::Number>().Int32Value();
        engine_->set_mirostat_m(m);
        return env.Undefined();
    }

    Napi::Value SetRopeFreqBase(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        if (info.Length() < 1 || !info[0].IsNumber()) {
            Napi::TypeError::New(env, "Expected number argument").ThrowAsJavaScriptException();
            return env.Null();
        }
        float freq_base = info[0].As<Napi::Number>().FloatValue();
        engine_->set_rope_freq_base(freq_base);
        return env.Undefined();
    }

    Napi::Value SetRopeFreqScale(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        if (info.Length() < 1 || !info[0].IsNumber()) {
            Napi::TypeError::New(env, "Expected number argument").ThrowAsJavaScriptException();
            return env.Null();
        }
        float freq_scale = info[0].As<Napi::Number>().FloatValue();
        engine_->set_rope_freq_scale(freq_scale);
        return env.Undefined();
    }

    Napi::Value SetYarnExtFactor(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        if (info.Length() < 1 || !info[0].IsNumber()) {
            Napi::TypeError::New(env, "Expected number argument").ThrowAsJavaScriptException();
            return env.Null();
        }
        float factor = info[0].As<Napi::Number>().FloatValue();
        engine_->set_yarn_ext_factor(factor);
        return env.Undefined();
    }

    Napi::Value SetYarnAttnFactor(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        if (info.Length() < 1 || !info[0].IsNumber()) {
            Napi::TypeError::New(env, "Expected number argument").ThrowAsJavaScriptException();
            return env.Null();
        }
        float factor = info[0].As<Napi::Number>().FloatValue();
        engine_->set_yarn_attn_factor(factor);
        return env.Undefined();
    }

    Napi::Value SetYarnBetaFast(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        if (info.Length() < 1 || !info[0].IsNumber()) {
            Napi::TypeError::New(env, "Expected number argument").ThrowAsJavaScriptException();
            return env.Null();
        }
        float beta = info[0].As<Napi::Number>().FloatValue();
        engine_->set_yarn_beta_fast(beta);
        return env.Undefined();
    }

    Napi::Value SetYarnBetaSlow(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        if (info.Length() < 1 || !info[0].IsNumber()) {
            Napi::TypeError::New(env, "Expected number argument").ThrowAsJavaScriptException();
            return env.Null();
        }
        float beta = info[0].As<Napi::Number>().FloatValue();
        engine_->set_yarn_beta_slow(beta);
        return env.Undefined();
    }

    Napi::Value SetYarnOrigCtx(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        if (info.Length() < 1 || !info[0].IsNumber()) {
            Napi::TypeError::New(env, "Expected number argument").ThrowAsJavaScriptException();
            return env.Null();
        }
        uint32_t ctx = info[0].As<Napi::Number>().Uint32Value();
        engine_->set_yarn_orig_ctx(ctx);
        return env.Undefined();
    }

    Napi::Value SetDefragThold(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        if (info.Length() < 1 || !info[0].IsNumber()) {
            Napi::TypeError::New(env, "Expected number argument").ThrowAsJavaScriptException();
            return env.Null();
        }
        float thold = info[0].As<Napi::Number>().FloatValue();
        engine_->set_defrag_thold(thold);
        return env.Undefined();
    }

    Napi::Value SetFlashAttn(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        if (info.Length() < 1 || !info[0].IsBoolean()) {
            Napi::TypeError::New(env, "Expected boolean argument").ThrowAsJavaScriptException();
            return env.Null();
        }
        bool enabled = info[0].As<Napi::Boolean>().Value();
        engine_->set_flash_attn(enabled);
        return env.Undefined();
    }

    Napi::Value SetOffloadKqv(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        if (info.Length() < 1 || !info[0].IsBoolean()) {
            Napi::TypeError::New(env, "Expected boolean argument").ThrowAsJavaScriptException();
            return env.Null();
        }
        bool enabled = info[0].As<Napi::Boolean>().Value();
        engine_->set_offload_kqv(enabled);
        return env.Undefined();
    }

    Napi::Value SetEmbeddings(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        if (info.Length() < 1 || !info[0].IsBoolean()) {
            Napi::TypeError::New(env, "Expected boolean argument").ThrowAsJavaScriptException();
            return env.Null();
        }
        bool enabled = info[0].As<Napi::Boolean>().Value();
        engine_->set_embeddings(enabled);
        return env.Undefined();
    }

    Napi::Value SetThreadsBatch(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        if (info.Length() < 1 || !info[0].IsNumber()) {
            Napi::TypeError::New(env, "Expected number argument").ThrowAsJavaScriptException();
            return env.Null();
        }
        int threads = info[0].As<Napi::Number>().Int32Value();
        engine_->set_threads_batch(threads);
        return env.Undefined();
    }

    Napi::Value SetUbatchSize(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        if (info.Length() < 1 || !info[0].IsNumber()) {
            Napi::TypeError::New(env, "Expected number argument").ThrowAsJavaScriptException();
            return env.Null();
        }
        int size = info[0].As<Napi::Number>().Int32Value();
        engine_->set_ubatch_size(size);
        return env.Undefined();
    }

    Napi::Value StopGeneration(const Napi::CallbackInfo& info) {
        std::cerr << "[LLMNodeBinding] StopGeneration called, this=" << this << std::endl << std::flush;
        Napi::Env env = info.Env();
        engine_->stop_generation();
        return env.Undefined();
    }

    static Napi::Value GetSystemInfo(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        std::string info_str = local_llm::InferenceEngine::get_system_info();
        return Napi::String::New(env, info_str);
    }
};

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    return LLMNodeBinding::Init(env, exports);
}

NODE_API_MODULE(llm_node, Init) 