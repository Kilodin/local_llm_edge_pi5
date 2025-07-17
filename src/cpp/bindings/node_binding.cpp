#include <napi.h>
#include "../inference/inference_engine.h"
#include <memory>
#include <thread>
#include <functional>

class LLMNodeBinding : public Napi::ObjectWrap<LLMNodeBinding> {
private:
    std::unique_ptr<local_llm::InferenceEngine> engine_;
    Napi::ThreadSafeFunction callback_tsfn_;
    std::thread worker_thread_;

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
            InstanceMethod("setRepeatPenalty", &LLMNodeBinding::SetRepeatPenalty),
            InstanceMethod("stopGeneration", &LLMNodeBinding::StopGeneration),
            StaticMethod("getSystemInfo", &LLMNodeBinding::GetSystemInfo)
        });

        exports.Set("LLMNodeBinding", func);
        return exports;
    }

    LLMNodeBinding(const Napi::CallbackInfo& info) : Napi::ObjectWrap<LLMNodeBinding>(info) {
        engine_ = std::make_unique<local_llm::InferenceEngine>();
    }

    ~LLMNodeBinding() {
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }

    Napi::Value Initialize(const Napi::CallbackInfo& info) {
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
        Napi::Env env = info.Env();
        
        if (info.Length() < 1 || !info[0].IsString()) {
            Napi::TypeError::New(env, "Expected string argument").ThrowAsJavaScriptException();
            return env.Null();
        }

        std::string prompt = info[0].As<Napi::String>().Utf8Value();
        int max_tokens = 256;
        
        if (info.Length() > 1 && info[1].IsNumber()) {
            max_tokens = info[1].As<Napi::Number>().Int32Value();
        }

        std::string result = engine_->generate_text(prompt, max_tokens);
        return Napi::String::New(env, result);
    }

    Napi::Value GenerateStream(const Napi::CallbackInfo& info) {
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

        // Create thread-safe function for callback
        callback_tsfn_ = Napi::ThreadSafeFunction::New(
            env,
            callback,
            "LLMStreamCallback",
            0,
            1
        );

        // Start generation in worker thread
        worker_thread_ = std::thread([this, prompt, max_tokens]() {
            engine_->generate_text_stream(prompt, [this](const std::string& text) {
                auto callback = [text](Napi::Env env, Napi::Function js_callback) {
                    js_callback.Call({Napi::String::New(env, text)});
                };
                callback_tsfn_.BlockingCall(callback);
            }, max_tokens);
        });

        return env.Undefined();
    }

    Napi::Value IsReady(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        return Napi::Boolean::New(env, engine_->is_ready());
    }

    Napi::Value GetModelInfo(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        std::string info_str = engine_->get_model_info();
        return Napi::String::New(env, info_str);
    }

    Napi::Value SetTemperature(const Napi::CallbackInfo& info) {
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
        Napi::Env env = info.Env();
        
        if (info.Length() < 1 || !info[0].IsNumber()) {
            Napi::TypeError::New(env, "Expected number argument").ThrowAsJavaScriptException();
            return env.Null();
        }

        float penalty = info[0].As<Napi::Number>().FloatValue();
        engine_->set_repeat_penalty(penalty);
        return env.Undefined();
    }

    Napi::Value StopGeneration(const Napi::CallbackInfo& info) {
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