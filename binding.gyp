{
  "targets": [
    {
      "target_name": "llm_node",
      "sources": [
        "src/cpp/bindings/node_binding.cpp",
        "src/cpp/model/llm_model.cpp",
        "src/cpp/inference/inference_engine.cpp",
        "src/cpp/inference/prompt_processor.cpp"
      ],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "src/cpp",
        "third_party/llama.cpp/include",
        "third_party/llama.cpp/src",
        "third_party/llama.cpp/ggml/include",
        "third_party/llama.cpp/ggml/src"
      ],
      "dependencies": [
        "<!(node -p \"require('node-addon-api').gyp\")"
      ],
      "cflags!": [ "-fno-exceptions" ],
      "cflags_cc!": [ "-fno-exceptions" ],
      "cflags_cc": [ "-fvisibility=hidden", "-fPIC" ],
      "defines": [ "NAPI_DISABLE_CPP_EXCEPTIONS" ],
      "xcode_settings": {
        "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
        "CLANG_CXX_LIBRARY": "libc++",
        "MACOSX_DEPLOYMENT_TARGET": "10.7"
      },
      "msvs_settings": {
        "VCCLCompilerTool": {
          "ExceptionHandling": 1
        }
      },
      "libraries": [
        "-L<(module_root_dir)/third_party/llama.cpp/build/bin",
        "-lllama",
        "-lggml",
        "-lggml-base",
        "-lggml-cpu"
      ],
      "library_dirs": [
        "<(module_root_dir)/third_party/llama.cpp/build/bin"
      ],
      "link_settings": {
        "libraries": [
          "-lpthread",
          "-lm"
        ]
      }
    }
  ]
} 