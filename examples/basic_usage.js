const { LLMNodeBinding } = require('../build/Release/llm_node');

async function basicExample() {
    console.log('🤖 Local LLM Inference - Basic Example\n');
    
    // Create LLM instance
    const llm = new LLMNodeBinding();
    
    // Show system information
    console.log('📊 System Information:');
    console.log(LLMNodeBinding.getSystemInfo());
    console.log();
    
    // Initialize model
    console.log('🔧 Initializing model...');
    const success = llm.initialize({
        modelPath: './models/tinyllama-1.1b-chat.gguf',
        contextSize: 2048,
        batchSize: 512,
        threads: 4,
        gpuLayers: 0,
        temperature: 0.7,
        topP: 0.9,
        topK: 40,
        repeatPenalty: 1.1,
        seed: 42
    });
    
    if (!success) {
        console.error('❌ Failed to initialize model');
        return;
    }
    
    console.log('✅ Model initialized successfully!');
    console.log('\n📋 Model Information:');
    console.log(llm.getModelInfo());
    console.log();
    
    // Generate text
    console.log('💬 Generating text...');
    const prompt = "Hello! Can you tell me a short joke?";
    console.log(`📝 Prompt: ${prompt}`);
    
    const result = llm.generate(prompt, 100);
    console.log(`🤖 Response: ${result}`);
    console.log();
    
    // Update parameters
    console.log('⚙️ Updating parameters...');
    llm.setTemperature(0.9);
    llm.setTopP(0.8);
    console.log('✅ Parameters updated');
    console.log();
    
    // Generate with new parameters
    console.log('💬 Generating with new parameters...');
    const newPrompt = "Write a haiku about programming";
    console.log(`📝 Prompt: ${newPrompt}`);
    
    const newResult = llm.generate(newPrompt, 50);
    console.log(`🤖 Response: ${newResult}`);
    console.log();
    
    console.log('🎉 Example completed!');
}

// Run example
if (require.main === module) {
    basicExample().catch(console.error);
}

module.exports = { basicExample }; 