const { LLMNodeBinding } = require('../build/Release/llm_node');

async function streamingExample() {
    console.log('🤖 Local LLM Inference - Streaming Example\n');
    
    // Create LLM instance
    const llm = new LLMNodeBinding();
    
    // Initialize model
    console.log('🔧 Initializing model...');
    const success = llm.initialize({
        modelPath: './models/llama-2-7b-chat.Q4_K_M.gguf',
        contextSize: 1024,
        batchSize: 256,
        threads: 4,
        temperature: 0.8,
        topP: 0.9,
        topK: 40,
        repeatPenalty: 1.1
    });
    
    if (!success) {
        console.error('❌ Failed to initialize model');
        return;
    }
    
    console.log('✅ Model initialized successfully!');
    console.log();
    
    // Streaming generation
    console.log('💬 Starting streaming generation...');
    const prompt = "Write a short story about a robot learning to paint";
    console.log(`📝 Prompt: ${prompt}`);
    console.log('🤖 Response:');
    
    let isFirstChunk = true;
    
    llm.generateStream(prompt, (text) => {
        if (isFirstChunk) {
            process.stdout.write(text);
            isFirstChunk = false;
        } else {
            // Only print the new part
            const newText = text.slice(-1);
            process.stdout.write(newText);
        }
    }, 200);
    
    console.log('\n\n✅ Streaming completed!');
}

// Run example
if (require.main === module) {
    streamingExample().catch(console.error);
}

module.exports = { streamingExample }; 