const { LLMNodeBinding } = require('../build/Release/llm_node');

// Mock model path for testing (you'll need to provide a real model)
const TEST_MODEL_PATH = './models/test-model.gguf';

function testSystemInfo() {
    console.log('🧪 Testing system info...');
    const info = LLMNodeBinding.getSystemInfo();
    console.assert(info && info.length > 0, 'System info should not be empty');
    console.log('✅ System info test passed');
}

function testInitialization() {
    console.log('🧪 Testing model initialization...');
    const llm = new LLMNodeBinding();
    
    // Test with invalid model path
    const success = llm.initialize({
        modelPath: '/nonexistent/model.gguf',
        threads: 1,
        contextSize: 512
    });
    
    console.assert(!success, 'Should fail with invalid model path');
    console.log('✅ Initialization test passed');
}

function testParameterUpdates() {
    console.log('🧪 Testing parameter updates...');
    const llm = new LLMNodeBinding();
    
    // These should not throw errors even without model loaded
    llm.setTemperature(0.8);
    llm.setTopP(0.9);
    llm.setTopK(50);
    llm.setRepeatPenalty(1.2);
    
    console.log('✅ Parameter update test passed');
}

function testReadyState() {
    console.log('🧪 Testing ready state...');
    const llm = new LLMNodeBinding();
    
    // Should be false without initialization
    const isReady = llm.isReady();
    console.assert(!isReady, 'Should not be ready without initialization');
    console.log('✅ Ready state test passed');
}

function runAllTests() {
    console.log('🚀 Running LLM System Tests\n');
    
    try {
        testSystemInfo();
        testInitialization();
        testParameterUpdates();
        testReadyState();
        
        console.log('\n🎉 All tests passed!');
    } catch (error) {
        console.error('\n❌ Test failed:', error.message);
        process.exit(1);
    }
}

// Run tests if this file is executed directly
if (require.main === module) {
    runAllTests();
}

module.exports = {
    testSystemInfo,
    testInitialization,
    testParameterUpdates,
    testReadyState,
    runAllTests
}; 