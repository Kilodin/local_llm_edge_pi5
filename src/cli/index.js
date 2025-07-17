#!/usr/bin/env node

const { Command } = require('commander');
const chalk = require('chalk');
const readline = require('readline');
const fs = require('fs-extra');
const path = require('path');

// Import the native addon
const { LLMNodeBinding } = require('../../build/Release/llm_node');

class LLMCLI {
    constructor() {
        this.program = new Command();
        this.llm = new LLMNodeBinding();
        this.isInitialized = false;
        this.rl = null;
        
        this.setupCommands();
    }
    
    setupCommands() {
        this.program
            .name('local-llm')
            .description('Local LLM inference using llama.cpp')
            .version('1.0.0');
        
        // Initialize command
        this.program
            .command('init')
            .description('Initialize the LLM model')
            .requiredOption('-m, --model <path>', 'Path to the model file')
            .option('-c, --context <size>', 'Context size', '2048')
            .option('-b, --batch <size>', 'Batch size', '512')
            .option('-t, --threads <count>', 'Number of threads', '4')
            .option('--gpu-layers <count>', 'Number of GPU layers', '0')
            .option('--temp <temperature>', 'Temperature', '0.7')
            .option('--top-p <value>', 'Top-p value', '0.9')
            .option('--top-k <value>', 'Top-k value', '40')
            .option('--repeat-penalty <value>', 'Repeat penalty', '1.1')
            .option('--seed <value>', 'Random seed', '42')
            .action(async (options) => {
                await this.initializeModel(options);
            });
        
        // Chat command
        this.program
            .command('chat')
            .description('Start interactive chat session')
            .action(() => {
                this.startChat();
            });
        
        // Generate command
        this.program
            .command('generate')
            .description('Generate text from prompt')
            .argument('<prompt>', 'Input prompt')
            .option('-m, --max-tokens <count>', 'Maximum tokens to generate', '256')
            .action(async (prompt, options) => {
                await this.generateText(prompt, options.maxTokens);
            });
        
        // Info command
        this.program
            .command('info')
            .description('Show system and model information')
            .action(() => {
                this.showInfo();
            });
        
        // Parameters command
        this.program
            .command('params')
            .description('Update generation parameters')
            .option('--temp <temperature>', 'Temperature')
            .option('--top-p <value>', 'Top-p value')
            .option('--top-k <value>', 'Top-k value')
            .option('--repeat-penalty <value>', 'Repeat penalty')
            .action((options) => {
                this.updateParameters(options);
            });
        
        // Server command
        this.program
            .command('server')
            .description('Start the web server')
            .option('-p, --port <port>', 'Port number', '3001')
            .action((options) => {
                this.startServer(options.port);
            });
    }
    
    async initializeModel(options) {
        try {
            console.log(chalk.blue('🔧 Initializing LLM model...'));
            
            // Check if model file exists
            if (!await fs.pathExists(options.model)) {
                console.error(chalk.red(`❌ Model file not found: ${options.model}`));
                process.exit(1);
            }
            
            const config = {
                modelPath: options.model,
                contextSize: parseInt(options.context),
                batchSize: parseInt(options.batch),
                threads: parseInt(options.threads),
                gpuLayers: parseInt(options.gpuLayers),
                temperature: parseFloat(options.temp),
                topP: parseFloat(options.topP),
                topK: parseInt(options.topK),
                repeatPenalty: parseFloat(options.repeatPenalty),
                seed: parseInt(options.seed)
            };
            
            const success = this.llm.initialize(config);
            this.isInitialized = success;
            
            if (success) {
                console.log(chalk.green('✅ Model initialized successfully!'));
                console.log(chalk.cyan('\n📊 Model Information:'));
                console.log(this.llm.getModelInfo());
            } else {
                console.error(chalk.red('❌ Failed to initialize model'));
                process.exit(1);
            }
            
        } catch (error) {
            console.error(chalk.red(`❌ Initialization error: ${error.message}`));
            process.exit(1);
        }
    }
    
    async generateText(prompt, maxTokens) {
        try {
            if (!this.isInitialized) {
                console.error(chalk.red('❌ Model not initialized. Run "local-llm init" first.'));
                process.exit(1);
            }
            
            console.log(chalk.blue('🤖 Generating text...'));
            console.log(chalk.yellow(`📝 Prompt: ${prompt}`));
            console.log(chalk.cyan('📤 Response:'));
            
            const result = this.llm.generate(prompt, parseInt(maxTokens));
            console.log(chalk.green(result));
            
        } catch (error) {
            console.error(chalk.red(`❌ Generation error: ${error.message}`));
        }
    }
    
    startChat() {
        if (!this.isInitialized) {
            console.error(chalk.red('❌ Model not initialized. Run "local-llm init" first.'));
            process.exit(1);
        }
        
        console.log(chalk.blue('💬 Starting chat session...'));
        console.log(chalk.cyan('Type "quit" or "exit" to end the session'));
        console.log(chalk.cyan('Type "stop" to stop current generation'));
        console.log('');
        
        this.rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout
        });
        
        this.rl.on('line', async (input) => {
            const trimmed = input.trim();
            
            if (trimmed === 'quit' || trimmed === 'exit') {
                console.log(chalk.blue('👋 Goodbye!'));
                this.rl.close();
                process.exit(0);
            }
            
            if (trimmed === 'stop') {
                this.llm.stopGeneration();
                console.log(chalk.yellow('⏹️  Generation stopped'));
                return;
            }
            
            if (trimmed === '') {
                return;
            }
            
            try {
                console.log(chalk.blue('🤖 Generating...'));
                
                // Use streaming for better UX
                let response = '';
                this.llm.generateStream(trimmed, (text) => {
                    response = text;
                    process.stdout.write(chalk.green(text));
                }, 256);
                
                console.log('\n');
                
            } catch (error) {
                console.error(chalk.red(`❌ Error: ${error.message}`));
            }
        });
        
        this.rl.on('close', () => {
            process.exit(0);
        });
    }
    
    showInfo() {
        console.log(chalk.blue('📊 System Information:'));
        console.log(LLMNodeBinding.getSystemInfo());
        
        if (this.isInitialized) {
            console.log(chalk.blue('\n🤖 Model Information:'));
            console.log(this.llm.getModelInfo());
        } else {
            console.log(chalk.yellow('\n⚠️  Model not initialized'));
        }
    }
    
    updateParameters(options) {
        if (!this.isInitialized) {
            console.error(chalk.red('❌ Model not initialized. Run "local-llm init" first.'));
            process.exit(1);
        }
        
        try {
            if (options.temp !== undefined) {
                this.llm.setTemperature(parseFloat(options.temp));
                console.log(chalk.green(`✅ Temperature set to ${options.temp}`));
            }
            
            if (options.topP !== undefined) {
                this.llm.setTopP(parseFloat(options.topP));
                console.log(chalk.green(`✅ Top-p set to ${options.topP}`));
            }
            
            if (options.topK !== undefined) {
                this.llm.setTopK(parseInt(options.topK));
                console.log(chalk.green(`✅ Top-k set to ${options.topK}`));
            }
            
            if (options.repeatPenalty !== undefined) {
                this.llm.setRepeatPenalty(parseFloat(options.repeatPenalty));
                console.log(chalk.green(`✅ Repeat penalty set to ${options.repeatPenalty}`));
            }
            
        } catch (error) {
            console.error(chalk.red(`❌ Parameter update error: ${error.message}`));
        }
    }
    
    startServer(port) {
        console.log(chalk.blue(`🚀 Starting server on port ${port}...`));
        
        // Import and start the server
        const LLMServer = require('../server/index.js');
        const server = new LLMServer();
        server.start(parseInt(port));
    }
    
    run() {
        this.program.parse();
    }
}

// Run CLI if this file is executed directly
if (require.main === module) {
    const cli = new LLMCLI();
    cli.run();
}

module.exports = LLMCLI; 