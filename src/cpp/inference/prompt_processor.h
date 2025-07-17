#pragma once

#include <string>
#include <vector>

namespace local_llm {

class PromptProcessor {
public:
    // Process and format prompts for different model types
    static std::string formatPrompt(const std::string& prompt, const std::string& modelType = "llama");
    
    // Extract system message from prompt
    static std::string extractSystemMessage(const std::string& prompt);
    
    // Format conversation history
    static std::string formatConversation(const std::vector<std::pair<std::string, std::string>>& messages);
    
    // Clean and normalize prompt
    static std::string cleanPrompt(const std::string& prompt);
    
    // Split prompt into chunks for processing
    static std::vector<std::string> splitPrompt(const std::string& prompt, size_t maxChunkSize = 1024);
    
    // Detect prompt type (chat, completion, etc.)
    static std::string detectPromptType(const std::string& prompt);

private:
    // Helper methods
    static std::string formatLlamaPrompt(const std::string& prompt);
    static std::string formatChatPrompt(const std::string& prompt);
    static std::string formatCompletionPrompt(const std::string& prompt);
};

} // namespace local_llm 