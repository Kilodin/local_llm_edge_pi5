#include "prompt_processor.h"
#include <sstream>
#include <algorithm>
#include <regex>

namespace local_llm {

std::string PromptProcessor::formatPrompt(const std::string& prompt, const std::string& modelType) {
    std::string cleaned = cleanPrompt(prompt);
    
    if (modelType == "llama" || modelType == "llama2") {
        return formatLlamaPrompt(cleaned);
    } else if (modelType == "chat") {
        return formatChatPrompt(cleaned);
    } else {
        return formatCompletionPrompt(cleaned);
    }
}

std::string PromptProcessor::extractSystemMessage(const std::string& prompt) {
    std::regex systemRegex(R"(\[SYSTEM\](.*?)\[/SYSTEM\])", std::regex::icase);
    std::smatch match;
    
    if (std::regex_search(prompt, match, systemRegex)) {
        return match[1].str();
    }
    
    return "";
}

std::string PromptProcessor::formatConversation(const std::vector<std::pair<std::string, std::string>>& messages) {
    std::ostringstream oss;
    
    for (const auto& [role, content] : messages) {
        if (role == "system") {
            oss << "[INST] <<SYS>>\n" << content << "\n<</SYS>>\n\n";
        } else if (role == "user") {
            oss << "[INST] " << content << " [/INST]";
        } else if (role == "assistant") {
            oss << content << "\n";
        }
    }
    
    return oss.str();
}

std::string PromptProcessor::cleanPrompt(const std::string& prompt) {
    std::string cleaned = prompt;
    
    // Remove excessive whitespace
    std::regex whitespaceRegex(R"(\s+)");
    cleaned = std::regex_replace(cleaned, whitespaceRegex, " ");
    
    // Trim leading and trailing whitespace
    cleaned.erase(0, cleaned.find_first_not_of(" \t\n\r"));
    cleaned.erase(cleaned.find_last_not_of(" \t\n\r") + 1);
    
    return cleaned;
}

std::vector<std::string> PromptProcessor::splitPrompt(const std::string& prompt, size_t maxChunkSize) {
    std::vector<std::string> chunks;
    std::string currentChunk;
    
    std::istringstream iss(prompt);
    std::string word;
    
    while (iss >> word) {
        if (currentChunk.length() + word.length() + 1 > maxChunkSize) {
            if (!currentChunk.empty()) {
                chunks.push_back(currentChunk);
                currentChunk.clear();
            }
        }
        
        if (!currentChunk.empty()) {
            currentChunk += " ";
        }
        currentChunk += word;
    }
    
    if (!currentChunk.empty()) {
        chunks.push_back(currentChunk);
    }
    
    return chunks;
}

std::string PromptProcessor::detectPromptType(const std::string& prompt) {
    std::string lowerPrompt = prompt;
    std::transform(lowerPrompt.begin(), lowerPrompt.end(), lowerPrompt.begin(), ::tolower);
    
    if (lowerPrompt.find("[inst]") != std::string::npos || 
        lowerPrompt.find("[/inst]") != std::string::npos) {
        return "llama";
    }
    
    if (lowerPrompt.find("user:") != std::string::npos || 
        lowerPrompt.find("assistant:") != std::string::npos) {
        return "chat";
    }
    
    return "completion";
}

std::string PromptProcessor::formatLlamaPrompt(const std::string& prompt) {
    // Check if already formatted
    if (prompt.find("[INST]") != std::string::npos) {
        return prompt;
    }
    
    // Simple formatting for Llama models
    return "[INST] " + prompt + " [/INST]";
}

std::string PromptProcessor::formatChatPrompt(const std::string& prompt) {
    // Format as a simple chat prompt
    return "User: " + prompt + "\nAssistant:";
}

std::string PromptProcessor::formatCompletionPrompt(const std::string& prompt) {
    // Return as-is for completion tasks
    return prompt;
}

} // namespace local_llm 