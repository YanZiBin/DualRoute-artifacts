/*
 * SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 DualRoute authors.
 * Modified by the DualRoute authors for the prompt-excluded release; numerical/parser/guard scope is documented in docs/REPRODUCTION_SCOPE.md.
 */
#include "llama.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

std::string SYSTEM_PROMPT;

struct worker_config {
    std::string model_path;
    int max_new_tokens = 2;
    int context_size = 2048;
    int n_gpu_layers = 0;
};

struct worker_result {
    bool ok = false;
    std::string label;
    std::string raw_prediction;
    std::string error;
};

std::string json_escape(const std::string & input) {
    std::string output;
    output.reserve(input.size() + 8);
    for (char ch : input) {
        switch (ch) {
            case '\\':
                output += "\\\\";
                break;
            case '"':
                output += "\\\"";
                break;
            case '\n':
                output += "\\n";
                break;
            case '\r':
                output += "\\r";
                break;
            case '\t':
                output += "\\t";
                break;
            default:
                output += ch;
                break;
        }
    }
    return output;
}

std::string build_prompt(const std::string & text) {
    return std::string("<|im_start|>system\n") + SYSTEM_PROMPT +
           "<|im_end|>\n<|im_start|>user\n" + text +
           "<|im_end|>\n<|im_start|>assistant\n";
}

bool is_route_label_char(char ch) {
    return ch == '0' || ch == '1' || ch == '2';
}

std::string extract_label(const std::string & raw_prediction) {
    for (size_t index = 0; index < raw_prediction.size(); ++index) {
        const char ch = raw_prediction[index];
        if (!is_route_label_char(ch)) {
            continue;
        }
        const bool left_ok =
            index == 0 || !std::isalnum(static_cast<unsigned char>(raw_prediction[index - 1]));
        const bool right_ok =
            index + 1 >= raw_prediction.size() ||
            !std::isalnum(static_cast<unsigned char>(raw_prediction[index + 1]));
        if (left_ok && right_ok) {
            return std::string(1, ch);
        }
    }

    std::string stripped = raw_prediction;
    stripped.erase(
        std::remove_if(
            stripped.begin(),
            stripped.end(),
            [](unsigned char ch) { return std::isspace(ch) != 0; }),
        stripped.end());
    if (stripped == "0" || stripped == "1" || stripped == "2") {
        return stripped;
    }
    return "";
}

bool parse_args(int argc, char ** argv, worker_config & config) {
    for (int index = 1; index < argc; ++index) {
        const char * arg = argv[index];
        if (std::strcmp(arg, "--model-path") == 0) {
            if (index + 1 >= argc) {
                return false;
            }
            config.model_path = argv[++index];
        } else if (std::strcmp(arg, "--max-new-tokens") == 0) {
            if (index + 1 >= argc) {
                return false;
            }
            config.max_new_tokens = std::atoi(argv[++index]);
        } else if (std::strcmp(arg, "--context-size") == 0) {
            if (index + 1 >= argc) {
                return false;
            }
            config.context_size = std::atoi(argv[++index]);
        } else if (std::strcmp(arg, "--ngl") == 0) {
            if (index + 1 >= argc) {
                return false;
            }
            config.n_gpu_layers = std::atoi(argv[++index]);
        } else if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
            return false;
        } else {
            return false;
        }
    }

    return !config.model_path.empty() && config.max_new_tokens > 0 && config.context_size > 0;
}

void print_usage(const char * argv0) {
    std::fprintf(
        stderr,
        "usage: %s --model-path MODEL.gguf [--max-new-tokens N] [--context-size N] [--ngl N]\n",
        argv0);
}

worker_result predict_once(
    llama_model * model,
    const llama_vocab * vocab,
    const worker_config & config,
    const std::string & text) {
    worker_result result;
    if (text.empty()) {
        result.error = "Input text cannot be empty.";
        return result;
    }

    const std::string prompt = build_prompt(text);
    const int n_prompt = -llama_tokenize(
        vocab,
        prompt.c_str(),
        static_cast<int32_t>(prompt.size()),
        nullptr,
        0,
        true,
        true);
    if (n_prompt <= 0) {
        result.error = "Failed to tokenize prompt.";
        return result;
    }

    std::vector<llama_token> prompt_tokens(static_cast<size_t>(n_prompt));
    if (llama_tokenize(
            vocab,
            prompt.c_str(),
            static_cast<int32_t>(prompt.size()),
            prompt_tokens.data(),
            static_cast<int32_t>(prompt_tokens.size()),
            true,
            true) < 0) {
        result.error = "Failed to tokenize prompt.";
        return result;
    }

    llama_context_params ctx_params = llama_context_default_params();
    const int requested_n_ctx =
        std::max(config.context_size, n_prompt + config.max_new_tokens + 8);
    ctx_params.n_ctx = static_cast<uint32_t>(requested_n_ctx);
    ctx_params.n_batch = static_cast<uint32_t>(std::min(requested_n_ctx, n_prompt));
    ctx_params.no_perf = true;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        result.error = "Failed to create llama context.";
        return result;
    }

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = true;
    llama_sampler * sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    std::string generated;
    generated.reserve(static_cast<size_t>(config.max_new_tokens) * 4);

    int n_pos = 0;
    for (int generated_count = 0; generated_count < config.max_new_tokens;) {
        if (llama_decode(ctx, batch) != 0) {
            result.error = "llama_decode failed.";
            llama_sampler_free(sampler);
            llama_free(ctx);
            return result;
        }

        n_pos += batch.n_tokens;
        llama_token token = llama_sampler_sample(sampler, ctx, -1);
        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }

        char piece_buffer[256];
        const int piece_len = llama_token_to_piece(
            vocab,
            token,
            piece_buffer,
            sizeof(piece_buffer),
            0,
            true);
        if (piece_len < 0) {
            result.error = "Failed to decode token piece.";
            llama_sampler_free(sampler);
            llama_free(ctx);
            return result;
        }

        generated.append(piece_buffer, static_cast<size_t>(piece_len));
        batch = llama_batch_get_one(&token, 1);
        ++generated_count;
    }

    llama_sampler_free(sampler);
    llama_free(ctx);

    const std::string label = extract_label(generated);
    if (label.empty()) {
        result.error = "Could not extract 0/1/2 from worker output.";
        result.raw_prediction = generated;
        return result;
    }

    result.ok = true;
    result.label = label;
    result.raw_prediction = generated;
    return result;
}

void emit_result(const worker_result & result) {
    if (result.ok) {
        std::cout << "{\"label\":\"" << json_escape(result.label)
                  << "\",\"raw_prediction\":\"" << json_escape(result.raw_prediction)
                  << "\"}" << std::endl;
    } else {
        std::cout << "{\"error\":\"" << json_escape(result.error)
                  << "\",\"raw_prediction\":\"" << json_escape(result.raw_prediction)
                  << "\"}" << std::endl;
    }
}

}  // namespace

int main(int argc, char ** argv) {
    worker_config config;
    if (!parse_args(argc, argv, config)) {
        print_usage(argv[0]);
        return 1;
    }

    const char * prompt_path = std::getenv("DUALROUTE_ROUTE_PROMPT_FILE");
    if (prompt_path == nullptr || *prompt_path == '\0') {
        std::fprintf(stderr, "Research prompts are not distributed. Set DUALROUTE_ROUTE_PROMPT_FILE.\n");
        return 1;
    }
    std::ifstream prompt_file(prompt_path, std::ios::binary);
    if (!prompt_file) {
        std::fprintf(stderr, "Cannot read the configured routing prompt file.\n");
        return 1;
    }
    SYSTEM_PROMPT.assign(std::istreambuf_iterator<char>(prompt_file), std::istreambuf_iterator<char>());
    if (std::none_of(SYSTEM_PROMPT.begin(), SYSTEM_PROMPT.end(), [](unsigned char ch) { return !std::isspace(ch); })) {
        std::fprintf(stderr, "The configured routing prompt must be nonempty.\n");
        return 1;
    }

    llama_log_set(
        [](enum ggml_log_level level, const char * text, void * /* user_data */) {
            if (level >= GGML_LOG_LEVEL_ERROR) {
                std::fprintf(stderr, "%s", text);
            }
        },
        nullptr);

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = config.n_gpu_layers;

    llama_model * model = llama_model_load_from_file(config.model_path.c_str(), model_params);
    if (model == nullptr) {
        std::fprintf(stderr, "failed to load model: %s\n", config.model_path.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::string line;
    while (std::getline(std::cin, line)) {
        emit_result(predict_once(model, vocab, config, line));
    }

    llama_model_free(model);
    return 0;
}
