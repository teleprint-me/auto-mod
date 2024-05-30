#include <cstring>
#include <format>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <pthread.h>
#include <string>
#include <sys/wait.h>
#include <wait.h>

void ggml_print_backtrace(void) {
    char attach[32];
    snprintf(attach, sizeof(attach), "attach %d", getpid());
    int pid = fork();
    if (pid == 0) {
        execlp(
            "gdb",
            "gdb",
            "--batch",
            "-ex",
            "set style enabled on",
            "-ex",
            attach,
            "-ex",
            "bt -frame-info source-and-location",
            "-ex",
            "detach",
            "-ex",
            "quit",
            (char*) NULL
        );
    } else {
        waitpid(pid, NULL, 0);
    }
}

#define GGML_ASSERT(x)                                                           \
    do {                                                                         \
        if (!(x)) {                                                              \
            fflush(stdout);                                                      \
            fprintf(stderr, "GGML_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
            ggml_print_backtrace();                                              \
            abort();                                                             \
        }                                                                        \
    } while (0)

enum llm_arch {
    LLM_ARCH_LLAMA,
    LLM_ARCH_FALCON,
    LLM_ARCH_BAICHUAN,
    LLM_ARCH_GROK,
    LLM_ARCH_GPT2,
    LLM_ARCH_GPTJ,
    LLM_ARCH_GPTNEOX,
    LLM_ARCH_MPT,
    LLM_ARCH_STARCODER,
    LLM_ARCH_REFACT,
    LLM_ARCH_BERT,
    LLM_ARCH_NOMIC_BERT,
    LLM_ARCH_JINA_BERT_V2,
    LLM_ARCH_BLOOM,
    LLM_ARCH_STABLELM,
    LLM_ARCH_QWEN,
    LLM_ARCH_QWEN2,
    LLM_ARCH_QWEN2MOE,
    LLM_ARCH_PHI2,
    LLM_ARCH_PHI3,
    LLM_ARCH_PLAMO,
    LLM_ARCH_CODESHELL,
    LLM_ARCH_ORION,
    LLM_ARCH_INTERNLM2,
    LLM_ARCH_MINICPM,
    LLM_ARCH_GEMMA,
    LLM_ARCH_STARCODER2,
    LLM_ARCH_MAMBA,
    LLM_ARCH_XVERSE,
    LLM_ARCH_COMMAND_R,
    LLM_ARCH_DBRX,
    LLM_ARCH_OLMO,
    LLM_ARCH_ARCTIC,
    LLM_ARCH_DEEPSEEK2,
    LLM_ARCH_UNKNOWN,
};

static std::map<llm_arch, std::string> LLM_ARCH_NAMES = {
    {LLM_ARCH_LLAMA, "llama"},
    {LLM_ARCH_FALCON, "falcon"},
    {LLM_ARCH_GROK, "grok"},
    {LLM_ARCH_GPT2, "gpt2"},
    {LLM_ARCH_GPTJ, "gptj"},
    {LLM_ARCH_GPTNEOX, "gptneox"},
    {LLM_ARCH_MPT, "mpt"},
    {LLM_ARCH_BAICHUAN, "baichuan"},
    {LLM_ARCH_STARCODER, "starcoder"},
    {LLM_ARCH_REFACT, "refact"},
    {LLM_ARCH_BERT, "bert"},
    {LLM_ARCH_NOMIC_BERT, "nomic-bert"},
    {LLM_ARCH_JINA_BERT_V2, "jina-bert-v2"},
    {LLM_ARCH_BLOOM, "bloom"},
    {LLM_ARCH_STABLELM, "stablelm"},
    {LLM_ARCH_QWEN, "qwen"},
    {LLM_ARCH_QWEN2, "qwen2"},
    {LLM_ARCH_QWEN2MOE, "qwen2moe"},
    {LLM_ARCH_PHI2, "phi2"},
    {LLM_ARCH_PHI3, "phi3"},
    {LLM_ARCH_PLAMO, "plamo"},
    {LLM_ARCH_CODESHELL, "codeshell"},
    {LLM_ARCH_ORION, "orion"},
    {LLM_ARCH_INTERNLM2, "internlm2"},
    {LLM_ARCH_MINICPM, "minicpm"},
    {LLM_ARCH_GEMMA, "gemma"},
    {LLM_ARCH_STARCODER2, "starcoder2"},
    {LLM_ARCH_MAMBA, "mamba"},
    {LLM_ARCH_XVERSE, "xverse"},
    {LLM_ARCH_COMMAND_R, "command-r"},
    {LLM_ARCH_DBRX, "dbrx"},
    {LLM_ARCH_OLMO, "olmo"},
    {LLM_ARCH_ARCTIC, "arctic"},
    {LLM_ARCH_DEEPSEEK2, "deepseek2"},
    {LLM_ARCH_UNKNOWN, "(unknown)"},
};

enum llm_kv {
    LLM_KV_GENERAL_ARCHITECTURE,
    LLM_KV_GENERAL_QUANTIZATION_VERSION,
    LLM_KV_GENERAL_ALIGNMENT,
    LLM_KV_GENERAL_NAME,
    LLM_KV_GENERAL_AUTHOR,
    LLM_KV_GENERAL_VERSION,
    LLM_KV_GENERAL_URL,
    LLM_KV_GENERAL_DESCRIPTION,
    LLM_KV_GENERAL_LICENSE,
    LLM_KV_GENERAL_SOURCE_URL,
    LLM_KV_GENERAL_SOURCE_HF_REPO,

    LLM_KV_VOCAB_SIZE,
    LLM_KV_CONTEXT_LENGTH,
    LLM_KV_EMBEDDING_LENGTH,
    LLM_KV_BLOCK_COUNT,
    LLM_KV_LEADING_DENSE_BLOCK_COUNT,
    LLM_KV_FEED_FORWARD_LENGTH,
    LLM_KV_EXPERT_FEED_FORWARD_LENGTH,
    LLM_KV_USE_PARALLEL_RESIDUAL,
    LLM_KV_TENSOR_DATA_LAYOUT,
    LLM_KV_EXPERT_COUNT,
    LLM_KV_EXPERT_USED_COUNT,
    LLM_KV_EXPERT_SHARED_COUNT,
    LLM_KV_EXPERT_WEIGHTS_SCALE,
    LLM_KV_POOLING_TYPE,
    LLM_KV_LOGIT_SCALE,

    LLM_KV_ATTENTION_HEAD_COUNT,
    LLM_KV_ATTENTION_HEAD_COUNT_KV,
    LLM_KV_ATTENTION_MAX_ALIBI_BIAS,
    LLM_KV_ATTENTION_CLAMP_KQV,
    LLM_KV_ATTENTION_KEY_LENGTH,
    LLM_KV_ATTENTION_VALUE_LENGTH,
    LLM_KV_ATTENTION_LAYERNORM_EPS,
    LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,
    LLM_KV_ATTENTION_CAUSAL,
    LLM_KV_ATTENTION_Q_LORA_RANK,
    LLM_KV_ATTENTION_KV_LORA_RANK,

    LLM_KV_ROPE_DIMENSION_COUNT,
    LLM_KV_ROPE_FREQ_BASE,
    LLM_KV_ROPE_SCALE_LINEAR,
    LLM_KV_ROPE_SCALING_TYPE,
    LLM_KV_ROPE_SCALING_FACTOR,
    LLM_KV_ROPE_SCALING_ATTN_FACTOR,
    LLM_KV_ROPE_SCALING_ORIG_CTX_LEN,
    LLM_KV_ROPE_SCALING_FINETUNED,
    LLM_KV_ROPE_SCALING_YARN_LOG_MUL,

    LLM_KV_SPLIT_NO,
    LLM_KV_SPLIT_COUNT,
    LLM_KV_SPLIT_TENSORS_COUNT,

    LLM_KV_SSM_INNER_SIZE,
    LLM_KV_SSM_CONV_KERNEL,
    LLM_KV_SSM_STATE_SIZE,
    LLM_KV_SSM_TIME_STEP_RANK,

    LLM_KV_TOKENIZER_MODEL,
    LLM_KV_TOKENIZER_TYPE,
    LLM_KV_TOKENIZER_NORM,
    LLM_KV_TOKENIZER_PRE,
    LLM_KV_TOKENIZER_ADDED,
    LLM_KV_TOKENIZER_VOCAB,
    LLM_KV_TOKENIZER_MERGES,
    LLM_KV_TOKENIZER_TOKEN_TYPE,
    LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT,
    LLM_KV_TOKENIZER_SCORES,
    LLM_KV_TOKENIZER_BOS_ID,
    LLM_KV_TOKENIZER_EOS_ID,
    LLM_KV_TOKENIZER_UNK_ID,
    LLM_KV_TOKENIZER_SEP_ID,
    LLM_KV_TOKENIZER_PAD_ID,
    LLM_KV_TOKENIZER_CLS_ID,
    LLM_KV_TOKENIZER_MASK_ID,
    LLM_KV_TOKENIZER_ADD_BOS,
    LLM_KV_TOKENIZER_ADD_EOS,
    LLM_KV_TOKENIZER_ADD_PREFIX,
    LLM_KV_TOKENIZER_HF_JSON,
    LLM_KV_TOKENIZER_RWKV,
    LLM_KV_TOKENIZER_PREFIX_ID,
    LLM_KV_TOKENIZER_SUFFIX_ID,
    LLM_KV_TOKENIZER_MIDDLE_ID,
    LLM_KV_TOKENIZER_EOT_ID,
};

static std::map<llm_kv, std::string> LLM_KV_NAMES = {
    {LLM_KV_GENERAL_ARCHITECTURE, "general.architecture"},
    {LLM_KV_GENERAL_QUANTIZATION_VERSION, "general.quantization_version"},
    {LLM_KV_GENERAL_ALIGNMENT, "general.alignment"},
    {LLM_KV_GENERAL_NAME, "general.name"},
    {LLM_KV_GENERAL_AUTHOR, "general.author"},
    {LLM_KV_GENERAL_VERSION, "general.version"},
    {LLM_KV_GENERAL_URL, "general.url"},
    {LLM_KV_GENERAL_DESCRIPTION, "general.description"},
    {LLM_KV_GENERAL_LICENSE, "general.license"},
    {LLM_KV_GENERAL_SOURCE_URL, "general.source.url"},
    {LLM_KV_GENERAL_SOURCE_HF_REPO, "general.source.repository"},

    {LLM_KV_VOCAB_SIZE, "%s.vocab_size"},
    {LLM_KV_CONTEXT_LENGTH, "%s.context_length"},
    {LLM_KV_EMBEDDING_LENGTH, "%s.embedding_length"},
    {LLM_KV_BLOCK_COUNT, "%s.block_count"},
    {LLM_KV_LEADING_DENSE_BLOCK_COUNT, "%s.leading_dense_block_count"},
    {LLM_KV_FEED_FORWARD_LENGTH, "%s.feed_forward_length"},
    {LLM_KV_EXPERT_FEED_FORWARD_LENGTH, "%s.expert_feed_forward_length"},
    {LLM_KV_USE_PARALLEL_RESIDUAL, "%s.use_parallel_residual"},
    {LLM_KV_TENSOR_DATA_LAYOUT, "%s.tensor_data_layout"},
    {LLM_KV_EXPERT_COUNT, "%s.expert_count"},
    {LLM_KV_EXPERT_USED_COUNT, "%s.expert_used_count"},
    {LLM_KV_EXPERT_SHARED_COUNT, "%s.expert_shared_count"},
    {LLM_KV_EXPERT_WEIGHTS_SCALE, "%s.expert_weights_scale"},
    {LLM_KV_POOLING_TYPE, "%s.pooling_type"},
    {LLM_KV_LOGIT_SCALE, "%s.logit_scale"},

    {LLM_KV_ATTENTION_HEAD_COUNT, "%s.attention.head_count"},
    {LLM_KV_ATTENTION_HEAD_COUNT_KV, "%s.attention.head_count_kv"},
    {LLM_KV_ATTENTION_MAX_ALIBI_BIAS, "%s.attention.max_alibi_bias"},
    {LLM_KV_ATTENTION_CLAMP_KQV, "%s.attention.clamp_kqv"},
    {LLM_KV_ATTENTION_KEY_LENGTH, "%s.attention.key_length"},
    {LLM_KV_ATTENTION_VALUE_LENGTH, "%s.attention.value_length"},
    {LLM_KV_ATTENTION_LAYERNORM_EPS, "%s.attention.layer_norm_epsilon"},
    {LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, "%s.attention.layer_norm_rms_epsilon"},
    {LLM_KV_ATTENTION_CAUSAL, "%s.attention.causal"},
    {LLM_KV_ATTENTION_Q_LORA_RANK, "%s.attention.q_lora_rank"},
    {LLM_KV_ATTENTION_KV_LORA_RANK, "%s.attention.kv_lora_rank"},

    {LLM_KV_ROPE_DIMENSION_COUNT, "%s.rope.dimension_count"},
    {LLM_KV_ROPE_FREQ_BASE, "%s.rope.freq_base"},
    {LLM_KV_ROPE_SCALE_LINEAR, "%s.rope.scale_linear"},
    {LLM_KV_ROPE_SCALING_TYPE, "%s.rope.scaling.type"},
    {LLM_KV_ROPE_SCALING_FACTOR, "%s.rope.scaling.factor"},
    {LLM_KV_ROPE_SCALING_ATTN_FACTOR, "%s.rope.scaling.attn_factor"},
    {LLM_KV_ROPE_SCALING_ORIG_CTX_LEN, "%s.rope.scaling.original_context_length"},
    {LLM_KV_ROPE_SCALING_FINETUNED, "%s.rope.scaling.finetuned"},
    {LLM_KV_ROPE_SCALING_YARN_LOG_MUL, "%s.rope.scaling.yarn_log_multiplier"},

    {LLM_KV_SPLIT_NO, "split.no"},
    {LLM_KV_SPLIT_COUNT, "split.count"},
    {LLM_KV_SPLIT_TENSORS_COUNT, "split.tensors.count"},

    {LLM_KV_SSM_CONV_KERNEL, "%s.ssm.conv_kernel"},
    {LLM_KV_SSM_INNER_SIZE, "%s.ssm.inner_size"},
    {LLM_KV_SSM_STATE_SIZE, "%s.ssm.state_size"},
    {LLM_KV_SSM_TIME_STEP_RANK, "%s.ssm.time_step_rank"},

    {LLM_KV_TOKENIZER_MODEL, "tokenizer.model"},
    {LLM_KV_TOKENIZER_TYPE, "tokenizer.type"},
    {LLM_KV_TOKENIZER_NORM, "tokenizer.norm"},
    {LLM_KV_TOKENIZER_PRE, "tokenizer.pre"},
    {LLM_KV_TOKENIZER_ADDED, "tokenizer.added"},
    {LLM_KV_TOKENIZER_VOCAB, "tokenizer.vocab"},
    {LLM_KV_TOKENIZER_MERGES, "tokenizer.merges"},
    {LLM_KV_TOKENIZER_TOKEN_TYPE, "tokenizer.token_type"},
    {LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT, "tokenizer.token_type_count"},
    {LLM_KV_TOKENIZER_SCORES, "tokenizer.scores"},
    {LLM_KV_TOKENIZER_BOS_ID, "tokenizer.bos_token_id"},
    {LLM_KV_TOKENIZER_EOS_ID, "tokenizer.eos_token_id"},
    {LLM_KV_TOKENIZER_UNK_ID, "tokenizer.unknown_token_id"},
    {LLM_KV_TOKENIZER_SEP_ID, "tokenizer.seperator_token_id"},
    {LLM_KV_TOKENIZER_PAD_ID, "tokenizer.padding_token_id"},
    {LLM_KV_TOKENIZER_CLS_ID, "tokenizer.cls_token_id"},
    {LLM_KV_TOKENIZER_MASK_ID, "tokenizer.mask_token_id"},
    {LLM_KV_TOKENIZER_ADD_BOS, "tokenizer.add_bos_token"},
    {LLM_KV_TOKENIZER_ADD_EOS, "tokenizer.add_eos_token"},
    {LLM_KV_TOKENIZER_ADD_PREFIX, "tokenizer.add_space_prefix"},
    {LLM_KV_TOKENIZER_HF_JSON, "tokenizer.huggingface.json"},
    {LLM_KV_TOKENIZER_RWKV, "tokenizer.rwkv.world"},
    {LLM_KV_TOKENIZER_PREFIX_ID, "tokenizer.prefix_token_id"},
    {LLM_KV_TOKENIZER_SUFFIX_ID, "tokenizer.suffix_token_id"},
    {LLM_KV_TOKENIZER_MIDDLE_ID, "tokenizer.middle_token_id"},
    {LLM_KV_TOKENIZER_EOT_ID, "tokenizer.eot_token_id"},
};

std::string getName(llm_kv kv) {
    return LLM_KV_NAMES.at(kv);
}

std::string getArchName(llm_arch arch) {
    return LLM_ARCH_NAMES.at(arch);
}

struct LLM_KV {
    LLM_KV(llm_arch arch) : arch(arch) {}

    llm_arch arch;

    std::string operator()(llm_kv kv) {
        return std::format("{} ({})", getName(kv), getArchName(arch));
    }

  private:
    static std::map<llm_kv, std::string>   LLM_KV_NAMES;
    static std::map<llm_arch, std::string> LLM_ARCH_NAMES;
};

enum gguf_vocab_type {
    GGUF_VOCAB_TYPE_NONE = 0, // Models has no vocab
    GGUF_VOCAB_TYPE_SPM  = 1, // Tokenizer Byte Level BPE with byte fallback
    GGUF_VOCAB_TYPE_BPE  = 2, // GPT-2 tokenizer Byte Level BPE
    GGUF_VOCAB_TYPE_WPM  = 3, // BERT tokenizer Byte Level WordPiece
};

enum gguf_token_type {
    GGUF_TOKEN_TYPE_UNDEFINED    = 0,
    GGUF_TOKEN_TYPE_NORMAL       = 1,
    GGUF_TOKEN_TYPE_UNKNOWN      = 2,
    GGUF_TOKEN_TYPE_CONTROL      = 3,
    GGUF_TOKEN_TYPE_USER_DEFINED = 4,
    GGUF_TOKEN_TYPE_UNUSED       = 5,
    GGUF_TOKEN_TYPE_BYTE         = 6,
};

enum gguf_token_flags {
    GGUF_TOKEN_FLAG_SPECIAL     = 0,
    GGUF_TOKEN_FLAG_NORMALIZED  = 1,
    GGUF_TOKEN_FLAG_LSTRIP      = 2,
    GGUF_TOKEN_FLAG_RSTRIP      = 3,
    GGUF_TOKEN_FLAG_SINGLE_WORD = 4,
};

struct gguf_vocab {
    using id     = int32_t;
    using token  = std::string;
    using ttype  = gguf_token_type;
    using tflags = gguf_token_flags;

    struct token_data {
        token  text;
        float  score;
        ttype  type;
        tflags flags;
    };

    std::string              arch;
    enum gguf_vocab_type     type = GGUF_VOCAB_TYPE_SPM;
    std::vector<std::string> norm;
    std::vector<std::string> pre;

    std::unordered_map<token, id> token_to_id;
    std::vector<token_data>       id_to_token;

    std::vector<id> special_tokens_cache;

    std::map<std::pair<std::string, std::string>, int> bpe_ranks;

    // default LLaMA special tokens
    id special_bos_id  = 1;
    id special_eos_id  = 2;
    id special_unk_id  = 0;
    id special_sep_id  = -1;
    id special_pad_id  = -1;
    id special_cls_id  = -1;
    id special_mask_id = -1;

    int special_add_bos = -1; // -1 unknown, 1 add, 0 don't add.
    int special_add_eos = -1; // -1 unknown, 1 add, 0 don't add.

    id linefeed_id       = 13;
    id special_prefix_id = -1;
    id special_suffix_id = -1;
    id special_middle_id = -1;
    id special_eot_id = -1; // TODO: move above after "eos_id", and here add "file separator" token

    bool add_space_prefix = true;

    int find_bpe_rank(const std::string &token_left, const std::string &token_right) const {
        GGML_ASSERT(token_left.find(' ') == std::string::npos);
        GGML_ASSERT(token_left.find('\n') == std::string::npos);
        GGML_ASSERT(token_right.find(' ') == std::string::npos);
        GGML_ASSERT(token_right.find('\n') == std::string::npos);

        auto it = bpe_ranks.find(std::make_pair(token_left, token_right));
        if (it == bpe_ranks.end()) {
            return -1;
        }

        return it->second;
    }
};

int main(int argc, char* argv[]) {
    if (1 == argc) {
        puts("Usage: vocab [-f <file>] [-v <vocab-type>]");
        return 1;
    }

    const char* const   short_options = "f:v:";
    const struct option long_options[]
        = {{"registry-file", required_argument, nullptr, 'f'},
           {"vocab-type", optional_argument, nullptr, 'v'}};

    int  opt;
    char registry_file_path[1024];
    char vocab_name[5] = "SPM"; // NOTE: avoid name conflict

    while ((opt = getopt_long(argc, argv, short_options, long_options, nullptr)) != -1) {
        switch (opt) {
            case 'f':
                if (optarg == nullptr || strlen(optarg) < 1) {
                    puts("Error: Invalid file path specified.");
                    return 1;
                }

                strcpy(registry_file_path, optarg);
                break;

            case 'v':
                if (optarg == nullptr || strlen(optarg) > 3) {
                    puts("Error: Invalid vocab type specified.");
                    return 1;
                }

                strncpy(vocab_name, optarg, 5);
                break;

            default:
                puts("Usage: vocab [-f <file>] [-v <vocab-type>]");
                return 1;
        }
    }

    std::ifstream  f(registry_file_path);
    nlohmann::json data = nlohmann::json::parse(f);

    if (data.is_null()) {
        puts("Error: Unable to parse registry file.");
        return 1;
    }

    for (const auto &config : data) {
        const std::string model_repo = config["model_repo"];
        const std::string model_arch = config["model_arch"];
        const std::string vocab_type = config["vocab_type"];

        if (vocab_type == vocab_name) {
            std::cout << "Model Repository:\t" << model_repo << '\n';
            std::cout << "Model Architecture:\t" << model_arch << '\n';

            // normalizer and pre_tokenizer are either null or an object
            const nlohmann::json normalizer    = config["normalizer"];
            const nlohmann::json pre_tokenizer = config["pre_tokenizer"];

            // NOTE: Normalizer may be one of null, Sequence, NFC, NFD, NFKC, NFKD...
            // Seems to be null, Sequence, or NFC in most cases
            // Attempt to handle cases where a key may or may not be present and have a null value
            // Not sure whether to default to NFD or NFC if normalizer is null
            // NOTE: The normalizer type key is not guaranteed
            if (!normalizer.is_null()) {
                const std::string norm_type = normalizer["type"];
                std::cout << "Normalizer Type:\t" << norm_type << std::endl;
                // sequence is an array
                if (0 == norm_type.compare("Sequence")) {
                    const nlohmann::json normalizers = normalizer["normalizers"];
                    if (normalizers.is_array()) {
                        for (const auto &norm : normalizers.items()) {
                            std::cout << "Norm Sequence Object:\t" << norm.key() << ":\t"
                                      << norm.value() << std::endl;
                        }
                    }
                } else {
                    // otherwise norm_type is an object potentially containing properties
                    // this varies from model to model
                    // maybe we can just dump the entire object? i don't know yet...
                    for (const auto &norm : normalizer.items()) {
                        std::cout << "Pre Object Pair:\t" << norm.key() << ":\t" << norm.value()
                                  << std::endl;
                    }
                }
            }

            // NOTE: The pre_tokenizer type is not guaranteed
            if (!pre_tokenizer.is_null()) {
                const std::string pre_tokenizer_type = pre_tokenizer["type"];
                std::cout << "Pre-tokenizer Type:" << pre_tokenizer_type << std::endl;
                // sequence is an array
                if (0 == pre_tokenizer_type.compare("Sequence")) {
                    // extract the array
                    // each element in the array is an object
                    const nlohmann::json pretokenizers = pre_tokenizer["pretokenizers"];
                    if (pretokenizers.is_array()) {
                        for (const auto &pre : pretokenizers.items()) {
                            std::cout << "Pre Sequence Object:\t" << pre.key() << ":\t"
                                      << pre.value() << std::endl;
                        }
                    }
                } else {
                    // otherwise pre_tokenizer_type is an object potentially containing properties
                    // this varies from model to model
                    // maybe we can just dump the entire object?
                    // i don't know yet
                    for (const auto &pre : pre_tokenizer.items()) {
                        std::cout << "Pre Object Pair:\t" << pre.key() << ":\t" << pre.value()
                                  << std::endl;
                    }
                }
            }
        }
    }

    return 0;
}
