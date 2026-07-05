#pragma once
#include <vector>
#include <string>
#include <unordered_map>

#ifdef VITS_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef VITS_BUILD
#            define VITS_API __declspec(dllexport)
#        else
#            define VITS_API __declspec(dllimport)
#        endif
#    else
#        define VITS_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define VITS_API
#endif

typedef enum VITSVersion
{
	VITS_VERSION_VITS,
	VITS_VERSION_MAX
} VITSVersion;

typedef struct SynthesizerInfo
{
	int n_vocab = 0;
	int inter_channels = 0;
	int hidden_channels = 0;
	int filter_channels = 0;
	int n_heads = 0;
	int n_layers = 0;
	int kernel_size = 0;

	int gin_channels = 0;

	int resblock = 0;
	int upsample_initial_channel = 0;
	std::vector<int> upsample_rates;
	std::vector<int> upsample_kernel_sizes;
	std::vector<int> resblock_kernel_sizes;
	std::vector<std::vector<int>> resblock_dilation_sizes;
	
	bool use_sdp = true;
	
	std::wstring tokens;

	std::unordered_map<std::wstring, int> speakers;
	std::vector<std::wstring> cleaners;

	bool add_blank = false;
	bool cleaned_text = true;
	int sampling_rate = 22050;

	int minimum_phoneme_length = 16 * 16 * 4 * 4;

} SynthesizerInfo;

enum vits_log_level_t
{
	VITS_LOG_DEBUG,
	VITS_LOG_INFO,
	VITS_LOG_WARN,
	VITS_LOG_ERROR
};

typedef struct SynthesizerGGMLn SynthesizerGGML;

typedef void (*vits_log_cb_t)(enum vits_log_level_t level, const char* text, void* data);

VITS_API SynthesizerGGML* vits_init_backend();
VITS_API void vits_load_config_from_file(SynthesizerGGML*, std::string cpath);
VITS_API bool vits_load_model_from_file(SynthesizerGGML*, std::string mpath);
VITS_API void vits_set_log_callback(vits_log_cb_t cb, void* dat);
VITS_API void vits_get_log_callback(vits_log_cb_t* cb, void** dat);
VITS_API void vits_verbose(bool verbose);
VITS_API bool vits_using_weight_norm(SynthesizerGGML*);
VITS_API SynthesizerInfo* vits_info(SynthesizerGGML*);
VITS_API VITSVersion vits_version(SynthesizerGGML*);
VITS_API std::vector<int> vits_tokenize(SynthesizerGGML*, std::wstring marks);
VITS_API float* vits_infer(SynthesizerGGML*, std::wstring mark, std::wstring speaker, size_t* sizeout, uint32_t** pl, float ls, float ns, float nsw);
VITS_API float* vits_infer(SynthesizerGGML*, std::vector<int> tok, std::wstring speaker, size_t* sizeout, uint32_t** pl, float ls, float ns, float nsw);
VITS_API void vits_free_backend(SynthesizerGGML*);
