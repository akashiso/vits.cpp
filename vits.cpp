#include "vits.h"
#include "vits.hpp"

typedef struct SynthesizerGGMLn
{
	vits::SynthesizerGGML synth;
	SynthesizerInfo info;
} SynthesizerGGMLn;

static bool verbose = false;

SynthesizerGGML* vits_init_backend()
{
	if (ggml_backend_reg_count() == 0)
	{
		printf(__FUNCTION__ " : No backends are loaded. use ggml_backend_load() or ggml_backend_load_all() to load a backend before calling this function\n");
		return NULL;
	}

	log_cb_t a;
	get_log_callback(&a, NULL);
	if (a == NULL)
		vits_set_log_callback(NULL, NULL);

	return new SynthesizerGGMLn();
}

void vits_load_config_from_file(SynthesizerGGML* syn, std::string cpath)
{
	syn->synth.load_config_from_file(cpath);
	syn->info.add_blank = syn->synth.add_blank;
	syn->info.cleaned_text = syn->synth.cleaned_text;
	syn->info.cleaners = syn->synth.cleaners;
	syn->info.sampling_rate = syn->synth.sampling_rate;
	syn->info.speakers = syn->synth.speakers;
	syn->info.tokens = syn->synth.tokens;

	syn->info.filter_channels = syn->synth.info.filter_channels;
	syn->info.gin_channels = syn->synth.info.gin_channels;
	syn->info.hidden_channels = syn->synth.info.hidden_channels;
	syn->info.inter_channels = syn->synth.info.inter_channels;
	syn->info.kernel_size = syn->synth.info.kernel_size;
	syn->info.n_heads = syn->synth.info.n_heads;
	syn->info.n_layers = syn->synth.info.n_layers;
	syn->info.n_vocab = syn->synth.info.n_vocab;
	syn->info.resblock = syn->synth.info.resblock;
	syn->info.resblock_dilation_sizes = syn->synth.info.resblock_dilation_sizes;
	syn->info.resblock_kernel_sizes = syn->synth.info.resblock_kernel_sizes;
	syn->info.upsample_initial_channel = syn->synth.info.upsample_initial_channel;
	syn->info.upsample_kernel_sizes = syn->synth.info.upsample_kernel_sizes;
	syn->info.upsample_rates = syn->synth.info.upsample_rates;
	
	syn->info.use_sdp = syn->synth.info.use_sdp;

	syn->info.minimum_phoneme_length = syn->synth.minimum_phoneme_length();
}

bool vits_load_model_from_file(SynthesizerGGML* syn, std::string mpath)
{
	return syn->synth.load_model_from_file(mpath);
}

void vits_set_log_callback(vits_log_cb_t cb, void* dat)
{
	if (cb != NULL)
		set_log_callback((log_cb_t)cb, dat);
	else
		set_log_callback([](log_level_t level, const char* log, void* dat)
						 {
							 const char* level_str;
							 FILE* out_stream = (level == LOG_ERROR) ? stderr : stdout;

							 if (!log || (!verbose && level <= LOG_DEBUG))
							 {
								 return;
							 }

							 switch (level)
							 {
								 case LOG_DEBUG:
									 level_str = "DEBUG";
									 break;
								 case LOG_INFO:
									 level_str = "INFO";
									 break;
								 case LOG_WARN:
									 level_str = "WARN";
									 break;
								 case LOG_ERROR:
									 level_str = "ERROR";
									 break;
								 default: /* Potential future-proofing */
									 level_str = "?????";
									 break;
							 }

							 fprintf(out_stream, "[%-5s] ", level_str);

							 fputs(log, out_stream);
							 fflush(out_stream);
						 }, NULL);
}

void vits_get_log_callback(vits_log_cb_t* cb, void** dat)
{
	log_cb_t a;
	get_log_callback(&a, dat);
	*cb = (vits_log_cb_t)a;
}

void vits_verbose(bool verbose1)
{
	verbose = verbose1;
}

SynthesizerInfo* vits_info(SynthesizerGGML* syn)
{
	return &syn->info;
}

bool vits_using_weight_norm(SynthesizerGGML* syn)
{
	return syn->synth.using_weight_norm;
}

VITSVersion vits_version(SynthesizerGGML* syn)
{
	return VITS_VERSION_VITS;
}

std::vector<int> vits_tokenize(SynthesizerGGML* syn, std::wstring marks)
{
	return syn->synth.tokenize(marks);
}

float* vits_infer(SynthesizerGGML* syn, std::wstring mark, std::wstring speaker, size_t* sizeout, uint32_t** duration, float ls, float ns, float nsw)
{
	return syn->synth.infer(mark, speaker, sizeout, duration, ls, ns, nsw);
}

float* vits_infer(SynthesizerGGML* syn, std::vector<int> ids, std::wstring speaker, size_t* sizeout, uint32_t** duration, float ls, float ns, float nsw)
{
	return syn->synth.infer(ids, speaker, sizeout, duration, ls, ns, nsw);
}

void vits_free_backend(SynthesizerGGML* syn)
{
	delete syn;
}
