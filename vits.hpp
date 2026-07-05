#include "ggml_extend.hpp"
#include "models.hpp"
#include "cJSON.h"
#include "util.h"

#include "ggml-backend.h"

namespace vits
{

	typedef struct SynthesizerInfo
	{
		int n_vocab;
		int inter_channels;
		int hidden_channels;
		int filter_channels;
		int n_heads;
		int n_layers;
		int kernel_size;

		int gin_channels;

		int resblock;
		int upsample_initial_channel;
		std::vector<int> upsample_rates;
		std::vector<int> upsample_kernel_sizes;
		std::vector<int> resblock_kernel_sizes;
		std::vector<std::vector<int>> resblock_dilation_sizes;

		bool use_sdp;

	} SynthesizerInfo;

	class TextEncoderRn : public GGMLRunner
	{
	public:
		TextEncoder te;

		TextEncoderRn(ggml_backend_t backend, std::map<std::string, enum ggml_type>& tensor_types,
					  const SynthesizerInfo* xn)
			: GGMLRunner(backend),
			te(xn->n_vocab, xn->inter_channels, xn->hidden_channels, xn->filter_channels, xn->n_heads, xn->n_layers, xn->kernel_size)
		{
			te.init(params_ctx, tensor_types, "enc_p");
		}

		std::string get_desc()
		{
			return "TextEncoderRn";
		}

		void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix)
		{
			te.get_param_tensors(tensors, prefix + "enc_p");
		}

		struct ggml_cgraph* build_graph(struct ggml_tensor* x)
		{
			struct ggml_cgraph* gf = ggml_new_graph(compute_ctx);
			x = to_backend(x);

			auto xmlogs = te.forward(compute_ctx, x);
			ggml_tensor* x_t = xmlogs[0];
			ggml_tensor* mlog_t = xmlogs[1];

			ggml_tensor* comp = ggml_concat(compute_ctx, x_t, mlog_t, 1);

			ggml_build_forward_expand(gf, comp);
			return gf;
		}

		void compute(const int n_threads,
					 struct ggml_tensor* x,
					 ggml_tensor** output,
					 ggml_context* output_ctx = NULL)
		{
			auto get_graph = [&]() -> struct ggml_cgraph* {
				return build_graph(x);
			};
			GGMLRunner::compute(get_graph, n_threads, true, output, output_ctx);
		}
	};

	class StochasticDurationPredictorRn : public GGMLRunner
	{
	public:

		StochasticDurationPredictor dp;

		StochasticDurationPredictorRn(ggml_backend_t backend, std::map<std::string, enum ggml_type>& tensor_types,
									  const SynthesizerInfo* xn)
			: GGMLRunner(backend),
			dp(xn->hidden_channels, 192, 3, 4, xn->gin_channels)
		{
			dp.init(params_ctx, tensor_types, "dp");
		}

		std::string get_desc()
		{
			return "SDPRn";
		}

		void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix)
		{
			dp.get_param_tensors(tensors, prefix + "dp");
		}

		struct ggml_cgraph* build_graph(struct ggml_tensor* xmlogs, struct ggml_tensor* emb_g, float nsw, float ls)
		{
			struct ggml_cgraph* gf = ggml_new_graph(compute_ctx);
			xmlogs = to_backend(xmlogs);
			if (emb_g)
				emb_g = to_backend(emb_g);

			ggml_tensor* x = ggml_view_2d_ext(compute_ctx, xmlogs, 0, xmlogs->ne[0], 0, xmlogs->ne[1] / 3);
			x = ggml_cont(compute_ctx, x);

			emb_g = ggml_reshape_2d(compute_ctx, emb_g, 1, ggml_nelements(emb_g));
			auto logw = dp.forward(compute_ctx, x, emb_g, nsw);
			ggml_tensor* w = ggml_exp_inplace(compute_ctx, logw);
			w = ggml_scale_inplace(compute_ctx, w, ls);

			ggml_build_forward_expand(gf, w);
			return gf;
		}

		void compute(const int n_threads,
					 struct ggml_tensor* x,
					 struct ggml_tensor* emb_g,
					 float nsw,
					 float ls,
					 ggml_tensor** output,
					 ggml_context* output_ctx = NULL)
		{
			auto get_graph = [&]() -> struct ggml_cgraph* {
				return build_graph(x, emb_g, nsw, ls);
			};
			GGMLRunner::compute(get_graph, n_threads, true, output, output_ctx);
		}
	};

	class DurationPredictorRn : public GGMLRunner
	{
	public:

		DurationPredictor dp;

		DurationPredictorRn(ggml_backend_t backend, std::map<std::string, enum ggml_type>& tensor_types,
							const SynthesizerInfo* xn)
			: GGMLRunner(backend),
			dp(xn->hidden_channels, 256, 3, xn->gin_channels)
		{
			dp.init(params_ctx, tensor_types, "dp");
		}

		std::string get_desc()
		{
			return "DPRn";
		}

		void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix)
		{
			dp.get_param_tensors(tensors, prefix + "dp");
		}

		struct ggml_cgraph* build_graph(struct ggml_tensor* xmlogs, struct ggml_tensor* emb_g, float nsw, float ls)
		{
			struct ggml_cgraph* gf = ggml_new_graph(compute_ctx);
			xmlogs = to_backend(xmlogs);
			if (emb_g)
				emb_g = to_backend(emb_g);

			ggml_tensor* x = ggml_view_2d_ext(compute_ctx, xmlogs, 0, xmlogs->ne[0], 0, xmlogs->ne[1] / 3);
			x = ggml_cont(compute_ctx, x);

			emb_g = ggml_reshape_2d(compute_ctx, emb_g, 1, ggml_nelements(emb_g));
			auto logw = dp.forward(compute_ctx, x, emb_g);
			ggml_tensor* w = ggml_exp_inplace(compute_ctx, logw);
			w = ggml_scale_inplace(compute_ctx, w, ls);

			ggml_build_forward_expand(gf, w);
			return gf;
		}

		void compute(const int n_threads,
					 struct ggml_tensor* x,
					 struct ggml_tensor* emb_g,
					 float nsw, //ignored
					 float ls,
					 ggml_tensor** output,
					 ggml_context* output_ctx = NULL)
		{
			auto get_graph = [&]() -> struct ggml_cgraph* {
				return build_graph(x, emb_g, nsw, ls);
			};
			GGMLRunner::compute(get_graph, n_threads, true, output, output_ctx);
		}
	};

	class FlowDecRn : public GGMLRunner
	{
	public:

		Generator dec;
		ResidualCouplingBlock flow;

		FlowDecRn(ggml_backend_t backend, std::map<std::string, enum ggml_type>& tensor_types,
				  const SynthesizerInfo* xn)
			: GGMLRunner(backend),
			flow(xn->inter_channels, xn->hidden_channels, 5, 1, 4, 4, xn->gin_channels),
			dec(xn->inter_channels, xn->resblock, xn->upsample_initial_channel, xn->upsample_rates, xn->upsample_kernel_sizes
				, xn->resblock_kernel_sizes, xn->resblock_dilation_sizes
				, xn->gin_channels)
		{
			flow.init(params_ctx, tensor_types, "flow");
			dec.init(params_ctx, tensor_types, "dec");
		}

		std::string get_desc()
		{
			return "FlowDecRn";
		}

		void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix)
		{
			dec.get_param_tensors(tensors, prefix + "dec");
			flow.get_param_tensors(tensors, prefix + "flow");
		}

		struct ggml_cgraph* build_graph(struct ggml_tensor* xmlogs, struct ggml_tensor* w_cumsum, struct ggml_tensor* emb_g, float ns, float y_length)
		{
			// DecFlow has exceeded the default graph size
			struct ggml_cgraph* gf = ggml_new_graph_custom(compute_ctx, GGML_DEFAULT_GRAPH_SIZE * 2, false);
			xmlogs = to_backend(xmlogs);
			w_cumsum = to_backend(w_cumsum);
			if (emb_g)
				emb_g = to_backend(emb_g);

			ggml_tensor* m = ggml_view_2d_ext(compute_ctx, xmlogs, 0, xmlogs->ne[0], xmlogs->ne[1] / 3, xmlogs->ne[1] / 3);
			ggml_tensor* logs = ggml_view_2d_ext(compute_ctx, xmlogs, 0, xmlogs->ne[0], 2 * xmlogs->ne[1] / 3, xmlogs->ne[1] / 3);

			ggml_tensor* attn = ggml_gen_seq_mask(compute_ctx, w_cumsum, y_length);
			attn = ggml_sub_inplace(compute_ctx, attn, ggml_view_2d_ext(compute_ctx,
																		ggml_pad_2d_ext(compute_ctx, attn,
																						0, 0,
																						1, 0),
																		0, attn->ne[0],
																		0, attn->ne[1]));
			attn = ggml_cont(compute_ctx, ggml_transpose(compute_ctx, attn));

			m = ggml_mul_mat(compute_ctx, attn, m);
			logs = ggml_mul_mat(compute_ctx, attn, logs);

			ggml_tensor* z = ggml_tensor_set_qck_randn(compute_ctx, m, ns);

			z = ggml_mul_inplace(compute_ctx, z, ggml_exp_inplace(compute_ctx, logs));
			z = ggml_add_inplace(compute_ctx, z, m);

			emb_g = ggml_reshape_2d(compute_ctx, emb_g, 1, ggml_nelements(emb_g));
			z = flow.forward(compute_ctx, z, emb_g, true);

			ggml_tensor* out = dec.forward(compute_ctx, z, emb_g);

			ggml_build_forward_expand(gf, out);
			return gf;
		}

		void compute(const int n_threads,
					 struct ggml_tensor* xmlogs,
					 struct ggml_tensor* w_cumsum,
					 struct ggml_tensor* emb_g,
					 float ns,
					 float y_length,
					 ggml_tensor** output,
					 ggml_context* output_ctx = NULL)
		{
			auto get_graph = [&]() -> struct ggml_cgraph* {
				return build_graph(xmlogs, w_cumsum, emb_g, ns, y_length);
			};
			GGMLRunner::compute(get_graph, n_threads, true, output, output_ctx);
		}
	};

	class SynthesizerGGML
	{
	public:

		ggml_backend_t backend = NULL;
		ggml_context* tmpctx = NULL;

		std::shared_ptr<TextEncoderRn> enc_p;
		std::shared_ptr<StochasticDurationPredictorRn> sdp;
		std::shared_ptr<DurationPredictorRn> dp;
		std::shared_ptr<FlowDecRn> decflow;
		std::shared_ptr<Embedding> emb_g;

		std::wstring tokens;

		std::unordered_map<std::wstring, int> speakers;
		std::vector<std::wstring> cleaners;

		bool add_blank = false;
		bool cleaned_text = true;
		bool using_weight_norm = false;
		int sampling_rate = 22050;

		int n_threads = 8;

		SynthesizerInfo info;

		SynthesizerGGML()
		{
			backend = ggml_backend_init_best();
			n_threads = get_num_physical_cores();
		}

		~SynthesizerGGML()
		{
			if (tmpctx)
				ggml_free(tmpctx);
			if (backend)
				ggml_backend_free(backend);
		}

		void load_config_from_file(std::string cpath)
		{
			FILE* fc = fopen(cpath.c_str(), "rb");
			fseek(fc, 0, SEEK_END);
			uint64_t len = ftell(fc);
			rewind(fc);
			char* dat = new char[len + 1];
			fread(dat, 1, len, fc);
			fclose(fc);
			dat[len] = '\0';
			std::wstring data = utf8_to_utf16(dat);

			cJSON* desc = cJSON_Parse(data.c_str());
			cJSON* model = cJSON_GetObjectItem(desc, L"model");
			info.inter_channels = (cJSON_GetObjectItem(model, L"inter_channels")->valueint);
			info.hidden_channels = (cJSON_GetObjectItem(model, L"hidden_channels")->valueint);
			info.filter_channels = (cJSON_GetObjectItem(model, L"filter_channels")->valueint);
			info.n_heads = (cJSON_GetObjectItem(model, L"n_heads")->valueint);
			info.n_layers = (cJSON_GetObjectItem(model, L"n_layers")->valueint);
			info.kernel_size = (cJSON_GetObjectItem(model, L"kernel_size")->valueint);
			info.gin_channels = (cJSON_GetObjectItem(model, L"gin_channels")->valueint);
			info.upsample_initial_channel = (cJSON_GetObjectItem(model, L"upsample_initial_channel")->valueint);
			info.resblock = (cJSON_GetObjectItem(model, L"resblock")->valuestring)[0] == L'1' ? 1 : 0;

			cJSON* use_sdp_v = cJSON_GetObjectItem(model, L"use_sdp");
			if (use_sdp_v != NULL)
				info.use_sdp = use_sdp_v->type != cJSON_False;
			else
				info.use_sdp = true;

			// required
			cJSON* rcbkrs = cJSON_GetObjectItem(model, L"resblock_kernel_sizes");
			int num = cJSON_GetArraySize(rcbkrs);
			for (int i = 0; i < num; i++)
			{
				info.resblock_kernel_sizes.push_back(cJSON_GetArrayItem(rcbkrs, i)->valueint);
			}

			cJSON* uprts = cJSON_GetObjectItem(model, L"upsample_rates");
			num = cJSON_GetArraySize(uprts);
			for (int i = 0; i < num; i++)
			{
				info.upsample_rates.push_back(cJSON_GetArrayItem(uprts, i)->valueint);
			}

			cJSON* upkrs = cJSON_GetObjectItem(model, L"upsample_kernel_sizes");
			num = cJSON_GetArraySize(upkrs);
			for (int i = 0; i < num; i++)
			{
				info.upsample_kernel_sizes.push_back(cJSON_GetArrayItem(upkrs, i)->valueint);
			}

			cJSON* rcbds = cJSON_GetObjectItem(model, L"resblock_dilation_sizes");
			num = cJSON_GetArraySize(rcbds);
			for (int i = 0; i < num; i++)
			{
				cJSON* thi = cJSON_GetArrayItem(rcbds, i);
				int num2 = cJSON_GetArraySize(thi);
				std::vector<int> tmp;

				for (int j = 0; j < num2; j++)
				{
					tmp.push_back(cJSON_GetArrayItem(thi, j)->valueint);
				}
				info.resblock_dilation_sizes.push_back(tmp);
			}

			cJSON* data1 = cJSON_GetObjectItem(desc, L"data");
			add_blank = cJSON_GetObjectItem(data1, L"add_blank")->type == cJSON_True;
			cleaned_text = cJSON_GetObjectItem(data1, L"cleaned_text")->type == cJSON_True;
			sampling_rate = cJSON_GetObjectItem(data1, L"sampling_rate")->valueint;
			cJSON* cleaners1 = cJSON_GetObjectItem(data1, L"text_cleaners");
			num = cJSON_GetArraySize(cleaners1);
			for (int i = 0; i < num; i++)
			{
				cJSON* thi = cJSON_GetArrayItem(cleaners1, i);
				cleaners.push_back(thi->valuestring);
			}

			cJSON* speakers1 = cJSON_GetObjectItem(desc, L"speakers");
			if (speakers1 != NULL)
			{
				num = cJSON_GetArraySize(speakers1);
				speakers.reserve(num);
				for (int i = 0; i < num; i++)
				{
					cJSON* thi = cJSON_GetArrayItem(speakers1, i);
					speakers[std::wstring(thi->string)] = thi->valueint;
				}
			}

			cJSON* symbols = cJSON_GetObjectItem(desc, L"symbols");
			num = cJSON_GetArraySize(symbols);
			tokens.reserve(num);
			for (int i = 0; i < num; i++)
			{
				tokens.push_back(cJSON_GetArrayItem(symbols, i)->valuestring[0]);
			}
			info.n_vocab = num;

			cJSON_Delete(desc);
		}

		bool load_model_from_file(std::string mpath)
		{
			struct ggml_init_params params = {
					   speakers.size() * info.gin_channels * sizeof(float) + 1024,
					   NULL,
					   false
			};

			tmpctx = ggml_init(params);

			ModelLoader ml;
			ml.init_from_file(mpath);

			enc_p = std::shared_ptr<TextEncoderRn>(new TextEncoderRn(backend, ml.tensor_storages_types, &info));

			if (info.use_sdp)
				sdp = std::shared_ptr<StochasticDurationPredictorRn>(new StochasticDurationPredictorRn(backend, ml.tensor_storages_types, &info));
			else
				dp = std::shared_ptr<DurationPredictorRn>(new DurationPredictorRn(backend, ml.tensor_storages_types, &info));

			decflow = std::shared_ptr<FlowDecRn>(new FlowDecRn(backend, ml.tensor_storages_types, &info));

			if (speakers.size() > 0)
			{
				emb_g = std::shared_ptr<Embedding>(new Embedding(speakers.size(), info.gin_channels));
				emb_g->init(tmpctx, ml.tensor_storages_types, "emb_g");
			}

			if (!enc_p->alloc_params_buffer())
			{
				LOG_ERROR("Failed to allocate TextEncoder params");
				return false;
			}

			if (info.use_sdp)
			{
				if (!sdp->alloc_params_buffer())
				{
					LOG_ERROR("Failed to allocate StochasticDurationPredictor params");
					return false;
				}
			}
			else
			{
				if (!dp->alloc_params_buffer())
				{
					LOG_ERROR("Failed to allocate DurationPredictor params");
					return false;
				}
			}

			if (!decflow->alloc_params_buffer())
			{
				LOG_ERROR("Failed to allocate FlowDec params");
				return false;
			}

			std::map<std::string, ggml_tensor*> tmap;
			enc_p->get_param_tensors(tmap, "");

			if (info.use_sdp)
				sdp->get_param_tensors(tmap, "");
			else
				dp->get_param_tensors(tmap, "");

			decflow->get_param_tensors(tmap, "");
			emb_g->get_param_tensors(tmap, "emb_g");

			std::set<std::string> ignored = { "enc_q" /*no supported*/,
											  "dp.post" /*training only*/ };

			if (tmap.find("dec.ups.0.weight_v") != tmap.end())
			{
				LOG_WARN("Using weight normalized model will lead to more parameter memory cost.");
				using_weight_norm = true;
			}

			bool ok = ml.load_tensors(tmap, backend, ignored);
			if (ok)
				LOG_INFO("load model from %s successful", mpath.c_str());
			else
				LOG_ERROR("load model from %s failed", mpath.c_str());

			return ok;
		}

		int64_t minimum_phoneme_length()
		{
			int64_t r = 1;

			for (int i : info.upsample_rates)
			{
				r *= i;
			}
			return r;
		}

		std::vector<int> tokenize(std::wstring marks)
		{
			if (add_blank)
			{
				std::vector<int> pos;
				pos.reserve(sizeof(int) * (marks.size() * 2 + 1));
				pos.push_back(0);

				for (wchar_t i : marks)
				{
					int tok = tokens.find(i);
					if (tok == -1)
					{
						LOG_WARN("Unknown token : %s , Default to blank.", utf16_to_utf8({ i }).c_str());
						tok = 0;
					}
					pos.push_back(tok);
					pos.push_back(0);
				}
				return pos;
			}
			else
			{
				std::vector<int> pos;
				pos.reserve(sizeof(int) * (marks.size()));

				for (wchar_t i : marks)
				{
					int tok = tokens.find(i);
					if (tok == -1)
					{
						LOG_WARN("Unknown token : %s , Default to blank.", utf16_to_utf8({ i }).c_str());
						tok = 0;
					}
					pos.push_back(tok);
				}
				return pos;
			}
		}

		float* infer(std::vector<int> ids, std::wstring speaker, size_t* sizeout, uint32_t** dur, float ls, float ns, float nsw)
		{
			struct ggml_init_params params = {
					   64 * 1024 * 1024,
					   NULL,
					   false
			};
			ggml_context* tmp = ggml_init(params);

			ggml_tensor* ids_t = ggml_new_tensor_1d(tmp, GGML_TYPE_I32, ids.size());
			memcpy(ids_t->data, ids.data(), ids.size() * sizeof(float));

			ggml_tensor* embg = NULL;

			if (speakers.size() > 0)
			{
				embg = ggml_new_tensor_1d(tmp, GGML_TYPE_F32, info.gin_channels);
				ggml_tensor* embg_w = emb_g->get_weight();

				int64_t sid = 0;
				auto sidz = speakers.find(speaker);
				if (sidz == speakers.end())
					LOG_WARN("Unknown speaker : %s, Default to the first one.", utf16_to_utf8(speaker).c_str());
				else
					sid = sidz->second;

				const float* embg_ri = (const float*)embg_w->data + sid * (uint64_t)info.gin_channels;
				float* embg_ro = (float*)embg->data;
				memcpy(embg_ro, embg_ri, info.gin_channels * sizeof(float));
			}

			ggml_tensor* xmlogs = 0;
			enc_p->compute(n_threads, ids_t, &xmlogs, tmp);

			ggml_tensor* dp_x = xmlogs;

			ggml_tensor* w = 0;
			if (info.use_sdp)
				sdp->compute(n_threads, dp_x, embg, nsw, ls, &w, tmp);
			else
				dp->compute(n_threads, dp_x, embg, nsw, ls, &w, tmp);

			w = ggml_tensor_ceil(w);

			ggml_tensor* w_cumsum = ggml_tensor_cumsum(w);
			float y_length = ggml_tensor_get_f32(w, w->ne[0] - 1);

			ggml_tensor* result = 0;
			decflow->compute(n_threads, xmlogs, w_cumsum, embg, ns, y_length, &result, tmp);

			float* arr = new float[result->ne[0]];
			memcpy(arr, result->data, sizeof(float) * result->ne[0]);

			if (dur)
			{
				uint32_t* durs = new uint32_t[w->ne[0]];
				int64_t min_pl = minimum_phoneme_length();

				for (int64_t i = 0; i < w->ne[0]; i++)
				{
					durs[i] = ggml_tensor_get_f32(w, i) * min_pl;
				}
				*dur = durs;
			}

			*sizeout = result->ne[0];
			ggml_free(tmp);

			return arr;
		}

		float* infer(std::wstring mark, std::wstring speaker, size_t* sizeout, uint32_t** dur, float ls, float ns, float nsw)
		{
			auto ids = tokenize(mark);
			return infer(ids, speaker, sizeout, dur, ls, ns, nsw);
		}
	};

}