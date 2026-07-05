#pragma once
#include "modules.hpp"
#include "attentions.hpp"

//verified
class TextEncoder : public GGMLBlock
{
protected:
	int hidden_channels;
	int out_channels;

public:
	TextEncoder(int n_vocab,
				int out_channels,
				int hidden_channels,
				int filter_channels,
				int n_heads,
				int n_layers,
				int kernel_size)
		: hidden_channels(hidden_channels),
		out_channels(out_channels)
	{
		blocks["emb"] = std::shared_ptr<GGMLBlock>(new Embedding(n_vocab, hidden_channels));
		blocks["encoder"] = std::shared_ptr<GGMLBlock>(new Encoder(hidden_channels,
																   filter_channels,
																   n_heads,
																   n_layers,
																   kernel_size));
		blocks["proj"] = std::shared_ptr<GGMLBlock>(new Conv1d(hidden_channels, out_channels * 2, 1));
	}

	//x, mlogs
	struct std::vector<ggml_tensor*> forward(struct ggml_context* ctx, struct ggml_tensor* x)
	{
		auto enc = std::dynamic_pointer_cast<Encoder>(blocks["encoder"]);
		auto emb = std::dynamic_pointer_cast<Embedding>(blocks["emb"]);
		auto conv = std::dynamic_pointer_cast<Conv1d>(blocks["proj"]);

		x = emb->forward(ctx, x);
		x = ggml_scale_inplace(ctx, x, sqrtf(hidden_channels));
		x = ggml_cont(ctx, ggml_transpose(ctx, x));

		x = enc->forward(ctx, x);
		ggml_tensor* stats = conv->forward(ctx, x);

		return { x, stats };
	}
};

class StochasticDurationPredictor : public GGMLBlock
{
protected:
	int n_flows;

public:
	StochasticDurationPredictor(int in_channels,
								int filter_channels,
								int kernel_size,
								int n_flows = 4,
								int gin_channels = 0)
		: n_flows(n_flows)
	{
		blocks["flows.0"] = std::shared_ptr<GGMLBlock>(new ElementwiseAffine(2));

		for (int i = 0; i < n_flows; i++)
		{
			std::string name = "flows." + std::to_string(i * 2 + 1);
			blocks[name] = std::shared_ptr<GGMLBlock>(new ConvFlow(2, filter_channels, kernel_size, 3));
		}

		blocks["post_pre"] = std::shared_ptr<GGMLBlock>(new Conv1d(1, filter_channels, 1));
		blocks["post_proj"] = std::shared_ptr<GGMLBlock>(new Conv1d(filter_channels, filter_channels, 1));
		blocks["post_convs"] = std::shared_ptr<GGMLBlock>(new DDSConv(filter_channels, kernel_size, 3));

		blocks["post_flows.0"] = std::shared_ptr<GGMLBlock>(new ElementwiseAffine(2));

		for (int i = 0; i < n_flows; i++)
		{
			std::string name = "post_flows." + std::to_string(i * 2 + 1);
			blocks[name] = std::shared_ptr<GGMLBlock>(new ConvFlow(2, filter_channels, kernel_size, 3));
		}
		

		blocks["pre"] = std::shared_ptr<GGMLBlock>(new Conv1d(in_channels, filter_channels, 1));
		blocks["proj"] = std::shared_ptr<GGMLBlock>(new Conv1d(filter_channels, filter_channels, 1));
		blocks["convs"] = std::shared_ptr<GGMLBlock>(new DDSConv(filter_channels, kernel_size, 3));

		if (gin_channels != 0)
			blocks["cond"] = std::shared_ptr<GGMLBlock>(new Conv1d(gin_channels, in_channels, 1));
	}

	//only reverse mode is implemented
	struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* g, float noise_scale = 1.0f)
	{
		auto pre = std::dynamic_pointer_cast<Conv1d>(blocks["pre"]);
		auto convs = std::dynamic_pointer_cast<DDSConv>(blocks["convs"]);
		auto proj = std::dynamic_pointer_cast<Conv1d>(blocks["proj"]);

		auto flow0 = std::dynamic_pointer_cast<ElementwiseAffine>(blocks["flows.0"]);

		x = pre->forward(ctx, x);
		if (g)
		{
			auto cond = std::dynamic_pointer_cast<Conv1d>(blocks["cond"]);
			x = ggml_add_inplace(ctx, x, cond->forward(ctx, g));
		}

		x = convs->forward(ctx, x, NULL);
		x = proj->forward(ctx, x);
		
		ggml_tensor* z = ggml_tensor_set_qck_randn(ctx, x->ne[0], 2, x->ne[2], 1, noise_scale);

		ggml_tensor* flip_idx = ggml_gen_rev_idx(ctx, z->ne[1]);

		//remove a useless vflow
		for (int i = n_flows - 1; i > 0; i--)
		{
			std::string name = "flows." + std::to_string(i * 2 + 1);
			auto flow = std::dynamic_pointer_cast<ConvFlow>(blocks[name]);

			z = ggml_get_rows(ctx, z, flip_idx); //flip
			z = flow->forward(ctx, z, x);
		}

		z = ggml_get_rows(ctx, z, flip_idx); //flip
		z = flow0->forward(ctx, z, true);

		return ggml_nn_split_y_2(ctx, z, 1)[0];
	}
};

class DurationPredictor : public GGMLBlock
{
public:
	DurationPredictor(int in_channels, 
					  int filter_channels,
					  int kernel_size, 
					  int gin_channels = 0)
	{
		blocks["conv_1"] = std::shared_ptr<GGMLBlock>(new Conv1d(in_channels, filter_channels, kernel_size, 1, kernel_size / 2));
		blocks["norm_1"] = std::shared_ptr<GGMLBlock>(new LayerNorm(filter_channels));
		blocks["conv_2"] = std::shared_ptr<GGMLBlock>(new Conv1d(filter_channels, filter_channels, kernel_size, 1, kernel_size / 2));
		blocks["norm_2"] = std::shared_ptr<GGMLBlock>(new LayerNorm(filter_channels));
		blocks["proj"]   = std::shared_ptr<GGMLBlock>(new Conv1d(in_channels, filter_channels, kernel_size, 1, kernel_size / 2));

		if (gin_channels != 0)
			blocks["cond"] = std::shared_ptr<GGMLBlock>(new Conv1d(gin_channels, in_channels, 1));
	}

	struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* g)
	{
		auto conv_1 = std::dynamic_pointer_cast<Conv1d>(blocks["conv_1"]);
		auto norm_1 = std::dynamic_pointer_cast<LayerNorm>(blocks["norm_1"]);
		auto conv_2 = std::dynamic_pointer_cast<Conv1d>(blocks["conv_2"]);
		auto norm_2 = std::dynamic_pointer_cast<LayerNorm>(blocks["norm_2"]);
		auto proj = std::dynamic_pointer_cast<Conv1d>(blocks["proj"]);

		if (g)
		{
			auto cond = std::dynamic_pointer_cast<Conv1d>(blocks["cond"]);
			x = ggml_add(ctx, x, cond->forward(ctx, g));
		}
		x = conv_1->forward(ctx, x);
		x = ggml_relu_inplace(ctx, x);
		x = norm_1->forward(ctx, x);

		x = conv_2->forward(ctx, x);
		x = ggml_relu_inplace(ctx, x);
		x = norm_2->forward(ctx, x);

		x = proj->forward(ctx, x);

		return x;
	}
};

class ResidualCouplingBlock : public GGMLBlock
{
protected:
	int n_flows;

public:
	ResidualCouplingBlock(int channels,
						  int hidden_channels,
						  int kernel_size,
						  int dilation_rate,
						  int n_layers,
						  int n_flows = 4,
						  int gin_channels = 0)
		: n_flows(n_flows)
	{
		for (int i = 0; i < n_flows; i++)
		{
			std::string name = "flows." + std::to_string(i * 2);
			blocks[name] = std::shared_ptr<GGMLBlock>(new ResidualCouplingLayer(channels, hidden_channels, kernel_size, 
																				dilation_rate, n_layers, gin_channels, true));
		}
	}

	struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* g, bool reverse)
	{
		if (reverse)
		{
			ggml_tensor* flip_idx = ggml_gen_rev_idx(ctx, x->ne[1]);

			for (int i = n_flows - 1; i >= 0; i--)
			{
				std::string name = "flows." + std::to_string(i * 2);
				auto rcl = std::dynamic_pointer_cast<ResidualCouplingLayer>(blocks[name]);

				x = ggml_get_rows(ctx, x, flip_idx); //flip
				x = rcl->forward(ctx, x, g, reverse)[0];
			}
			return x;
		}
		else
		{
			ggml_tensor* flip_idx = ggml_gen_rev_idx(ctx, x->ne[1]);

			for (int i = 0; i < n_flows; i++)
			{
				std::string name = "flows." + std::to_string(i * 2);
				auto rcl = std::dynamic_pointer_cast<ResidualCouplingLayer>(blocks[name]);

				x = rcl->forward(ctx, x, g, reverse)[0];
				x = ggml_get_rows(ctx, x, flip_idx); //flip
			}
			return x;
		}
	}
};

class PosteriorEncoder : public GGMLBlock
{
protected:
	int out_channels;

public:
	PosteriorEncoder(int in_channels,
					 int out_channels,
					 int hidden_channels,
					 int kernel_size,
					 int dilation_rate,
					 int n_layers,
					 int gin_channels = 0)
		: out_channels(out_channels)
	{

		blocks["pre"] = std::shared_ptr<GGMLBlock>(new Conv1d(in_channels, hidden_channels, 1));
		blocks["enc"] = std::shared_ptr<GGMLBlock>(new WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels));
		blocks["proj"] = std::shared_ptr<GGMLBlock>(new Conv1d(hidden_channels, out_channels * 2, 1));
	}

	struct std::vector<ggml_tensor*> forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* g)
	{
		auto pre = std::dynamic_pointer_cast<Conv1d>(blocks["pre"]);
		auto enc = std::dynamic_pointer_cast<WN>(blocks["enc"]);
		auto proj = std::dynamic_pointer_cast<Conv1d>(blocks["proj"]);

		x = pre->forward(ctx, x);
		x = enc->forward(ctx, x, g);
		ggml_tensor* stats = proj->forward(ctx, x);

		auto mlogs = ggml_nn_split_y_2(ctx, stats, out_channels);
		ggml_tensor* m = mlogs[0];
		ggml_tensor* logs = mlogs[1];
		ggml_tensor* rand = ggml_tensor_set_qck_randn(ctx, m, 1.2f);
		ggml_tensor* z = ggml_add(ctx, m, ggml_mul_inplace(ctx, ggml_exp(ctx, logs), rand));

		return { z, m, logs };
	}
};

class Generator : public GGMLBlock
{
public:
	int num_kernels, num_upsamples, resblock1;

public:
	Generator(int initial_channel, 
			  int resblock, 
			  int upsample_initial_channel,
			  const std::vector<int>& upsample_rates,
			  const std::vector<int>& upsample_kernel_sizes, 
			  const std::vector<int>& resblock_kernel_sizes,
			  const std::vector<std::vector<int>>& resblock_dilation_sizes,
			  int gin_channels = 0)
	{

		num_kernels = resblock_kernel_sizes.size();
		num_upsamples = upsample_rates.size();
		resblock1 = resblock;

		blocks["conv_pre"] = std::shared_ptr<GGMLBlock>(new Conv1d(initial_channel, upsample_initial_channel, 7, 1, 3));
		for (int i = 0; i < upsample_rates.size(); i++)
		{
			std::string name = "ups." + std::to_string(i);
			int u = upsample_rates[i];
			int k = upsample_kernel_sizes[i];
			blocks[name] = std::shared_ptr<GGMLBlock>(new ConvTranspose1d(upsample_initial_channel >> i, upsample_initial_channel >> (i + 1),
																 k, u, (k - u) / 2));
		}

		for (int i = 0; i < upsample_rates.size(); i++)
		{
			int ch = upsample_initial_channel >> (i + 1);
			for (int j = 0; j < resblock_kernel_sizes.size(); j++)
			{
				int k = resblock_kernel_sizes[j];
				const int* d = resblock_dilation_sizes[j].data();
				std::string name = "resblocks." + std::to_string(i * resblock_kernel_sizes.size() + j);

				if (resblock1)
					blocks[name] = std::shared_ptr<GGMLBlock>(new ResBlock1(ch, d, k));
				else
					blocks[name] = std::shared_ptr<GGMLBlock>(new ResBlock2(ch, d, k));
			}
		}
		int ch = upsample_initial_channel >> (upsample_rates.size());
		blocks["conv_post"] = std::shared_ptr<GGMLBlock>(new Conv1d(ch, 1, 7, 1, 3, 1, false));

		if (gin_channels != 0)
			blocks["cond"] = std::shared_ptr<GGMLBlock>(new Conv1d(gin_channels, upsample_initial_channel, 1));
	}

	struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* g)
	{
		auto convpre = std::dynamic_pointer_cast<Conv1d>(blocks["conv_pre"]);
		auto convpost = std::dynamic_pointer_cast<Conv1d>(blocks["conv_post"]);

		x = convpre->forward(ctx, x);
		if (g)
		{
			auto cond = std::dynamic_pointer_cast<Conv1d>(blocks["cond"]);
			x = ggml_add_inplace(ctx, x, cond->forward(ctx, g));
		}
		
		for (size_t i = 0; i < num_upsamples; i++)
		{
			x = ggml_leaky_relu(ctx, x, 0.1, true);
			std::string name = "ups." + std::to_string(i);
			auto ups = std::dynamic_pointer_cast<ConvTranspose1d>(blocks[name]);
			x = ups->forward(ctx, x);

			ggml_tensor* xs = NULL;
			for (size_t j = 0; j < num_kernels; j++)
			{
				std::string name1 = "resblocks." + std::to_string(i * num_kernels + j);

				if (resblock1)
				{
					if (xs == NULL)
						xs = std::dynamic_pointer_cast<ResBlock1>(blocks[name1])->forward(ctx, x);
					else
						xs = ggml_add_inplace(ctx, xs, std::dynamic_pointer_cast<ResBlock1>(blocks[name1])->forward(ctx, x));
				}
				else
				{
					if (xs == NULL)
						xs = std::dynamic_pointer_cast<ResBlock2>(blocks[name1])->forward(ctx, x);
					else
						xs = ggml_add_inplace(ctx, xs, std::dynamic_pointer_cast<ResBlock2>(blocks[name1])->forward(ctx, x));
				}
			}
			x = ggml_scale_inplace(ctx, xs, 1.f / num_kernels);
		}

		x = ggml_leaky_relu(ctx, x, 0.01f, true);
		x = convpost->forward(ctx, x);
		x = ggml_tanh_inplace(ctx, x);

		return x;
	}
};