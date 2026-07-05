#pragma once
#include "ggml_extend.hpp"
#include "transforms.hpp"

class ConvReluNorm : public GGMLBlock
{
protected:
	int n_layers;

public:
	ConvReluNorm(int in_channels,
				 int hidden_channels,
				 int out_channels,
				 int kernel_size,
				 int n_layers)
		: n_layers(n_layers)
	{
		for (int i = 0; i < n_layers; i++)
		{
			std::string name = "conv_layers." + std::to_string(i);
			blocks[name] = std::shared_ptr<GGMLBlock>(new Conv1d(hidden_channels, hidden_channels, kernel_size, 1, kernel_size / 2));

			name = "norm_layers." + std::to_string(i);
			blocks[name] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_channels));
		}
		blocks["proj"] = std::shared_ptr<GGMLBlock>(new Conv1d(hidden_channels, out_channels, 1));
	}

	struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x)
	{
		ggml_tensor* x_ori = x;
		for (int i = 0; i < n_layers; i++)
		{
			std::string name = "conv_layers." + std::to_string(i);
			auto conv_1 = std::dynamic_pointer_cast<Conv1d>(blocks[name]);

			name = "norm_layers." + std::to_string(i);
			auto norm_1 = std::dynamic_pointer_cast<LayerNorm>(blocks[name]);

			ggml_tensor* y = conv_1->forward(ctx, x);
			x = ggml_add_inplace(ctx, x, y);
			x = norm_1->forward(ctx, x);
			x = ggml_relu(ctx, x);
		}
		auto conv_1 = std::dynamic_pointer_cast<Conv1d>(blocks["proj"]);
		x = conv_1->forward(ctx, x);
		x = ggml_add(ctx, x, x_ori);
		return x;
	}
};

class DDSConv : public GGMLBlock
{
protected:
	int n_layers;

public:
	DDSConv(int channels,
			int kernel_size, 
			int n_layers)
		: n_layers(n_layers)
	{
		for (int i = 0; i < n_layers; i++)
		{
			int dilation = pow(kernel_size, i);
			int	padding = (kernel_size * dilation - dilation) / 2;

			std::string name = "convs_sep." + std::to_string(i);
			blocks[name] = std::shared_ptr<GGMLBlock>(new DepthwiseConv1d(channels, channels, kernel_size,
																		  1, padding, dilation));

			name = "convs_1x1." + std::to_string(i);
			blocks[name] = std::shared_ptr<GGMLBlock>(new Conv1d(channels, channels, 1));

			name = "norms_1." + std::to_string(i);
			blocks[name] = std::shared_ptr<GGMLBlock>(new LayerNorm(channels));

			name = "norms_2." + std::to_string(i);
			blocks[name] = std::shared_ptr<GGMLBlock>(new LayerNorm(channels));
		}
	}

	struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* g)
	{
		if (g)
			x = ggml_add(ctx, x, g);

		for (int i = 0; i < n_layers; i++)
		{
			std::string name = "convs_sep." + std::to_string(i);
			auto convs = std::dynamic_pointer_cast<DepthwiseConv1d>(blocks[name]);

			name = "norms_1." + std::to_string(i);
			auto norms_1 = std::dynamic_pointer_cast<LayerNorm>(blocks[name]);

			name = "convs_1x1." + std::to_string(i);
			auto convs11 = std::dynamic_pointer_cast<Conv1d>(blocks[name]);

			name = "norms_2." + std::to_string(i);
			auto norms_2 = std::dynamic_pointer_cast<LayerNorm>(blocks[name]);

			ggml_tensor* y = convs->forward(ctx, x);
			y = norms_1->forward(ctx, y);
			y = ggml_gelu_inplace(ctx, y);

			y = convs11->forward(ctx, y);
			y = norms_2->forward(ctx, y);
			y = ggml_gelu_inplace(ctx, y);

			x = ggml_add_inplace(ctx, x, y);
		}
		return x;
	}
};

class WN : public GGMLBlock
{
protected:
	int n_layers;
	int hidden_channels;

public:
	WN(int hidden_channels,
	   int kernel_size,
	   int dilation_rate,
	   int n_layers, 
	   int gin_channels = 0)
		: n_layers(n_layers),
		hidden_channels(hidden_channels)
	{
		if (gin_channels)
		{
			std::string name = "cond_layer";
			blocks[name] = std::shared_ptr<GGMLBlock>(new Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1));
		}

		for (int i = 0; i < n_layers; i++)
		{
			int dilation = pow(dilation_rate, i);
			int	padding = ((kernel_size * dilation - dilation) / 2);

			std::string name = "in_layers." + std::to_string(i);
			blocks[name] = std::shared_ptr<GGMLBlock>(new Conv1d(hidden_channels, 2 * hidden_channels, kernel_size, 1, 
																 padding, dilation));
			int res_skip_channels;
			if (i < n_layers - 1)
				res_skip_channels = 2 * hidden_channels;
			else
				res_skip_channels = hidden_channels;

			name = "res_skip_layers." + std::to_string(i);
			blocks[name] = std::shared_ptr<GGMLBlock>(new Conv1d(hidden_channels, res_skip_channels, 1));
		}
	}

	struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* g)
	{
		ggml_tensor* output = NULL;

		if (g)
		{
			auto cond = std::dynamic_pointer_cast<Conv1d>(blocks["cond_layer"]);
			g = cond->forward(ctx, g);
		}

		for (int i = 0; i < n_layers; i++)
		{
			std::string name = "in_layers." + std::to_string(i);
			auto in_layer = std::dynamic_pointer_cast<Conv1d>(blocks[name]);
			ggml_tensor* x_in = in_layer->forward(ctx, x);

			ggml_tensor* g_l;
			if (g)
			{
				int cond_offset = i * 2 * hidden_channels;
				g_l = ggml_view_4d_ext(ctx, g,
									   0, g->ne[0],
									   cond_offset, 2 * hidden_channels,
									   0, g->ne[2],
									   0, g->ne[3]);
			}
			else
			{
				g_l = NULL;
			}

			ggml_tensor* in_act = g_l == NULL ? x_in : ggml_add_inplace(ctx, x_in, g_l);
			auto ina = ggml_nn_split_y_3(ctx, in_act, hidden_channels);
			in_act = ggml_mul_inplace(ctx, ggml_tanh_inplace(ctx, ina[0]), ggml_sigmoid_inplace(ctx, ina[1]));

			name = "res_skip_layers." + std::to_string(i);
			auto res_skip_act = std::dynamic_pointer_cast<Conv1d>(blocks[name]);
			in_act = res_skip_act->forward(ctx, in_act);

			if (i < n_layers - 1)
			{
				auto res_acts = ggml_nn_split_y_3(ctx, in_act, hidden_channels);
				x = ggml_add_inplace(ctx, x, res_acts[0]);
				if (output)
					output = ggml_add_inplace(ctx, output, res_acts[1]);
				else
					output = res_acts[1];
			}
			else
				if (output)
					output = ggml_add_inplace(ctx, output, in_act);
				else
					output = in_act;
		}
		return output;
	}
};

class ResBlock1 : public GGMLBlock
{
protected:
	int dilation[3] = { 1, 3, 5 };

public:
	ResBlock1(int channels, 
			  const int dilation[3],
			  int kernel_size = 3
			  )
	{
		this->dilation[0] = dilation[0];
		this->dilation[1] = dilation[1];
		this->dilation[2] = dilation[2];

		for (int i = 0; i < 3; i++)
		{
			std::string name = "convs1." + std::to_string(i);
			int padding = (kernel_size * dilation[i] - dilation[i]) / 2;
			blocks[name] = std::shared_ptr<GGMLBlock>(new Conv1d(channels, channels, kernel_size, 1, padding, dilation[i]));

			name = "convs2." + std::to_string(i);
			padding = (kernel_size - 1) / 2;
			blocks[name] = std::shared_ptr<GGMLBlock>(new Conv1d(channels, channels, kernel_size, 1, padding));
		}
	}

	struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x)
	{
		ggml_tensor* x_ori = x;
		for (int i = 0; i < 3; i++)
		{
			std::string name = "convs1." + std::to_string(i);
			auto conv_1 = std::dynamic_pointer_cast<Conv1d>(blocks[name]);

			name = "convs2." + std::to_string(i);
			auto conv_2 = std::dynamic_pointer_cast<Conv1d>(blocks[name]);

			ggml_tensor* xt = ggml_leaky_relu(ctx, x_ori, 0.1, false);
			xt = conv_1->forward(ctx, xt);
			xt = ggml_leaky_relu(ctx, xt, 0.1, true);
			xt = conv_2->forward(ctx, xt);
			x_ori = ggml_add_inplace(ctx, xt, x_ori);
		}
		return x_ori;
	}
};

class ResBlock2 : public GGMLBlock
{
protected:
	int  dilation[2] = { 1, 3 };

public:
	ResBlock2(int channels,
			  const int dilation[2],
			  int kernel_size = 3
	)
	{
		this->dilation[0] = dilation[0];
		this->dilation[1] = dilation[1];

		for (int i = 0; i < 2; i++)
		{
			std::string name = "convs." + std::to_string(i);
			int padding = (kernel_size * dilation[i] - dilation[i]) / 2;
			blocks[name] = std::shared_ptr<GGMLBlock>(new Conv1d(channels, channels, kernel_size, 1, padding, dilation[i]));
		}
	}

	struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x)
	{
		for (int i = 0; i < 2; i++)
		{
			std::string name = "convs." + std::to_string(i);
			auto conv_1 = std::dynamic_pointer_cast<Conv1d>(blocks[name]);

			ggml_tensor* xt = ggml_leaky_relu(ctx, x, 0.1, false);
			xt = conv_1->forward(ctx, xt);
			x = ggml_add_inplace(ctx, x, xt);
		}
		return x;
	}
};

class ElementwiseAffine : public GGMLBlock
{
protected:
	int channels;

public:
	ElementwiseAffine(int in_channels)
		: channels(in_channels)
	{
	}

	void init_params(struct ggml_context* ctx, std::map<std::string, enum ggml_type>& tensor_types, const std::string prefix = "")
	{
		params["m"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, channels);
		params["logs"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, channels);
	}

	struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, bool rev)
	{
		if (rev)
		{
			ggml_tensor* xm = ggml_sub(ctx, x, params["m"]);
			ggml_tensor* expn = ggml_exp_inplace(ctx, ggml_neg(ctx, params["logs"]));
			xm = ggml_mul_inplace(ctx, xm, expn);
			return xm;
		}
		else
		{
			ggml_tensor* expn = ggml_exp(ctx, params["logs"]);
			ggml_tensor* xm = ggml_mul_inplace(ctx, x, expn);
			xm = ggml_add_inplace(ctx, xm, params["m"]);
			return xm;
		}
	}
};

class ResidualCouplingLayer : public GGMLBlock
{
protected:
	int channels;
	bool mean_only;

public:
	ResidualCouplingLayer(int channels,
						  int hidden_channels,
						  int kernel_size,
						  int dilation_rate,
						  int n_layers,
						  int gin_channels = 0,
						  bool mean_only = false
	)
		: channels(channels),
		mean_only(mean_only)
	{
		blocks["pre"] = std::shared_ptr<GGMLBlock>(new Conv1d(channels / 2, hidden_channels, 1));
		blocks["enc"] = std::shared_ptr<GGMLBlock>(new WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels));
		blocks["post"] = std::shared_ptr<GGMLBlock>(new Conv1d(hidden_channels, channels / 2 * (2 - mean_only), 1));
	}

	struct std::vector<ggml_tensor*> forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* g, bool reverse)
	{
		auto pre = std::dynamic_pointer_cast<Conv1d>(blocks["pre"]);
		auto enc = std::dynamic_pointer_cast<WN>(blocks["enc"]);
		auto post = std::dynamic_pointer_cast<Conv1d>(blocks["post"]);

		auto x01 = ggml_nn_split_y_2(ctx, x, channels / 2);
		ggml_tensor* x0 = x01[0];
		ggml_tensor* x1 = x01[1];

		ggml_tensor* h = pre->forward(ctx, x0);
		h = enc->forward(ctx, h, g);
		ggml_tensor* stats = post->forward(ctx, h);

		ggml_tensor* m, *logs;
		if (!mean_only)
		{
			auto mlogs = x01;
			m = mlogs[0];
			logs = mlogs[1];
		}
		else
			m = stats;

		if (reverse)
		{
			ggml_tensor* a = ggml_sub_inplace(ctx, x1, m);
			if (!mean_only)
			{
				x1 = ggml_exp_inplace(ctx, ggml_neg_inplace(ctx, logs));
				x1 = ggml_mul_inplace(ctx, x1, a);
			}
			else
				x1 = a;
			//x = ggml_concat(ctx, x0, x1, 1);

			x = ggml_nop(ctx, x, { x0, x1 });

			return { x };
		}
		else
		{
			if (mean_only)
				x1 = ggml_mul_inplace(ctx, x1, ggml_exp(ctx, logs));
			x1 = ggml_add_inplace(ctx, x1, m);

			//x = ggml_concat(ctx, x0, x1, 1);
			x = ggml_nop(ctx, x, { x0, x1 });

			ggml_tensor* logdet = ggml_sum(ctx, logs);

			return { x, logdet };
		}

	}
};

class ConvFlow : public GGMLBlock
{
protected:
	int in_channels;
	int filter_channels;
	int num_bins;
	float tail_bound;

public:
	ConvFlow(int in_channels, 
			 int filter_channels,
			 int kernel_size,
			 int n_layers, 
			 int num_bins = 10,
			 float tail_bound = 5.f
	)
		: in_channels(in_channels),
		num_bins(num_bins),
		tail_bound(tail_bound),
		filter_channels(filter_channels)
	{
		blocks["pre"] = std::shared_ptr<GGMLBlock>(new Conv1d(in_channels / 2, filter_channels, 1));
		blocks["convs"] = std::shared_ptr<GGMLBlock>(new DDSConv(filter_channels, kernel_size, n_layers));
		blocks["proj"] = std::shared_ptr<GGMLBlock>(new Conv1d(filter_channels, in_channels / 2 * (num_bins * 3 - 1), 1));
	}

	struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* g)
	{
		auto x01 = ggml_nn_split_y_3(ctx, x, in_channels / 2);
		ggml_tensor* x0 = ggml_cont(ctx, x01[0]);
		ggml_tensor* x1 = ggml_cont(ctx, x01[1]);

		auto pre = std::dynamic_pointer_cast<Conv1d>(blocks["pre"]);
		auto convs = std::dynamic_pointer_cast<DDSConv>(blocks["convs"]);
		auto proj = std::dynamic_pointer_cast<Conv1d>(blocks["proj"]);

		ggml_tensor* h = pre->forward(ctx, x0);

		h = convs->forward(ctx, h, g);
		h = proj->forward(ctx, h);

		// [b, cx?, t] -> [b, c, t, ?]
		h = ggml_reshape_3d(ctx, h, x0->ne[0], h->ne[1], h->ne[2]);
		h = ggml_cont(ctx, ggml_transpose(ctx, h));

		ggml_tensor* unnormalized_widths = ggml_view_4d_ext(ctx, h,
															0, num_bins,
															0, h->ne[1],
															0, h->ne[2],
															0, h->ne[3]);
		ggml_tensor* unnormalized_heights = ggml_view_4d_ext(ctx, h,
															 num_bins, num_bins,
															 0, h->ne[1],
															 0, h->ne[2],
															 0, h->ne[3]);
		ggml_tensor* unnormalized_derivatives = ggml_view_4d_ext(ctx, h,
																 2 * num_bins, h->ne[0] - 2 * num_bins,
																 0, h->ne[1],
																 0, h->ne[2],
																 0, h->ne[3]);
		unnormalized_derivatives = ggml_cont(ctx, unnormalized_derivatives);
		unnormalized_widths = ggml_cont(ctx, unnormalized_widths);
		unnormalized_heights = ggml_cont(ctx, unnormalized_heights);

		unnormalized_widths = ggml_scale_inplace(ctx, unnormalized_widths, 1.f / sqrtf(filter_channels));
		unnormalized_heights = ggml_scale_inplace(ctx, unnormalized_heights, 1.f / sqrtf(filter_channels));

		/*
		x1, logabsdet = piecewise_rational_quadratic_transform(x1,
															   unnormalized_widths,
															   unnormalized_heights,
															   unnormalized_derivatives,
															   inverse = reverse,
															   tails = 'linear',
															   tail_bound = self.tail_bound
		)*/

		x1 = unconstrained_rational_quadratic_spline(ctx, x1,
													 unnormalized_widths,
													 unnormalized_heights,
													 unnormalized_derivatives,
													 true,
													 tail_bound);

		x = ggml_concat(ctx, x0, x1, 1);

		return x;
	}
};