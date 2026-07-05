#pragma once
#include "ggml_extend.hpp"
#include <algorithm>

class MultiHeadAttention : public GGMLBlock
{
protected:

	int32_t channels;
	int32_t out_channels;
	int32_t n_heads;
	int32_t window_size;
	bool proximal_bias;
	bool heads_share;

public:
	MultiHeadAttention(uint32_t channels,
					   uint32_t out_channels,
					   uint32_t n_heads,
					   uint32_t window_size = 0,
					   bool proximal_bias = false,
					   bool heads_share = true)
		: channels(channels),
		out_channels(out_channels),
		n_heads(n_heads),
		window_size(window_size),
		proximal_bias(proximal_bias),
		heads_share(heads_share)
	{
		blocks["conv_k"] = std::shared_ptr<GGMLBlock>(new Conv1d(channels, channels, 1));
		blocks["conv_q"] = std::shared_ptr<GGMLBlock>(new Conv1d(channels, channels, 1, 1, 0, 1, true, true));
		blocks["conv_v"] = std::shared_ptr<GGMLBlock>(new Conv1d(channels, channels, 1, 1, 0, 1, true, true));
		blocks["conv_o"] = std::shared_ptr<GGMLBlock>(new Conv1d(channels, out_channels, 1));
	}

	ggml_tensor* _matmul_with_relative_values(ggml_context* ctx, ggml_tensor* x, ggml_tensor* y)
	{
		y = ggml_cast(ctx, ggml_transpose(ctx, y), GGML_TYPE_F16);

		return ggml_mul_mat(ctx, y, x);
	}

	ggml_tensor* _matmul_with_relative_keys(ggml_context* ctx, ggml_tensor* x, ggml_tensor* y)
	{
		if (y->type != GGML_TYPE_F16)
			y = ggml_cast(ctx, y, GGML_TYPE_F16);

		return ggml_mul_mat(ctx, y, x);
	}

	ggml_tensor* _get_relative_embeddings(ggml_context* ctx, ggml_tensor* relative_embeddings, int length)
	{
		int max_relative_position = 2 * window_size + 1;
		// Pad first before slice to avoid using cond ops.
		int pad_length = std::max(length - (window_size + 1), 0);
		int slice_start_position = std::max((window_size + 1) - length, 0);
		int slice_end_position = slice_start_position + 2 * length - 1;
		ggml_tensor* padded_relative_embeddings;

		if (pad_length)
			padded_relative_embeddings = ggml_pad_4d_ext(ctx, relative_embeddings, 0, 0, pad_length, pad_length, 0, 0, 0, 0);
		else
			padded_relative_embeddings = relative_embeddings;
		padded_relative_embeddings = ggml_view_4d_ext(ctx, padded_relative_embeddings,
													  0, padded_relative_embeddings->ne[0],
													  slice_start_position, slice_end_position,
													  0, padded_relative_embeddings->ne[2], 0, 1);
		return padded_relative_embeddings;
	}

	ggml_tensor* _relative_position_to_absolute_position(ggml_context* ctx, ggml_tensor* x)
	{
		int64_t batch = x->ne[3];
		int64_t heads = x->ne[2];
		int64_t length = x->ne[1];

		x = ggml_pad(ctx, x, 1, 0, 0, 0);
		ggml_tensor* x_flat = ggml_reshape_3d(ctx, x, length * 2 * length, heads, batch);
		x_flat = ggml_pad(ctx, x_flat, length - 1, 0, 0, 0);

		ggml_tensor* x_final = ggml_reshape_4d(ctx, x_flat, 2 * length - 1, length + 1, heads, batch);
		return ggml_view_4d_ext(ctx, x_final, length - 1, x_final->ne[0] - (length - 1),
								0, length, 0, x_final->ne[2], 0, x_final->ne[3]);
	}

	ggml_tensor* _absolute_position_to_relative_position(ggml_context* ctx, ggml_tensor* x)
	{
		int64_t batch = x->ne[3];
		int64_t heads = x->ne[2];
		int64_t length = x->ne[1];

		x = ggml_pad(ctx, x, length - 1, 0, 0, 0);
		ggml_tensor* x_flat = ggml_reshape_3d(ctx, x, length * length + length * (length - 1), heads, batch);
		x_flat = ggml_pad_4d_ext(ctx, x_flat, length, 0, 0, 0, 0, 0, 0, 0);

		ggml_tensor* x_final = ggml_reshape_4d(ctx, x_flat, 2 * length, length, heads, batch);
		return ggml_view_4d_ext(ctx, x_final, 1, x_final->ne[0] - 1,
								0, x_final->ne[1], 0, x_final->ne[2], 0, x_final->ne[3]);
	}

	void init_params(struct ggml_context* ctx, std::map<std::string, enum ggml_type>& tensor_types, const std::string prefix = "")
	{
		int n_heads_rel = heads_share ? 1 : n_heads;
		params["emb_rel_k"] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, channels / n_heads, window_size * 2 + 1, n_heads_rel);
		params["emb_rel_v"] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, channels / n_heads, window_size * 2 + 1, n_heads_rel);
	}

	std::vector<ggml_tensor*> attention(ggml_context* ctx,
										ggml_tensor* q,
										ggml_tensor* k,
										ggml_tensor* v)
	{
		//reshape[b, d, t] ->[b, n_h, t, d_k]
		//b, d, t_s, t_t = (*key.size(), query.size(2))

		int64_t b = k->ne[2];

		int64_t d = k->ne[1];
		int64_t t_s = k->ne[0];

		int64_t t_t = q->ne[0];

		q = ggml_scale_inplace(ctx, q, 1.f / sqrtf(channels / n_heads));

		q = ggml_transpose(ctx, ggml_reshape_4d(ctx, q, t_t, channels / n_heads, n_heads, b));
		k = ggml_transpose(ctx, ggml_reshape_4d(ctx, k, t_s, channels / n_heads, n_heads, b));
		v = (ggml_reshape_4d(ctx, v, t_s, channels / n_heads, n_heads, b));

		q = ggml_cast(ctx, q, GGML_TYPE_F16);
		k = ggml_cast(ctx, k, GGML_TYPE_F16);

		ggml_tensor* scores = ggml_mul_mat(ctx, k, q);

		if (window_size != 0)
		{
			ggml_tensor* key_relative_embeddings = _get_relative_embeddings(ctx, params["emb_rel_k"], t_s);
			ggml_tensor* rel_logits = _matmul_with_relative_keys(ctx, q, key_relative_embeddings);
			ggml_tensor* scores_local = _relative_position_to_absolute_position(ctx, rel_logits);
			scores = ggml_add_inplace(ctx, scores, scores_local);
		}

		ggml_tensor* p_attn = ggml_soft_max_inplace(ctx, scores);
		ggml_tensor* output = ggml_mul_mat(ctx, v, p_attn);

		if (window_size != 0)
		{
			ggml_tensor* relative_weights = _absolute_position_to_relative_position(ctx, p_attn);
			ggml_tensor* value_relative_embeddings = _get_relative_embeddings(ctx, params["emb_rel_v"], t_s);
			output = ggml_add_inplace(ctx, output, _matmul_with_relative_values(ctx, relative_weights, value_relative_embeddings));
		}

		output = ggml_cont_3d(ctx, ggml_transpose(ctx, output), t_t, d, b);

		return { p_attn, output };
	}

	struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* c)
	{
		auto conv_k = std::dynamic_pointer_cast<Conv1d>(blocks["conv_k"]);
		auto conv_q = std::dynamic_pointer_cast<Conv1d>(blocks["conv_q"]);
		auto conv_v = std::dynamic_pointer_cast<Conv1d>(blocks["conv_v"]);
		auto conv_o = std::dynamic_pointer_cast<Conv1d>(blocks["conv_o"]);

		if (x == c)
		{
			x = c = conv_k->im2col(ctx, x);
			conv_k->x_im2col = true;
		}
		else
		{
			c = conv_v->im2col(ctx, c);
			conv_k->x_im2col = false;
		}

		ggml_tensor* k_t = conv_k->forward(ctx, x);
		ggml_tensor* q_t = conv_q->forward(ctx, c);
		ggml_tensor* v_t = conv_v->forward(ctx, c);

		auto attn = attention(ctx, q_t, k_t, v_t);
		x = attn[1];

		return conv_o->forward(ctx, x);
	}
};

class FFN : public GGMLBlock
{
protected:
	int in_channels, out_channels, filter_channels, kernel_size;
	bool relu;

public:
	FFN(int in_channels,
		int out_channels,
		int filter_channels,
		int kernel_size,
		bool relu)
		: in_channels(in_channels),
		out_channels(out_channels),
		filter_channels(filter_channels),
		kernel_size(kernel_size),
		relu(relu)
	{
		int pad = 0;
		if (kernel_size & 1)
			pad = kernel_size / 2;

		blocks["conv_1"] = std::shared_ptr<GGMLBlock>(new Conv1d(in_channels, filter_channels, kernel_size, 1, pad));
		blocks["conv_2"] = std::shared_ptr<GGMLBlock>(new Conv1d(filter_channels, out_channels, kernel_size, 1, pad));
	}

	struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x)
	{
		auto conv_1 = std::dynamic_pointer_cast<Conv1d>(blocks["conv_1"]);
		auto conv_2 = std::dynamic_pointer_cast<Conv1d>(blocks["conv_2"]);

		if ((kernel_size & 1) == 0)
			x = ggml_pad_2d_ext(ctx, x, (kernel_size - 1) / 2, kernel_size / 2, 0, 0);

		x = conv_1->forward(ctx, x);

		if (relu)
			x = ggml_relu_inplace(ctx, x);
		else
			x = ggml_gelu_quick_inplace(ctx, x);

		if ((kernel_size & 1) == 0)
			x = ggml_pad_2d_ext(ctx, x, (kernel_size - 1) / 2, kernel_size / 2, 0, 0);
		x = conv_2->forward(ctx, x);

		return x;
	}
};

class Encoder : public GGMLBlock
{
protected:
	int hidden_channels, filter_channels, n_heads, n_layers, kernel_size, window_size;

public:
	Encoder(int hidden_channels,
			int filter_channels,
			int n_heads,
			int n_layers,
			int kernel_size = 1,
			int window_size = 4)
		: hidden_channels(hidden_channels),
		filter_channels(filter_channels),
		n_heads(n_heads),
		n_layers(n_layers),
		kernel_size(kernel_size),
		window_size(window_size)
	{
		for (int i = 0; i < n_layers; i++)
		{
			std::string name = "attn_layers." + std::to_string(i);
			blocks[name] = std::shared_ptr<GGMLBlock>(new MultiHeadAttention(hidden_channels, hidden_channels, n_heads, window_size));

			name = "norm_layers_1." + std::to_string(i);
			blocks[name] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_channels));

			name = "ffn_layers." + std::to_string(i);
			blocks[name] = std::shared_ptr<GGMLBlock>(new FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, true));

			name = "norm_layers_2." + std::to_string(i);
			blocks[name] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_channels));
		}
	}

	struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x)
	{
		for (int i = 0; i < n_layers; i++)
		{
			std::string name = "attn_layers." + std::to_string(i);
			auto attn_layer = std::dynamic_pointer_cast<MultiHeadAttention>(blocks[name]);

			name = "norm_layers_1." + std::to_string(i);
			auto norm_1 = std::dynamic_pointer_cast<LayerNorm>(blocks[name]);

			name = "ffn_layers." + std::to_string(i);
			auto ffn = std::dynamic_pointer_cast<FFN>(blocks[name]);

			name = "norm_layers_2." + std::to_string(i);
			auto norm_2 = std::dynamic_pointer_cast<LayerNorm>(blocks[name]);

			ggml_tensor* y = attn_layer->forward(ctx, x, x);
			x = ggml_add_inplace(ctx, y, x);
			x = norm_1->forward(ctx, x);

			y = ffn->forward(ctx, x);
			x = ggml_add_inplace(ctx, x, y);
			x = norm_2->forward(ctx, x);
		}
		return x;
	}
};

// don't use this!!!
class Decoder : public GGMLBlock
{
protected:
	int hidden_channels, filter_channels, n_heads, n_layers, kernel_size, window_size;

public:
	Decoder(int hidden_channels,
			int filter_channels,
			int n_heads,
			int n_layers,
			int kernel_size = 1,
			int window_size = 4)
		: hidden_channels(hidden_channels),
		filter_channels(filter_channels),
		n_heads(n_heads),
		n_layers(n_layers),
		kernel_size(kernel_size),
		window_size(window_size)
	{
		for (int i = 0; i < n_layers; i++)
		{
			std::string name = "self_attn_layers." + std::to_string(i);
			blocks[name] = std::shared_ptr<GGMLBlock>(new MultiHeadAttention(hidden_channels, hidden_channels, n_heads, 0, window_size));

			name = "norm_layers_0." + std::to_string(i);
			blocks[name] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_channels));

			name = "encdec_attn_layers." + std::to_string(i);
			blocks[name] = std::shared_ptr<GGMLBlock>(new MultiHeadAttention(hidden_channels, hidden_channels, n_heads, 0, window_size));

			name = "norm_layers_1." + std::to_string(i);
			blocks[name] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_channels));

			name = "ffn_layers." + std::to_string(i);
			blocks[name] = std::shared_ptr<GGMLBlock>(new FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, true));

			name = "norm_layers_2." + std::to_string(i);
			blocks[name] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_channels));
		}
	}

	struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x)
	{
		for (int i = 0; i < n_layers; i++)
		{
			std::string name = "self_attn_layers." + std::to_string(i);
			auto attn_layer = std::dynamic_pointer_cast<MultiHeadAttention>(blocks[name]);

			name = "norm_layers_0." + std::to_string(i);
			auto norm_0 = std::dynamic_pointer_cast<LayerNorm>(blocks[name]);

			name = "encdec_attn_layers." + std::to_string(i);
			auto ed_attn_layer = std::dynamic_pointer_cast<MultiHeadAttention>(blocks[name]);

			name = "norm_layers_1." + std::to_string(i);
			auto norm_1 = std::dynamic_pointer_cast<LayerNorm>(blocks[name]);

			name = "ffn_layers." + std::to_string(i);
			auto ffn = std::dynamic_pointer_cast<FFN>(blocks[name]);

			name = "norm_layers_2." + std::to_string(i);
			auto norm_2 = std::dynamic_pointer_cast<LayerNorm>(blocks[name]);

			ggml_tensor* y = attn_layer->forward(ctx, x, x);
			x = ggml_add_inplace(ctx, x, y);
			x = norm_0->forward(ctx, x);

			y = ed_attn_layer->forward(ctx, x, x);
			x = ggml_add_inplace(ctx, x, y);
			x = norm_1->forward(ctx, x);

			y = ffn->forward(ctx, x);
			x = ggml_add_inplace(ctx, x, y);
			x = norm_2->forward(ctx, x);
		}
		return x;
	}
};