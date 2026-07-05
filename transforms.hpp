#include "ggml_extend.hpp"

//in-place!
__STATIC_INLINE__ ggml_tensor* rational_quadratic_spline(ggml_context* ctx,
														 ggml_tensor* one,
														 ggml_tensor* eps,
														 ggml_tensor* inputs,
														 ggml_tensor* unnormalized_widths,
														 ggml_tensor* unnormalized_heights,
														 ggml_tensor* unnormalized_derivatives,
														 bool inverse = false,
														 float low = 0.,
														 float hi = 1.)
{
	int num_bins = unnormalized_widths->ne[0];

	ggml_tensor* bin_size = ggml_scale(ctx, one, 1e-3f);

	ggml_tensor* low_t = ggml_scale(ctx, one, low);

	ggml_tensor* hi_t = ggml_scale(ctx, one, (hi + 1e-5f - low) / (hi - low));

	ggml_tensor* widths = ggml_soft_max_inplace(ctx, unnormalized_widths);
	widths = ggml_scale_inplace(ctx, widths, 1 - 1.e-3f * num_bins);
	widths = ggml_add1_inplace(ctx, widths, bin_size);

	ggml_tensor* cumwidths = ggml_cumsum_rows(ctx, widths);
	cumwidths = ggml_pad_2d_ext(ctx, cumwidths, 1, 0, 0, 0, hi_t);
	cumwidths = ggml_add1_inplace(ctx, ggml_scale_inplace(ctx, cumwidths, (hi - low)), low_t);
	widths = ggml_sub(ctx,
					  ggml_view_4d_ext(ctx, cumwidths,
									   1, cumwidths->ne[0] - 1,
									   0, cumwidths->ne[1],
									   0, cumwidths->ne[2],
									   0, cumwidths->ne[3]),
					  ggml_view_4d_ext(ctx, cumwidths,
									   0, cumwidths->ne[0] - 1,
									   0, cumwidths->ne[1],
									   0, cumwidths->ne[2],
									   0, cumwidths->ne[3])
	);

	//soft-plus
	ggml_tensor* derivatives = ggml_log_inplace(ctx,
												ggml_add1_inplace(ctx,
																  ggml_exp_inplace(ctx, unnormalized_derivatives), one));
	derivatives = ggml_add1_inplace(ctx, derivatives, bin_size);

	ggml_tensor* heights = ggml_soft_max_inplace(ctx, unnormalized_heights);
	heights = ggml_scale_inplace(ctx, heights, 1 - 1.e-3f * num_bins);
	heights = ggml_add1_inplace(ctx, heights, bin_size);

	ggml_tensor* cumheights = ggml_cumsum_rows(ctx, heights);
	cumheights = ggml_pad_2d_ext(ctx, cumheights, 1, 0, 0, 0, hi_t);
	cumheights = ggml_add1_inplace(ctx, ggml_scale_inplace(ctx, cumheights, (hi - low)), low_t);
	heights = ggml_sub(ctx,
					   ggml_view_4d_ext(ctx, cumheights,
										1, cumheights->ne[0] - 1,
										0, cumheights->ne[1],
										0, cumheights->ne[2],
										0, cumheights->ne[3]),
					   ggml_view_4d_ext(ctx, cumheights,
										0, cumheights->ne[0] - 1,
										0, cumheights->ne[1],
										0, cumheights->ne[2],
										0, cumheights->ne[3])
	);

	ggml_tensor* bin_locations = 1 ? cumheights : cumwidths;

	inputs = ggml_reshape_2d(ctx, inputs, 1, inputs->ne[0]);

	ggml_tensor* bin_idx = ggml_add1_inplace(ctx, ggml_neg_inplace(ctx, ggml_sub(ctx, bin_locations, inputs)), eps/*as EPS*/);
	bin_idx = ggml_step_inplace(ctx, bin_idx);
	bin_idx = ggml_mul_inplace(ctx, bin_idx, ggml_arange(ctx, 0, bin_idx->ne[0], 1));
	bin_idx = ggml_argmax(ctx, bin_idx);

	auto input_cumwidths = ggml_colsel_per_row_2d(ctx, cumwidths, bin_idx);
	auto input_bin_widths = ggml_colsel_per_row_2d(ctx, widths, bin_idx);
	auto input_cumheights = ggml_colsel_per_row_2d(ctx, cumheights, bin_idx);

	auto delta = ggml_div(ctx, heights, widths);
	auto input_delta = ggml_colsel_per_row_2d(ctx, delta, bin_idx);

	auto input_derivatives = ggml_colsel_per_row_2d(ctx, derivatives, bin_idx);
	auto input_derivatives_plus_one = ggml_colsel_per_row_2d(ctx, ggml_cont(ctx, ggml_view_4d_ext(ctx, derivatives,
																								  1, derivatives->ne[0] - 1,
																								  0, derivatives->ne[1],
																								  0, derivatives->ne[2],
																								  0, derivatives->ne[3])), bin_idx);

	auto input_heights = ggml_colsel_per_row_2d(ctx, heights, bin_idx);
	auto intermediate1 = ggml_sub_inplace(ctx, ggml_add_inplace(ctx, input_derivatives_plus_one, input_derivatives), ggml_scale(ctx, input_delta, 2));

	ggml_tensor* outputs;

	if (1)
	{
		inputs = ggml_flat(ctx, inputs);

		auto intermediate2 = ggml_sub(ctx, inputs, input_cumheights);
		auto intermediate3 = ggml_mul_inplace(ctx, intermediate1, intermediate2);

		auto a = ggml_add_inplace(ctx, ggml_mul_inplace(ctx, ggml_sub(ctx, input_delta, input_derivatives), input_heights), intermediate3);
		auto b = ggml_sub_inplace(ctx, ggml_mul(ctx, input_heights, input_derivatives), intermediate3);
		auto c = ggml_mul_inplace(ctx, ggml_neg(ctx, input_delta), intermediate2);

		auto b_pow = ggml_sqr(ctx, b);
		auto a_4 = ggml_scale_inplace(ctx, a, 4);
		auto discriminant = (ggml_sub_inplace(ctx, b_pow, ggml_mul_inplace(ctx, a_4, c)));
		auto root = ggml_div_inplace(ctx,
									 ggml_scale(ctx, c, 2),
									 ggml_sub_inplace(ctx, ggml_neg(ctx, b), ggml_sqrt_inplace(ctx, discriminant))
		);

		outputs = ggml_add_inplace(ctx, ggml_mul_inplace(ctx, root, input_bin_widths), input_cumwidths);
		outputs = ggml_reshape_3d(ctx, outputs, outputs->ne[0], 1, 1);
	}

	return outputs;
}

//in-place!
__STATIC_INLINE__ ggml_tensor* unconstrained_rational_quadratic_spline(ggml_context* ctx,
																	   ggml_tensor* inputs,
																	   ggml_tensor* unnormalized_widths,
																	   ggml_tensor* unnormalized_heights,
																	   ggml_tensor* unnormalized_derivatives,
																	   bool inverse = false,
																	   float tail_bound = 1.)
{
	//inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
	//outside_interval_mask = ~inside_interval_mask

	ggml_tensor* one = ggml_arange(ctx, 0, 1, 1); // 0
	one = ggml_exp_inplace(ctx, one); // 1

	ggml_tensor* eps = ggml_scale(ctx, one, 1.e-7F);

	ggml_tensor* in_cpy = ggml_dup(ctx, inputs);
	ggml_tensor* inside_interval_cnd = ggml_clamp(ctx, in_cpy, -tail_bound + 1.e-7f, tail_bound - 1.e-7f);
	inside_interval_cnd = ggml_sub_inplace(ctx, inside_interval_cnd, inputs);
	inside_interval_cnd = ggml_abs_inplace(ctx, inside_interval_cnd);
	inside_interval_cnd = ggml_step_inplace(ctx, inside_interval_cnd);

	ggml_tensor* outside_interval_mask = inside_interval_cnd;
	ggml_tensor* inside_interval_mask = ggml_add1_inplace(ctx, ggml_neg(ctx, inside_interval_cnd), one);

	ggml_tensor* outputs;

	//linear tails
	/*
	* unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
	* constant = np.log(np.exp(1 - min_derivative) - 1)
	* unnormalized_derivatives[..., 0] = constant
	* unnormalized_derivatives[..., -1] = constant*/

	float constant = logf(expf(1.f - 0.001f) - 1);

	ggml_tensor* const_t = ggml_scale(ctx, one, constant);
	unnormalized_derivatives = ggml_pad_2d_ext(ctx, unnormalized_derivatives, 1, 1, 0, 0, const_t);

	//outputs[outside_interval_mask] = inputs[outside_interval_mask]
	outputs = ggml_mul(ctx, inputs, outside_interval_mask);

	//unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
	//unnormalized_heights = unnormalized_heights[inside_interval_mask, :],
	//unnormalized_derivatives = unnormalized_derivatives[inside_interval_mask, :],

	ggml_tensor* inside_interval_mask_inf = ggml_mask_zr2n(ctx, inside_interval_mask, -1e4f, one);
	inside_interval_mask_inf = ggml_reshape_2d(ctx, inside_interval_mask_inf, 1, inside_interval_mask_inf->ne[0]);

	unnormalized_derivatives = ggml_mul_inplace(ctx, unnormalized_derivatives, inside_interval_mask_inf);
	unnormalized_heights = ggml_mul_inplace(ctx, unnormalized_heights, inside_interval_mask_inf);
	unnormalized_widths = ggml_mul_inplace(ctx, unnormalized_widths, inside_interval_mask_inf);

	inputs = ggml_mul(ctx, inputs, inside_interval_mask);

	ggml_tensor* out = rational_quadratic_spline(ctx, one, eps, inputs,
												 unnormalized_widths,
												 unnormalized_heights,
												 unnormalized_derivatives,
												 true,
												 -tail_bound, tail_bound);

	out = ggml_mul_inplace(ctx, out, inside_interval_mask);

	outputs = ggml_add_inplace(ctx, outputs, out);

	return outputs;
}