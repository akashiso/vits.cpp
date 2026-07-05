#ifndef __GGML_EXTEND_HPP__
#define __GGML_EXTEND_HPP__

#include <assert.h>
#include <inttypes.h>
#include <stdarg.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include "model.h"

#include "util.h"

#define EPS 1e-05f

#ifndef __STATIC_INLINE__
#define __STATIC_INLINE__ static inline
#endif


// set tensor[i, j, k, l]
// set tensor[l]
// set tensor[k, l]
// set tensor[j, k, l]
__STATIC_INLINE__ void ggml_tensor_set_f32(struct ggml_tensor* tensor, float value, int l, int k = 0, int j = 0, int i = 0)
{
	GGML_ASSERT(tensor->nb[0] == sizeof(float));
	*(float*)((char*)(tensor->data) + i * tensor->nb[3] + j * tensor->nb[2] + k * tensor->nb[1] + l * tensor->nb[0]) = value;
}

__STATIC_INLINE__ float ggml_tensor_get_f32(const ggml_tensor* tensor, int l, int k = 0, int j = 0, int i = 0)
{
	if (tensor->buffer != NULL)
	{
		float value;
		ggml_backend_tensor_get(tensor, &value, i * tensor->nb[3] + j * tensor->nb[2] + k * tensor->nb[1] + l * tensor->nb[0], sizeof(float));
		return value;
	}
	GGML_ASSERT(tensor->nb[0] == sizeof(float));
	return *(float*)((char*)(tensor->data) + i * tensor->nb[3] + j * tensor->nb[2] + k * tensor->nb[1] + l * tensor->nb[0]);
}

__STATIC_INLINE__ int ggml_tensor_get_i32(const ggml_tensor* tensor, int l, int k = 0, int j = 0, int i = 0)
{
	if (tensor->buffer != NULL)
	{
		float value;
		ggml_backend_tensor_get(tensor, &value, i * tensor->nb[3] + j * tensor->nb[2] + k * tensor->nb[1] + l * tensor->nb[0], sizeof(int));
		return value;
	}
	GGML_ASSERT(tensor->nb[0] == sizeof(int));
	return *(int*)((char*)(tensor->data) + i * tensor->nb[3] + j * tensor->nb[2] + k * tensor->nb[1] + l * tensor->nb[0]);
}

__STATIC_INLINE__ ggml_fp16_t ggml_tensor_get_f16(const ggml_tensor* tensor, int l, int k = 0, int j = 0, int i = 0)
{
	GGML_ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
	return *(ggml_fp16_t*)((char*)(tensor->data) + i * tensor->nb[3] + j * tensor->nb[2] + k * tensor->nb[1] + l * tensor->nb[0]);
}

__STATIC_INLINE__ struct ggml_tensor* ggml_group_norm_32(struct ggml_context* ctx,
														 struct ggml_tensor* a)
{
	const float eps = 1e-6f;  // default eps parameter
	return ggml_group_norm(ctx, a, 32, eps);
}










__STATIC_INLINE__ struct ggml_tensor* ggml_tensor_set_qck_randn(struct ggml_context* ctx, int64_t w, int64_t h, int64_t c1, int64_t n1, float domain = 2.f)
{
	if (domain == 0)
	{
		ggml_tensor* n = ggml_arange(ctx, 0, 1, 1);
		n = ggml_repeat_4d(ctx, n, w, h, c1, n1);

		return n;
	}

	std::default_random_engine generator;
	std::normal_distribution<float> distribution(0, 16.f);
	generator.seed(clock());

	float c = cosf(distribution(generator)) + 1.1f;
	float d = sinf(distribution(generator)) + 1.1f;

	float s0 = (distribution(generator));
	float s1 = (distribution(generator));

	int total = w * h * c1 * n1;
	total = (total & ~255) + ((total & 255) ? 256 : 0);

	c *= 64.f;
	d *= 64.f;

	ggml_tensor* x = ggml_arange(ctx, s0 / c, s0 / c + total, 1);
	ggml_tensor* y = ggml_arange(ctx, s1 / d, s1 / d + total, 1);
	x = ggml_scale_inplace(ctx, x, c); //prevent precise loss
	y = ggml_scale_inplace(ctx, x, d);

	ggml_tensor* z = ggml_arange(ctx, -domain, domain, (2.f * domain) / 256.f);
	z = ggml_repeat(ctx, z, x);

	x = ggml_div_inplace(ctx, ggml_cos_inplace(ctx, x), ggml_sin_inplace(ctx, y));
	x = ggml_reshape_2d(ctx, x, 256, total >> 8);

	ggml_tensor* idx = ggml_argsort(ctx, x, GGML_SORT_ORDER_ASC);
	z = ggml_reshape_3d(ctx, z, 1, 256, total >> 8);
	z = ggml_get_rows(ctx, z, idx);
	z = ggml_view_1d(ctx, z, w * h * c1 * n1, 0);
	z = ggml_reshape_4d(ctx, z, w, h, c1, n1);

	return z;
}

__STATIC_INLINE__ struct ggml_tensor* ggml_tensor_set_qck_randn(struct ggml_context* ctx, struct ggml_tensor* src, float domain = 2.f)
{
	return ggml_tensor_set_qck_randn(ctx, src->ne[0], src->ne[1], src->ne[2], src->ne[3], domain);
}


__STATIC_INLINE__ float ggml_tensor_sum(struct ggml_tensor* src)
{
	float mean = 0.0f;
	int64_t nelements = ggml_nelements(src);
	float* data = (float*)src->data;
	for (int i = 0; i < nelements; i++)
	{
		mean += data[i];
	}
	return mean;
}

__STATIC_INLINE__ ggml_tensor* ggml_tensor_ceil(struct ggml_tensor* src)
{
	int64_t nelements = ggml_nelements(src);
	float* data = (float*)src->data;
	for (int i = 0; i < nelements; i++)
	{
		data[i] = ceilf(data[i]);
	}
	return src;
}

__STATIC_INLINE__ ggml_tensor* ggml_tensor_cpy(struct ggml_context* ctx, struct ggml_tensor* src)
{
	ggml_tensor* dst = ggml_dup_tensor(ctx, src);
	memcpy(dst->data, src->data, ggml_nbytes(src));
	return dst;
}

__STATIC_INLINE__ ggml_tensor* ggml_tensor_cumsum(struct ggml_tensor* output)
{
	int64_t width = output->ne[0];
	int64_t height = output->ne[1];
	int64_t channels = output->ne[2];
	for (int k = 0; k < channels; k++)
	{
		for (int iy = 0; iy < height; iy++)
		{
			float sum = 0.f;
			for (int ix = 0; ix < width; ix++)
			{
				float value = ggml_tensor_get_f32(output, ix, iy, k);
				sum += value;
				ggml_tensor_set_f32(output, sum, ix, iy, k);
			}
		}
	}
	return output;
}

__STATIC_INLINE__ struct ggml_tensor* ggml_nn_linear(struct ggml_context* ctx,
													 struct ggml_tensor* x,
													 struct ggml_tensor* w,
													 struct ggml_tensor* b)
{
	x = ggml_mul_mat(ctx, w, x);
	if (b != NULL)
	{
		x = ggml_add_inplace(ctx, x, b);
	}
	return x;
}

// w: [OC，IC, KH, KW]
// x: [N, IC, IH, IW]
// b: [OC,]
// result: [N, OC, OH, OW]
__STATIC_INLINE__ struct ggml_tensor* ggml_nn_conv_2d(struct ggml_context* ctx,
													  struct ggml_tensor* x,
													  struct ggml_tensor* w,
													  struct ggml_tensor* b,
													  int s0 = 1,
													  int s1 = 1,
													  int p0 = 0,
													  int p1 = 0,
													  int d0 = 1,
													  int d1 = 1)
{
	x = ggml_conv_2d(ctx, w, x, s0, s1, p0, p1, d0, d1);
	if (b != NULL)
	{
		b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
		// b = ggml_repeat(ctx, b, x);
		x = ggml_add_inplace(ctx, x, b);
	}
	return x;
}

__STATIC_INLINE__ struct ggml_tensor* ggml_conv_1d_dw_impl(
	struct ggml_context* ctx,
	struct ggml_tensor* a,
	struct ggml_tensor* b,
	int                   s0,
	int                   p0,
	int                   d0)
{
	struct ggml_tensor* new_a = a; //[C, 1, KW]
	struct ggml_tensor* new_b = ggml_reshape_4d(ctx, b, b->ne[0], 1, b->ne[1], b->ne[2]); //[C, W] => [C, 1, W]

	struct ggml_tensor* im2col = ggml_im2col(ctx, new_a, new_b, s0, 0, p0, 0, d0, 0, false, GGML_TYPE_F16); //[C, OW, KW]

	struct ggml_tensor* result = ggml_mul_mat(ctx, im2col, a); //[C, 1, OW]

	result = ggml_reshape_3d(ctx, result, result->ne[0], result->ne[2], result->ne[3]); //[C, OW]

	return result;
}

__STATIC_INLINE__ struct ggml_tensor* ggml_nn_conv_1d(struct ggml_context* ctx,
													  struct ggml_tensor* x,
													  struct ggml_tensor* w,
													  struct ggml_tensor* b,
													  int s0 = 1,
													  int p0 = 0,
													  int d0 = 1)
{
	x = ggml_conv_1d(ctx, w, x, s0, p0, d0);
	if (b != NULL)
	{
		b = ggml_reshape_3d(ctx, b, 1, b->ne[0], 1);
		// b = ggml_repeat(ctx, b, x);
		x = ggml_add_inplace(ctx, x, b);
	}
	return x;
}

__STATIC_INLINE__ struct ggml_tensor* ggml_nn_conv_1d_im2col(struct ggml_context* ctx,
															 struct ggml_tensor* im2col,
															 struct ggml_tensor* a,
															 struct ggml_tensor* b,
															 int s0 = 1,
															 int p0 = 0,
															 int d0 = 1)
{
	struct ggml_tensor* result =
		ggml_mul_mat(ctx,
					 ggml_reshape_2d(ctx, im2col, im2col->ne[0], (im2col->ne[2] * im2col->ne[1])), // [N, OL, IC * K] => [N*OL, IC * K]
					 ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1]), a->ne[2]));                    // [OC，IC, K] => [OC, IC * K]

	result = ggml_reshape_3d(ctx, result, im2col->ne[1], a->ne[2], im2col->ne[2]); // [N, OC, OL]

	if (b != NULL)
	{
		b = ggml_reshape_3d(ctx, b, 1, b->ne[0], 1);
		// b = ggml_repeat(ctx, b, x);
		result = ggml_add_inplace(ctx, result, b);
	}
	return result;
}

__STATIC_INLINE__ struct ggml_tensor* ggml_nn_conv_1d_dw(struct ggml_context* ctx,
														 struct ggml_tensor* x,
														 struct ggml_tensor* w,
														 struct ggml_tensor* b,
														 int s0 = 1,
														 int p0 = 0,
														 int d0 = 1)
{
	x = ggml_conv_1d_dw_impl(ctx, w, x, s0, p0, d0);
	if (b != NULL)
	{
		b = ggml_reshape_3d(ctx, b, 1, b->ne[0], 1);
		// b = ggml_repeat(ctx, b, x);
		x = ggml_add_inplace(ctx, x, b);
	}
	return x;
}



// w: [OC，IC, KD, 1 * 1]
// x: [N, IC, IH, IW]
// b: [OC,]
// result: [N, OC, OH, OW]
__STATIC_INLINE__ struct ggml_tensor* ggml_nn_conv_3d_nx1x1_bak(struct ggml_context* ctx,
																struct ggml_tensor* x,
																struct ggml_tensor* w,
																struct ggml_tensor* b,
																int s2 = 1,
																int p2 = 1,
																int d2 = 1)
{
	GGML_ASSERT(w->ne[0] == 1);
	// timesteps = x.shape[0]
	// x = rearrange(x, "(b t) c h w -> b c t h w", t=timesteps)
	// x = conv3d(x)
	// return rearrange(x, "b c t h w -> (b t) c h w")
	int64_t T = x->ne[3];
	int64_t B = x->ne[3] / T;
	int64_t C = x->ne[2];
	int64_t H = x->ne[1];
	int64_t W = x->ne[0];

	x = ggml_reshape_4d(ctx, x, W * H, C, T, B);           // (b t) c h w -> b t c (h w)
	x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));  // b t c (h w) -> b c t (h w)
	x = ggml_conv_2d(ctx, w, x, 1, s2, 0, p2, 1, d2);      // [B, OC, T, OH * OW]
	if (b != NULL)
	{
		b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
		x = ggml_add_inplace(ctx, x, b);
	}
	x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));  // b c t (h w) -> b t c (h w)
	x = ggml_reshape_4d(ctx, x, W, H, C, T * B);           // b t c (h w) -> (b t) c h w
	return x;                                              // [B*T, OC, OH, OW]
}

// w: [OC，IC, KD, 1 * 1]
// x: [N, IC, ID, IH*IW]
// b: [OC,]
// result: [N, OC, OD, OH*OW]
__STATIC_INLINE__ struct ggml_tensor* ggml_nn_conv_3d_nx1x1(struct ggml_context* ctx,
															struct ggml_tensor* x,
															struct ggml_tensor* w,
															struct ggml_tensor* b,
															int s2 = 1,
															int p2 = 1,
															int d2 = 1)
{
	x = ggml_conv_2d(ctx, w, x, 1, s2, 0, p2, 1, d2);  // [N, OC, T, OH * OW]
	if (b != NULL)
	{
		b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
		x = ggml_add_inplace(ctx, x, b);
	}
	return x;  // [N, OC, T, OH * OW]
}

__STATIC_INLINE__ std::vector<struct ggml_tensor*> ggml_nn_split_x_2(struct ggml_context* ctx,
																	 struct ggml_tensor* x,
																	 int posx)
{
	auto x1 = ggml_view_2d(ctx, x, posx, x->ne[1], x->nb[1], 0);
	auto x2 = ggml_view_2d(ctx, x, x->ne[0] - posx, x->ne[1], x->nb[1], posx * x->nb[0]);
	return { x1, x2 };
}

__STATIC_INLINE__ std::vector<struct ggml_tensor*> ggml_nn_split_y_2(struct ggml_context* ctx,
																	 struct ggml_tensor* x,
																	 int64_t posy)
{
	auto x1 = ggml_view_2d(ctx, x, x->ne[0], posy, x->nb[1], 0);
	auto x2 = ggml_view_2d(ctx, x, x->ne[0], x->ne[1] - posy, x->nb[1], posy * x->nb[1]);
	return { x1, x2 };
}

__STATIC_INLINE__ std::vector<struct ggml_tensor*> ggml_nn_split_y_3(struct ggml_context* ctx,
																	 struct ggml_tensor* x,
																	 int64_t posy)
{
	auto x1 = ggml_view_3d(ctx, x, x->ne[0], posy, x->ne[2], x->nb[1], x->nb[2], 0);
	auto x2 = ggml_view_3d(ctx, x, x->ne[0], x->ne[1] - posy, x->ne[2], x->nb[1], x->nb[2], posy * x->nb[1]);
	return { x1, x2 };
}

__STATIC_INLINE__ struct ggml_tensor* ggml_view_4d_ext(struct ggml_context* ctx,
													   struct ggml_tensor* x,
													   int64_t px, int64_t sx,
													   int64_t py, int64_t sy,
													   int64_t pz, int64_t sz,
													   int64_t pw, int64_t sw)
{
	x = ggml_view_4d(ctx, x, sx, sy, sz, sw,
					 x->nb[1], x->nb[2], x->nb[3], px * x->nb[0] + py * x->nb[1] + pz * x->nb[2] + pw * x->nb[3]);
	return x;
}

__STATIC_INLINE__ struct ggml_tensor* ggml_view_2d_ext(struct ggml_context* ctx,
													   struct ggml_tensor* x,
													   int64_t px, int64_t sx,
													   int64_t py, int64_t sy)
{
	x = ggml_view_4d_ext(ctx, x, px, sx, py, sy, 0, x->ne[2], 0, x->ne[3]);
	return x;
}

//set all tensors in y as leaves of x, useful for in-place subpart calculations
//unavailable when computing backward
__STATIC_INLINE__ struct ggml_tensor* ggml_nop(struct ggml_context* ctx,
											   struct ggml_tensor* x,
											   std::vector<struct ggml_tensor*> y)
{
	x = ggml_view_4d_ext(ctx, x, 0, x->ne[0], 0, x->ne[1], 0, x->ne[2], 0, x->ne[3]);
	memcpy(x->src + 1, y.data(), y.size() * sizeof(ggml_tensor*));
	return x;
}

__STATIC_INLINE__ struct ggml_tensor* ggml_flat(struct ggml_context* ctx,
												struct ggml_tensor* x)
{
	x = ggml_reshape_1d(ctx, x, ggml_nelements(x));
	return x;
}

//doesn't expect X is contiguous
__STATIC_INLINE__ struct ggml_tensor* ggml_nn_flip(struct ggml_context* ctx,
												   struct ggml_tensor* x,
												   bool cont_out = true)
{
	ggml_tensor* idx = ggml_argsort(ctx, ggml_arange(ctx, 0, x->ne[1], 1), GGML_SORT_ORDER_DESC);

	ggml_tensor* ox = x; //[W, H, C, N]
	if (ox->ne[2] > 1)
	{
		x = ggml_permute(ctx, x, 2, 3, 0, 1); //[C, N, W, H]
		x = ggml_cont_2d(ctx, x, ox->ne[0] * ox->ne[2] * ox->ne[3], ox->ne[1]); //[N*C*W, H]
	}

	x = ggml_get_rows(ctx, x, idx);

	if (ox->ne[2] > 1)
	{
		x = ggml_reshape_4d(ctx, x, ox->ne[2], ox->ne[3], ox->ne[0], ox->ne[1]); //[C, N, W, H]
		x = ggml_permute(ctx, x, 2, 3, 0, 1); //[W, H, C, N]
		if (cont_out)
			x = ggml_cont(ctx, x);
	}
	return x;
}

__STATIC_INLINE__ struct ggml_tensor* ggml_gen_rev_idx(struct ggml_context* ctx, int l)
{
	ggml_tensor* idx = ggml_argsort(ctx, ggml_arange(ctx, 0, l, 1), GGML_SORT_ORDER_DESC);

	return idx;
}

//fix CUDA missing OP_SET, in-place
__STATIC_INLINE__ struct ggml_tensor* ggml_set_4d_ext(struct ggml_context* ctx,
													  struct ggml_tensor* dst,
													  struct ggml_tensor* src,
													  int64_t px, int64_t py, int64_t pz, int64_t pw)
{
	ggml_tensor* ctr = ggml_view_4d_ext(ctx, dst, px, src->ne[0], py, src->ne[1], pz, src->ne[2], pw, src->ne[3]);
	src = ggml_cpy(ctx, src, ctr);
	src = ggml_view_4d(ctx, src, dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
					   dst->nb[1], dst->nb[2], dst->nb[3], -(px * dst->nb[0] + py * dst->nb[1] + pz * dst->nb[2] + pw * dst->nb[3]));
	return src;
}

//fix CUDA missing OP_SET, in-place
__STATIC_INLINE__ struct ggml_tensor* ggml_set_2d_ext(struct ggml_context* ctx,
													  struct ggml_tensor* dst,
													  struct ggml_tensor* src,
													  int64_t px, int64_t py)
{
	src = ggml_set_4d_ext(ctx, dst, src, px, py, 0, 0);
	return src;
}

//fix CUDA PAD failed, pad on the left side is available
//will convert to F32
__STATIC_INLINE__ struct ggml_tensor* ggml_pad_4d_ext(struct ggml_context* ctx,
													  struct ggml_tensor* x,
													  int64_t xl, int64_t xr,
													  int64_t yt, int64_t yb,
													  int64_t zn, int64_t zf,
													  int64_t c1, int64_t c2,
													  struct ggml_tensor* val = 0)
{
	if (xl + xr == 0 && yt + yb == 0 && zn + zf == 0 && c1 + c2 == 0)
		return x;

	if (val == 0)
		val = ggml_arange(ctx, 0, 1, 1);

	ggml_tensor* n = ggml_repeat_4d(ctx, val, x->ne[0] + xl + xr,
									x->ne[1] + yt + yb,
									x->ne[2] + zn + zf,
									x->ne[3] + c1 + c2);

	ggml_tensor* ctr = ggml_view_4d_ext(ctx, n, xl, x->ne[0], yt, x->ne[1], zn, x->ne[2], c1, x->ne[3]);
	x = ggml_cpy(ctx, x, ctr);
	x = ggml_view_4d(ctx, x, n->ne[0], n->ne[1], n->ne[2], n->ne[3],
					 n->nb[1], n->nb[2], n->nb[3], -(xl * n->nb[0] + yt * n->nb[1] + zn * n->nb[2] + c1 * n->nb[3]));
	return x;
}

//fix CUDA PAD failed, left pad available
__STATIC_INLINE__ struct ggml_tensor* ggml_pad_2d_ext(struct ggml_context* ctx,
													  struct ggml_tensor* x,
													  int64_t xl, int64_t xr,
													  int64_t yt, int64_t yb,
													  struct ggml_tensor* val = 0)
{
	if (xl + xr == 0 && yt + yb == 0)
		return x;
	x = ggml_pad_4d_ext(ctx, x, xl, xr, yt, yb, 0, 0, 0, 0, val);
	return x;
}

__STATIC_INLINE__ struct ggml_tensor* ggml_gen_diag_mask(struct ggml_context* ctx, int l)
{
	ggml_tensor* diag = ggml_arange(ctx, 1, l + 1, 1);
	ggml_tensor* d = ggml_repeat_4d(ctx, ggml_arange(ctx, 0, l, 1), l, l, 1, 1);
	diag = ggml_reshape_2d(ctx, diag, 1, l);

	d = ggml_sub_inplace(ctx, d, diag);
	d = ggml_neg_inplace(ctx, d);
	d = ggml_step_inplace(ctx, d);

	return d;
}

__STATIC_INLINE__ struct ggml_tensor* ggml_cumsum_rows(struct ggml_context* ctx,
													   struct ggml_tensor* x)
{
	ggml_tensor* msk = ggml_gen_diag_mask(ctx, x->ne[0]);

	x = ggml_mul_mat(ctx, msk, x);
	return x;
}

__STATIC_INLINE__ struct ggml_tensor* ggml_colsel_per_row_2d(struct ggml_context* ctx,
															 struct ggml_tensor* x,
															 struct ggml_tensor* idx)
{
	x = ggml_reshape_3d(ctx, x, 1, x->ne[0], x->ne[1]);
	idx = ggml_reshape_2d(ctx, idx, 1, idx->ne[0]);
	x = ggml_get_rows(ctx, x, idx);
	x = ggml_reshape_1d(ctx, x, idx->ne[1]);
	return x;
}

__STATIC_INLINE__ struct ggml_tensor* ggml_mask_zr2n(struct ggml_context* ctx,
													 struct ggml_tensor* x,
													 float f,
													 struct ggml_tensor* one = NULL)
{
	if (one == NULL)
	{
		one = ggml_arange(ctx, 0, 1, 1); // 0
		one = ggml_exp_inplace(ctx, one); // 1
	}

	x = ggml_sub_inplace(ctx, x, one);
	x = ggml_scale_inplace(ctx, x, -f + 1);
	x = ggml_add1_inplace(ctx, x, one);

	return x;
}

__STATIC_INLINE__ struct ggml_tensor* ggml_gen_seq_mask(struct ggml_context* ctx,
														struct ggml_tensor* x,
														int l)
{
	ggml_tensor* d = ggml_arange(ctx, 0, l, 1);
	d = ggml_repeat_4d(ctx, d, l, x->ne[0], 1, 1);

	d = ggml_sub_inplace(ctx, d, ggml_reshape_2d(ctx, x, 1, x->ne[0]));
	d = ggml_neg_inplace(ctx, d);
	d = ggml_step_inplace(ctx, d);

	return d;
}

__STATIC_INLINE__ struct ggml_tensor* ggml_nn_conv_transpose_1d(struct ggml_context* ctx,
																struct ggml_tensor* x,
																struct ggml_tensor* w,
																struct ggml_tensor* b,
																int s0 = 1,
																int p0 = 0,
																int d0 = 1)
{
	// the GGML's implementation is much too slow when KW is too small
	// x = ggml_conv_transpose_1d(ctx, w, x, s0, p0, d0);

	// more effective way, but the memory cost...
	// up to 70x faster in RTX 2060 12GB (KW = 16, IW > 819200, S0 = 8)
	// DO NOT USE IT IF YOU ARE UNSURE ABOUT HOW MUCH MEMORY IT WILL TAKE
	
	if (s0 > 1)
	{
		ggml_tensor* xo = x;
		x = ggml_reshape_2d(ctx, x, 1, ggml_nelements(x));
		//x = ggml_pad(ctx, x, s0 - 1, 0, 0, 0);
		x = ggml_pad_2d_ext(ctx, x, 0, s0 - 1, 0, 0); //fix CUDA OP-PAD failed
		x = ggml_reshape_4d(ctx, x, xo->ne[0] * s0, xo->ne[1], xo->ne[2], xo->ne[3]);
		x = ggml_view_2d_ext(ctx, x, 0, x->ne[0] - s0 + 1,
							 0, x->ne[1]);
	}

	w = ggml_transpose(ctx, ggml_permute(ctx, w, 0, 2, 1, 3));
	w = ggml_nn_flip(ctx, w, false); //F16 => F32
	w = ggml_transpose(ctx, w);
	w = ggml_cast(ctx, w, GGML_TYPE_F16);

	x = ggml_conv_1d(ctx, w, x, 1, w->ne[0] - 1, d0);

	if (b != NULL)
	{
		b = ggml_reshape_4d(ctx, b, 1, b->ne[0], 1, 1);
		// b = ggml_repeat(ctx, b, x);
		x = ggml_add_inplace(ctx, x, b);
	}
	return x;
}

__STATIC_INLINE__ struct ggml_tensor* ggml_nn_conv_transpose_1d_quick(struct ggml_context* ctx,
																	  struct ggml_tensor* x,
																	  struct ggml_tensor* w,
																	  struct ggml_tensor* b,
																	  int s0 = 1,
																	  int p0 = 0,
																	  int d0 = 1/*???*/)
{
	// the GGML's implementation is much too slow when KW is too small
	// x = ggml_conv_transpose_1d(ctx, w, x, s0, p0, d0);

	// specialize for KW % s0 == 0
	// batching is not implemented
	// more effective (especially in memory cost, compared to ggml_nn_conv_transpose_1d)
	// up to 80x faster in RTX 2060 12GB (KW = 16, IW > 819200, S0 = 8)

	int KW1 = w->ne[0] / s0;
	int IC = w->ne[2];
	int OC = w->ne[1];

	w = ggml_reshape_4d(ctx, w, s0, KW1, w->ne[1], w->ne[2]); // [S0, KW1, OC, IC]
	w = ggml_nn_flip(ctx, w, false); //F16 => F32
	w = ggml_transpose(ctx, w);

	w = ggml_permute(ctx, w, 0, 1, 3, 2); // [KW1, S0, IC, OC]
	w = ggml_permute(ctx, w, 0, 2, 1, 3); // [KW1, IC, S0, OC]
	w = ggml_cast(ctx, w, GGML_TYPE_F16); // cont

	struct ggml_tensor* im2col = ggml_im2col(ctx, w, x, 1, 0, KW1 - 1, 0, d0, 0, false, GGML_TYPE_F16); // [KW1 * IC, OW]

	x = ggml_mul_mat(ctx,
					 im2col,
					 ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1], w->ne[2] * w->ne[3])); // [KW1 * IC, S0 * OC]

	x = ggml_reshape_3d(ctx, x, x->ne[0], s0, OC); // [OW, S0 * OC] => [OW, S0, OC]
	x = ggml_cont(ctx, ggml_transpose(ctx, x)); // [OW, S0, OC] => [S0, OW, OC]
	x = ggml_reshape_2d(ctx, x, x->ne[0] * x->ne[1], x->ne[2]); // [OW * S0, OC]

	if (b != NULL)
	{
		b = ggml_reshape_4d(ctx, b, 1, b->ne[0], 1, 1);
		// b = ggml_repeat(ctx, b, x);
		x = ggml_add_inplace(ctx, x, b);
	}
	return x;
}

__STATIC_INLINE__ struct ggml_tensor* ggml_nn_layer_norm(struct ggml_context* ctx,
														 struct ggml_tensor* x,
														 struct ggml_tensor* w,
														 struct ggml_tensor* b,
														 float eps = EPS)
{
	x = ggml_cont(ctx, ggml_transpose(ctx, x));
	x = ggml_norm_inplace(ctx, x, eps);
	if (w != NULL)
	{
		x = ggml_mul_inplace(ctx, x, w);
		if (b != NULL)
		{
			x = ggml_add_inplace(ctx, x, b);
		}
	}
	x = ggml_cont(ctx, ggml_transpose(ctx, x));
	return x;
}

__STATIC_INLINE__ struct ggml_tensor* ggml_nn_group_norm(struct ggml_context* ctx,
														 struct ggml_tensor* x,
														 struct ggml_tensor* w,
														 struct ggml_tensor* b,
														 int num_groups = 32)
{
	if (ggml_n_dims(x) >= 3 && w != NULL && b != NULL)
	{
		w = ggml_reshape_4d(ctx, w, 1, 1, w->ne[0], 1);
		b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
	}

	const float eps = 1e-6f;  // default eps parameter
	x = ggml_group_norm(ctx, x, num_groups, eps);
	if (w != NULL && b != NULL)
	{
		x = ggml_mul(ctx, x, w);
		// b = ggml_repeat(ctx, b, x);
		x = ggml_add_inplace(ctx, x, b);
	}
	return x;
}

__STATIC_INLINE__ void ggml_backend_tensor_get_and_sync(ggml_backend_t backend, const struct ggml_tensor* tensor, void* data, size_t offset, size_t size)
{

	if (!ggml_backend_is_cpu(backend))
	{
		ggml_backend_tensor_get_async(backend, tensor, data, offset, size);
		ggml_backend_synchronize(backend);
	}
	else
	{
		ggml_backend_tensor_get(tensor, data, offset, size);
	}
}


#define MAX_PARAMS_TENSOR_NUM 32768
#define MAX_GRAPH_SIZE 32768

__STATIC_INLINE__ size_t ggml_tensor_num(ggml_context* ctx)
{
	size_t num = 0;
	for (ggml_tensor* t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t))
	{
		num++;
	}
	return num;
}

struct GGMLRunner
{
protected:
	typedef std::function<struct ggml_cgraph* ()> get_graph_cb_t;

	struct ggml_context* params_ctx = NULL;
	ggml_backend_buffer_t params_buffer = NULL;

	struct ggml_context* compute_ctx = NULL;
	struct ggml_gallocr* compute_allocr = NULL;

	std::map<struct ggml_tensor*, const void*> backend_tensor_data_map;

	ggml_backend_t backend = NULL;

	void alloc_params_ctx()
	{
		struct ggml_init_params params;
		params.mem_size = static_cast<size_t>(MAX_PARAMS_TENSOR_NUM * ggml_tensor_overhead());
		params.mem_buffer = NULL;
		params.no_alloc = true;

		params_ctx = ggml_init(params);
		GGML_ASSERT(params_ctx != NULL);
	}

	void free_params_ctx()
	{
		if (params_ctx != NULL)
		{
			ggml_free(params_ctx);
			params_ctx = NULL;
		}
	}

	void alloc_compute_ctx()
	{
		struct ggml_init_params params;
		params.mem_size = static_cast<size_t>(ggml_tensor_overhead() * MAX_GRAPH_SIZE + ggml_graph_overhead());
		params.mem_buffer = NULL;
		params.no_alloc = true;

		compute_ctx = ggml_init(params);
		GGML_ASSERT(compute_ctx != NULL);
	}

	void free_compute_ctx()
	{
		if (compute_ctx != NULL)
		{
			ggml_free(compute_ctx);
			compute_ctx = NULL;
		}
	}

	bool alloc_compute_buffer(get_graph_cb_t get_graph)
	{
		if (compute_allocr != NULL)
		{
			return true;
		}
		reset_compute_ctx();
		struct ggml_cgraph* gf = get_graph();
		backend_tensor_data_map.clear();
		compute_allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

		if (!ggml_gallocr_reserve(compute_allocr, gf))
		{
			// failed to allocate the compute buffer
			LOG_ERROR("%s: failed to allocate the compute buffer\n", get_desc().c_str());
			free_compute_buffer();
			return false;
		}

		// compute the required memory
		size_t compute_buffer_size = ggml_gallocr_get_buffer_size(compute_allocr, 0);
		LOG_INFO("%s compute buffer size: %.2f MB(%s)",
				  get_desc().c_str(),
				  compute_buffer_size / 1024.0 / 1024.0,
				  ggml_backend_is_cpu(backend) ? "RAM" : "VRAM");
		return true;
	}

	void cpy_data_to_backend_tensor()
	{
		for (auto& kv : backend_tensor_data_map)
		{
			auto tensor = kv.first;
			auto data = kv.second;

			ggml_backend_tensor_set(tensor, data, 0, ggml_nbytes(tensor));
		}

		backend_tensor_data_map.clear();
	}

public:
	virtual std::string get_desc() = 0;

	GGMLRunner(ggml_backend_t backend)
		: backend(backend)
	{
		alloc_params_ctx();
	}

	virtual ~GGMLRunner()
	{
		free_params_buffer();
		free_compute_buffer();
		free_params_ctx();
		free_compute_ctx();
	}

	void reset_compute_ctx()
	{
		free_compute_ctx();
		alloc_compute_ctx();
	}

	bool alloc_params_buffer()
	{
		uint32_t num_tensors = ggml_tensor_num(params_ctx);
		params_buffer = ggml_backend_alloc_ctx_tensors(params_ctx, backend);
		if (params_buffer == NULL)
		{
			LOG_ERROR("%s alloc params backend buffer failed, num_tensors = %i",
					  get_desc().c_str(),
					  num_tensors);
			return false;
		}
		size_t params_buffer_size = ggml_backend_buffer_get_size(params_buffer);
		printf("%s params backend buffer size = % 6.2f MB(%s) (%i tensors)\n",
			   get_desc().c_str(),
			   params_buffer_size / (1024.0 * 1024.0),
			   ggml_backend_is_cpu(backend) ? "RAM" : "VRAM",
			   num_tensors);
		return true;
	}

	void free_params_buffer()
	{
		if (params_buffer != NULL)
		{
			ggml_backend_buffer_free(params_buffer);
			params_buffer = NULL;
		}
	}

	size_t get_params_buffer_size()
	{
		if (params_buffer != NULL)
		{
			return ggml_backend_buffer_get_size(params_buffer);
		}
		return 0;
	}

	void free_compute_buffer()
	{
		if (compute_allocr != NULL)
		{
			ggml_gallocr_free(compute_allocr);
			compute_allocr = NULL;
		}
	}

	// do copy after alloc graph
	void set_backend_tensor_data(struct ggml_tensor* tensor, const void* data)
	{
		backend_tensor_data_map[tensor] = data;
	}

	struct ggml_tensor* to_backend(struct ggml_tensor* tensor)
	{
		GGML_ASSERT(compute_ctx != NULL);
		if (tensor == NULL)
		{
			return NULL;
		}
		// it's performing a compute, check if backend isn't cpu
		if (!ggml_backend_is_cpu(backend) && (tensor->buffer == NULL || ggml_backend_buffer_is_host(tensor->buffer)))
		{
			// pass input tensors to gpu memory
			auto backend_tensor = ggml_dup_tensor(compute_ctx, tensor);

			set_backend_tensor_data(backend_tensor, tensor->data);
			return backend_tensor;
		}
		else
		{
			return tensor;
		}
	}

	void compute(get_graph_cb_t get_graph,
				 int n_threads,
				 bool free_compute_buffer_immediately = true,
				 struct ggml_tensor** output = NULL,
				 struct ggml_context* output_ctx = NULL)
	{
		alloc_compute_buffer(get_graph);
		reset_compute_ctx();
		struct ggml_cgraph* gf = get_graph();
		GGML_ASSERT(ggml_gallocr_alloc_graph(compute_allocr, gf));
		cpy_data_to_backend_tensor();
		if (ggml_backend_is_cpu(backend))
		{
			ggml_backend_cpu_set_n_threads(backend, n_threads);
		}

		ggml_backend_graph_compute(backend, gf);
	#ifdef GGML_PERF
		ggml_graph_print(gf);
	#endif
		if (output != NULL)
		{
			auto result = ggml_graph_node(gf, -1);
			if (*output == NULL && output_ctx != NULL)
			{
				*output = ggml_dup_tensor(output_ctx, result);
			}
			if (*output != NULL)
			{
				ggml_backend_tensor_get_and_sync(backend, result, (*output)->data, 0, ggml_nbytes(*output));
			}
		}

		if (free_compute_buffer_immediately)
		{
			free_compute_buffer();
		}
	}
};

class GGMLBlock
{
protected:
	typedef std::unordered_map<std::string, struct ggml_tensor*> ParameterMap;
	typedef std::unordered_map<std::string, std::shared_ptr<GGMLBlock>> GGMLBlockMap;
	GGMLBlockMap blocks;
	ParameterMap params;

	void init_blocks(struct ggml_context* ctx, std::map<std::string, enum ggml_type>& tensor_types, const std::string prefix = "")
	{
		for (auto& pair : blocks)
		{
			auto& block = pair.second;
			block->init(ctx, tensor_types, prefix + pair.first);
		}
	}

	virtual void init_params(struct ggml_context* ctx, std::map<std::string, enum ggml_type>& tensor_types, const std::string prefix = "") {}

public:
	void init(struct ggml_context* ctx, std::map<std::string, enum ggml_type>& tensor_types, std::string prefix = "")
	{
		if (prefix.size() > 0)
		{
			prefix = prefix + ".";
		}
		init_blocks(ctx, tensor_types, prefix);
		init_params(ctx, tensor_types, prefix);
	}

	size_t get_params_num()
	{
		size_t num_tensors = params.size();
		for (auto& pair : blocks)
		{
			auto& block = pair.second;

			num_tensors += block->get_params_num();
		}
		return num_tensors;
	};

	size_t get_params_mem_size()
	{
		size_t mem_size = 0;
		for (auto& pair : blocks)
		{
			auto& block = pair.second;

			mem_size += block->get_params_mem_size();
		}

		for (auto& pair : params)
		{
			mem_size += ggml_nbytes(pair.second);
		}

		return mem_size;
	}

	void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, std::string prefix = "")
	{
		if (prefix.size() > 0)
		{
			prefix = prefix + ".";
		}
		for (auto& pair : blocks)
		{
			auto& block = pair.second;
			block->get_param_tensors(tensors, prefix + pair.first);
		}

		for (auto& pair : params)
		{
			struct ggml_tensor* param = pair.second;
			tensors[prefix + pair.first] = pair.second;
		}
	}
};

class UnaryBlock : public GGMLBlock
{
public:
	virtual struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) = 0;
};

class Linear : public UnaryBlock
{
protected:
	int64_t in_features;
	int64_t out_features;
	bool bias;
	bool force_f32;

	void init_params(struct ggml_context* ctx, std::map<std::string, enum ggml_type>& tensor_types, const std::string prefix = "")
	{
		enum ggml_type wtype = (tensor_types.find(prefix + "weight") != tensor_types.end()) ? tensor_types[prefix + "weight"] : GGML_TYPE_F32;
		if (in_features % ggml_blck_size(wtype) != 0 || force_f32)
		{
			wtype = GGML_TYPE_F32;
		}
		params["weight"] = ggml_new_tensor_2d(ctx, wtype, in_features, out_features);
		if (bias)
		{
			enum ggml_type wtype = GGML_TYPE_F32;  //(tensor_types.ypes.find(prefix + "bias") != tensor_types.end()) ? tensor_types[prefix + "bias"] : GGML_TYPE_F32;
			params["bias"] = ggml_new_tensor_1d(ctx, wtype, out_features);
		}
	}

public:
	Linear(int64_t in_features,
		   int64_t out_features,
		   bool bias = true,
		   bool force_f32 = false)
		: in_features(in_features),
		out_features(out_features),
		bias(bias),
		force_f32(force_f32)
	{
	}

	struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x)
	{
		struct ggml_tensor* w = params["weight"];
		struct ggml_tensor* b = NULL;
		if (bias)
		{
			b = params["bias"];
		}
		return ggml_nn_linear(ctx, x, w, b);
	}
};

class Embedding : public UnaryBlock
{
protected:
	int64_t embedding_dim;
	int64_t num_embeddings;
	void init_params(struct ggml_context* ctx, std::map<std::string, enum ggml_type>& tensor_types, const std::string prefix = "")
	{
		enum ggml_type wtype = (tensor_types.find(prefix + "weight") != tensor_types.end()) ? tensor_types[prefix + "weight"] : GGML_TYPE_F32;
		params["weight"] = ggml_new_tensor_2d(ctx, wtype, embedding_dim, num_embeddings);
	}

public:
	Embedding(int64_t num_embeddings, int64_t embedding_dim)
		: embedding_dim(embedding_dim),
		num_embeddings(num_embeddings)
	{
	}

	struct ggml_tensor* get_weight()
	{
		return params["weight"];
	}

	struct ggml_tensor* forward(struct ggml_context* ctx,
								struct ggml_tensor* input_ids)
	{
		// input_ids: [N, n_token]
		auto weight = params["weight"];

		// There are issues with ggml batch inference, so we are expanding it here first.
		// TODO: fix ggml batch inference
		int64_t n = input_ids->ne[1];
		input_ids = ggml_reshape_1d(ctx, input_ids, input_ids->ne[0] * input_ids->ne[1]);

		input_ids = ggml_reshape_3d(ctx, input_ids, input_ids->ne[0], 1, input_ids->ne[1]);
		auto embedding = ggml_get_rows(ctx, weight, input_ids);
		embedding = ggml_reshape_3d(ctx, embedding, embedding->ne[0], embedding->ne[1] / n, n);

		// [N, n_token, embedding_dim]
		return embedding;
	}
};

class Conv1d : public UnaryBlock
{
protected:
	int64_t in_channels;
	int64_t out_channels;
	int kernel_size;
	int stride;
	int padding;
	int dilation;
	bool bias;

	void init_params(struct ggml_context* ctx, std::map<std::string, enum ggml_type>& tensor_types, const std::string prefix = "")
	{
		enum ggml_type wtype = GGML_TYPE_F16;  //(tensor_types.find(prefix + "weight") != tensor_types.end()) ? tensor_types[prefix + "weight"] : GGML_TYPE_F16;
		params["weight"] = ggml_new_tensor_3d(ctx, wtype, kernel_size, in_channels, out_channels);
		if (bias)
		{
			enum ggml_type wtype = GGML_TYPE_F32;  // (tensor_types.find(prefix + "bias") != tensor_types.end()) ? tensor_types[prefix + "bias"] : GGML_TYPE_F32;
			params["bias"] = ggml_new_tensor_1d(ctx, wtype, out_channels);
		}
	}

public:
	bool x_im2col;

	Conv1d(int64_t in_channels,
		   int64_t out_channels,
		   int kernel_size,
		   int stride = 1,
		   int padding = 0,
		   int dilation = 1,
		   bool bias = true,
		   bool x_im2col = false)
		: in_channels(in_channels),
		out_channels(out_channels),
		kernel_size(kernel_size),
		stride(stride),
		padding(padding),
		dilation(dilation),
		bias(bias),
		x_im2col(x_im2col)
	{
	}

	struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x)
	{
		struct ggml_tensor* w = params["weight"];
		struct ggml_tensor* b = NULL;
		if (bias)
		{
			b = params["bias"];
		}
		if (x_im2col)
			return ggml_nn_conv_1d_im2col(ctx, x, w, b, stride, padding, dilation);
		else
			return ggml_nn_conv_1d(ctx, x, w, b, stride, padding, dilation);
	}

	struct ggml_tensor* im2col(struct ggml_context* ctx, struct ggml_tensor* x)
	{
		return ggml_im2col(ctx, params["weight"], x, stride, 0, padding, 0, dilation, 0, false, GGML_TYPE_F16);
	}
};

class DepthwiseConv1d : public UnaryBlock
{
protected:
	int64_t in_channels;
	int64_t out_channels;
	int kernel_size;
	int stride;
	int padding;
	int dilation;
	bool bias;

	void init_params(struct ggml_context* ctx, std::map<std::string, enum ggml_type>& tensor_types, const std::string prefix = "")
	{
		enum ggml_type wtype = GGML_TYPE_F16;  //(tensor_types.find(prefix + "weight") != tensor_types.end()) ? tensor_types[prefix + "weight"] : GGML_TYPE_F16;
		params["weight"] = ggml_new_tensor_3d(ctx, wtype, kernel_size, 1, out_channels);
		if (bias)
		{
			enum ggml_type wtype = GGML_TYPE_F32;  // (tensor_types.find(prefix + "bias") != tensor_types.end()) ? tensor_types[prefix + "bias"] : GGML_TYPE_F32;
			params["bias"] = ggml_new_tensor_1d(ctx, wtype, out_channels);
		}
	}

public:
	DepthwiseConv1d(int64_t in_channels,
					int64_t out_channels,
					int kernel_size,
					int stride = 1,
					int padding = 0,
					int dilation = 1,
					bool bias = true)
		: in_channels(in_channels),
		out_channels(out_channels),
		kernel_size(kernel_size),
		stride(stride),
		padding(padding),
		dilation(dilation),
		bias(bias)
	{
	}

	struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x)
	{
		struct ggml_tensor* w = params["weight"];
		struct ggml_tensor* b = NULL;
		if (bias)
		{
			b = params["bias"];
		}
		return ggml_nn_conv_1d_dw(ctx, x, w, b, stride, padding, dilation);
	}
};

class ConvTranspose1d : public UnaryBlock
{
protected:
	int64_t in_channels;
	int64_t out_channels;
	int kernel_size;
	int stride;
	int padding;
	int dilation;
	bool bias;

	void init_params(struct ggml_context* ctx, std::map<std::string, enum ggml_type>& tensor_types, const std::string prefix = "")
	{
		enum ggml_type wtype = GGML_TYPE_F16;  //(tensor_types.find(prefix + "weight") != tensor_types.end()) ? tensor_types[prefix + "weight"] : GGML_TYPE_F16;
		params["weight"] = ggml_new_tensor_3d(ctx, wtype, kernel_size, out_channels, in_channels);
		if (bias)
		{
			enum ggml_type wtype = GGML_TYPE_F32;  // (tensor_types.find(prefix + "bias") != tensor_types.end()) ? tensor_types[prefix + "bias"] : GGML_TYPE_F32;
			params["bias"] = ggml_new_tensor_1d(ctx, wtype, out_channels);
		}
	}

public:
	ConvTranspose1d(int64_t in_channels,
					int64_t out_channels,
					int kernel_size,
					int stride = 1,
					int padding = 0,
					int dilation = 1,
					bool bias = true)
		: in_channels(in_channels),
		out_channels(out_channels),
		kernel_size(kernel_size),
		stride(stride),
		padding(padding),
		dilation(dilation),
		bias(bias)
	{
	}

	struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x)
	{
		struct ggml_tensor* w = params["weight"];
		struct ggml_tensor* b = NULL;
		if (bias)
		{
			b = params["bias"];
		}

		if (w->ne[0] % stride == 0)
			return ggml_nn_conv_transpose_1d_quick(ctx, x, w, b, stride, 0, dilation);
		else
			return ggml_nn_conv_transpose_1d(ctx, x, w, b, stride, 0, dilation);
	}
};

class Conv2d : public UnaryBlock
{
protected:
	int64_t in_channels;
	int64_t out_channels;
	std::pair<int, int> kernel_size;
	std::pair<int, int> stride;
	std::pair<int, int> padding;
	std::pair<int, int> dilation;
	bool bias;

	void init_params(struct ggml_context* ctx, std::map<std::string, enum ggml_type>& tensor_types, const std::string prefix = "")
	{
		enum ggml_type wtype = GGML_TYPE_F16;  //(tensor_types.find(prefix + "weight") != tensor_types.end()) ? tensor_types[prefix + "weight"] : GGML_TYPE_F16;
		params["weight"] = ggml_new_tensor_4d(ctx, wtype, kernel_size.second, kernel_size.first, in_channels, out_channels);
		if (bias)
		{
			enum ggml_type wtype = GGML_TYPE_F32;  // (tensor_types.find(prefix + "bias") != tensor_types.end()) ? tensor_types[prefix + "bias"] : GGML_TYPE_F32;
			params["bias"] = ggml_new_tensor_1d(ctx, wtype, out_channels);
		}
	}

public:
	Conv2d(int64_t in_channels,
		   int64_t out_channels,
		   std::pair<int, int> kernel_size,
		   std::pair<int, int> stride = { 1, 1 },
		   std::pair<int, int> padding = { 0, 0 },
		   std::pair<int, int> dilation = { 1, 1 },
		   bool bias = true)
		: in_channels(in_channels),
		out_channels(out_channels),
		kernel_size(kernel_size),
		stride(stride),
		padding(padding),
		dilation(dilation),
		bias(bias)
	{
	}

	struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x)
	{
		struct ggml_tensor* w = params["weight"];
		struct ggml_tensor* b = NULL;
		if (bias)
		{
			b = params["bias"];
		}
		return ggml_nn_conv_2d(ctx, x, w, b, stride.second, stride.first, padding.second, padding.first, dilation.second, dilation.first);
	}
};

class Conv3dnx1x1 : public UnaryBlock
{
protected:
	int64_t in_channels;
	int64_t out_channels;
	int64_t kernel_size;
	int64_t stride;
	int64_t padding;
	int64_t dilation;
	bool bias;

	void init_params(struct ggml_context* ctx, std::map<std::string, enum ggml_type>& tensor_types, const std::string prefix = "")
	{
		enum ggml_type wtype = GGML_TYPE_F16;                                                              //(tensor_types.find(prefix + "weight") != tensor_types.end()) ? tensor_types[prefix + "weight"] : GGML_TYPE_F16;
		params["weight"] = ggml_new_tensor_4d(ctx, wtype, 1, kernel_size, in_channels, out_channels);  // 5d => 4d
		if (bias)
		{
			enum ggml_type wtype = GGML_TYPE_F32;  //(tensor_types.find(prefix + "bias") != tensor_types.end()) ? tensor_types[prefix + "bias"] : GGML_TYPE_F32;
			params["bias"] = ggml_new_tensor_1d(ctx, wtype, out_channels);
		}
	}

public:
	Conv3dnx1x1(int64_t in_channels,
				int64_t out_channels,
				int64_t kernel_size,
				int64_t stride = 1,
				int64_t padding = 0,
				int64_t dilation = 1,
				bool bias = true)
		: in_channels(in_channels),
		out_channels(out_channels),
		kernel_size(kernel_size),
		stride(stride),
		padding(padding),
		dilation(dilation),
		bias(bias)
	{
	}

	// x: [N, IC, ID, IH*IW]
	// result: [N, OC, OD, OH*OW]
	struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x)
	{
		struct ggml_tensor* w = params["weight"];
		struct ggml_tensor* b = NULL;
		if (bias)
		{
			b = params["bias"];
		}
		return ggml_nn_conv_3d_nx1x1(ctx, x, w, b, stride, padding, dilation);
	}
};

class LayerNorm : public UnaryBlock
{
protected:
	int64_t normalized_shape;
	float eps;
	bool elementwise_affine;
	bool bias;

	void init_params(struct ggml_context* ctx, std::map<std::string, enum ggml_type>& tensor_types, const std::string prefix = "")
	{
		if (elementwise_affine)
		{
			enum ggml_type wtype = GGML_TYPE_F32;  //(tensor_types.ypes.find(prefix + "weight") != tensor_types.end()) ? tensor_types[prefix + "weight"] : GGML_TYPE_F32;
			params["gamma"] = ggml_new_tensor_1d(ctx, wtype, normalized_shape);
			if (bias)
			{
				enum ggml_type wtype = GGML_TYPE_F32;  //(tensor_types.ypes.find(prefix + "bias") != tensor_types.end()) ? tensor_types[prefix + "bias"] : GGML_TYPE_F32;
				params["beta"] = ggml_new_tensor_1d(ctx, wtype, normalized_shape);
			}
		}
	}

public:
	LayerNorm(int64_t normalized_shape,
			  float eps = 1e-05f,
			  bool elementwise_affine = true,
			  bool bias = true)
		: normalized_shape(normalized_shape),
		eps(eps),
		elementwise_affine(elementwise_affine),
		bias(bias)
	{
	}

	struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x)
	{
		struct ggml_tensor* w = NULL;
		struct ggml_tensor* b = NULL;

		if (elementwise_affine)
		{
			w = params["gamma"];
			if (bias)
			{
				b = params["beta"];
			}
		}
		return ggml_nn_layer_norm(ctx, x, w, b, eps);
	}
};

class GroupNorm : public GGMLBlock
{
protected:
	int64_t num_groups;
	int64_t num_channels;
	float eps;
	bool affine;

	void init_params(struct ggml_context* ctx, std::map<std::string, enum ggml_type>& tensor_types, const std::string prefix = "")
	{
		if (affine)
		{
			enum ggml_type wtype = GGML_TYPE_F32;  //(tensor_types.find(prefix + "weight") != tensor_types.end()) ? tensor_types[prefix + "weight"] : GGML_TYPE_F32;
			enum ggml_type bias_wtype = GGML_TYPE_F32;  //(tensor_types.find(prefix + "bias") != tensor_types.end()) ? tensor_types[prefix + "bias"] : GGML_TYPE_F32;
			params["weight"] = ggml_new_tensor_1d(ctx, wtype, num_channels);
			params["bias"] = ggml_new_tensor_1d(ctx, bias_wtype, num_channels);
		}
	}

public:
	GroupNorm(int64_t num_groups,
			  int64_t num_channels,
			  float eps = 1e-05f,
			  bool affine = true)
		: num_groups(num_groups),
		num_channels(num_channels),
		eps(eps),
		affine(affine)
	{
	}

	struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x)
	{
		struct ggml_tensor* w = NULL;
		struct ggml_tensor* b = NULL;
		if (affine)
		{
			w = params["weight"];
			b = params["bias"];
		}
		return ggml_nn_group_norm(ctx, x, w, b, num_groups);
	}
};

class GroupNorm32 : public GroupNorm
{
public:
	GroupNorm32(int64_t num_channels)
		: GroupNorm(32, num_channels, 1e-06f)
	{
	}
};


#endif  // __GGML_EXTEND__HPP__
