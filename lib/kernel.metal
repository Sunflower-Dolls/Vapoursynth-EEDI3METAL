#include <metal_stdlib>
using namespace metal;

struct EEDI3Params {
    int width;
    int height;
    int mdis;
    int tpitch;
    float alpha, beta, gamma;
    float remainingWeight;
    int cost3;
    int ucubic;
    int has_mclip;
    int nrad;
    int field;
    int stride;
    int dh;
};

METAL_FUNC int mirror_x(int i, int size) {
    // [-size, 2*size]
    int i_abs = abs(i);
    if (i_abs >= size) {
        i_abs = 2 * size - 2 - i_abs;
    }
    return clamp(i_abs, 0, size - 1);
}

METAL_FUNC int get_clamped_y_row_index(int y_target, int h_src,
                                       int field_parity, int size, bool dh) {
    int fy = (y_target - field_parity) >> 1;

    int limit_h = dh ? (size >> 1) : (size - field_parity + 1) >> 1;

    int i = fy;
    if (i < 0) {
        i = (field_parity == 0) ? -i : -1 - i;
    } else if (i >= limit_h) {
        i = (field_parity == 0) ? (2 * limit_h - 1 - i) : (2 * limit_h - 2 - i);
    }
    i = clamp(i, 0, limit_h - 1);

    if (dh)
        return i;
    return i * 2 + field_parity;
}

kernel void copy_field(device float* src_buf [[buffer(0)]],
                       device float* dst_buf [[buffer(1)]],
                       constant EEDI3Params& p [[buffer(2)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= (uint)p.width)
        return;

    int target_field = 1 - p.field;
    int dst_y = target_field + gid.y * 2;

    if (dst_y >= p.height)
        return;

    uint src_y = p.dh ? gid.y : (uint)dst_y;

    float val = src_buf[src_y * p.stride + gid.x];
    dst_buf[dst_y * p.stride + gid.x] = val;
}

kernel void dilate_mask(device float* mclip_buf [[buffer(0)]],
                        device bool* bmask_global [[buffer(1)]],
                        constant EEDI3Params& p [[buffer(2)]],
                        threadgroup float* shared_mem [[threadgroup(0)]],
                        uint2 gid [[thread_position_in_grid]],
                        uint2 tid [[thread_position_in_threadgroup]],
                        uint2 tg_pos [[threadgroup_position_in_grid]]) {
    
    int mdis = p.mdis;
    int halo = mdis;
    int sw = 32 + 2 * halo + 1;
    
    int ty = tid.y;
    int tx = tid.x;
    
    threadgroup float* my_shared_row = shared_mem + ty * sw;
    
    int group_start_x = tg_pos.x * 32;
    
    uint src_y;
    if (p.dh) {
        src_y = p.field + gid.y * 2;
        if (p.dh) {
            src_y = gid.y;
        } else {
            src_y = p.field + gid.y * 2;
        }
    } else {
        if (p.dh) {
            src_y = gid.y;
        } else {
            src_y = p.field + gid.y * 2;
        }
    }

    bool valid_row = (src_y < (uint)p.height);
    
    device float* row_mclip = nullptr;
    if (valid_row) {
        row_mclip = mclip_buf + src_y * p.stride;
    }

    for (int k = tx; k < sw; k += 32) {
        int global_x = group_start_x - halo + k;
        
        float val = 0.0f;
        if (valid_row) {
             if (global_x >= 0 && global_x < p.width) {
                 val = row_mclip[global_x];
             }
        }
        my_shared_row[k] = val;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (gid.x >= (uint)p.width || !valid_row)
        return;

    bool mask = false;
    int center = halo + tx;

    for (int k = -mdis; k <= mdis; ++k) {
        if (my_shared_row[center + k] != 0.0f) {
            mask = true;
            break;
        }
    }
    
    bmask_global[gid.y * p.width + gid.x] = mask;
}

kernel void calc_costs(device const float* src_buf [[buffer(0)]],
                       device float* cost_buffer [[buffer(1)]],
                       constant EEDI3Params& p [[buffer(2)]],
                       device const bool* bmask_global [[buffer(3)]],
                       threadgroup float* shared_mem [[threadgroup(0)]],
                       uint2 gid [[thread_position_in_grid]],
                       uint2 tid [[thread_position_in_threadgroup]],
                       uint2 tg_pos [[threadgroup_position_in_grid]]) {
    int halo = (p.cost3 ? 2 * p.mdis : p.mdis) + p.nrad;
    int sw = 32 + 2 * halo + 1;

    int t_x = tid.x;
    int t_y = tid.y;
    int y = p.field + gid.y * 2;

    threadgroup float* s_m3 = shared_mem + (t_y * 4 + 0) * sw;
    threadgroup float* s_m1 = shared_mem + (t_y * 4 + 1) * sw;
    threadgroup float* s_p1 = shared_mem + (t_y * 4 + 2) * sw;
    threadgroup float* s_p3 = shared_mem + (t_y * 4 + 3) * sw;

    if (y < p.height) {
        int ref_field = 1 - p.field;
        int src_h = p.height;
        int stride = p.stride;

        int r_m3 = get_clamped_y_row_index(y - 3, src_h, ref_field, src_h, p.dh);
        int r_m1 = get_clamped_y_row_index(y - 1, src_h, ref_field, src_h, p.dh);
        int r_p1 = get_clamped_y_row_index(y + 1, src_h, ref_field, src_h, p.dh);
        int r_p3 = get_clamped_y_row_index(y + 3, src_h, ref_field, src_h, p.dh);

        device const float* g_row_m3 = src_buf + r_m3 * stride;
        device const float* g_row_m1 = src_buf + r_m1 * stride;
        device const float* g_row_p1 = src_buf + r_p1 * stride;
        device const float* g_row_p3 = src_buf + r_p3 * stride;

        int group_base_x = tg_pos.x * 32;
        
        for (int k = t_x; k < sw; k += 32) {
            int global_read_x = group_base_x - halo + k;
            int ix = mirror_x(global_read_x, p.width);
            
            s_m3[k] = g_row_m3[ix];
            s_m1[k] = g_row_m1[ix];
            s_p1[k] = g_row_p1[ix];
            s_p3[k] = g_row_p3[ix];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (gid.x >= (uint)p.width || y >= p.height)
        return;

    int s_center = halo + t_x;

    device const bool* bmask = (p.has_mclip && bmask_global)
                                   ? (bmask_global + gid.y * p.width)
                                   : nullptr;
    int x = gid.x;
    bool use_eedi3 = (!bmask || bmask[x]);
    int width = p.width;

    device float* dst_ptr =
        cost_buffer + (size_t)gid.y * width * p.tpitch + (size_t)x * p.tpitch;

    int umax = min(min(x, width - 1 - x), p.mdis);

    for (int u_idx = 0; u_idx < p.tpitch; ++u_idx) {
        int u = u_idx - p.mdis;

        if (!use_eedi3 || abs(u) > umax) {
            dst_ptr[u_idx] = FLT_MAX;
            continue;
        }

        float s0 = 0.0f;
        threadgroup float* ptr_m3_pos = s_m3 + s_center + u;
        threadgroup float* ptr_m1_neg = s_m1 + s_center - u;
        threadgroup float* ptr_m1_pos = s_m1 + s_center + u;
        threadgroup float* ptr_p1_neg = s_p1 + s_center - u;
        threadgroup float* ptr_p1_pos = s_p1 + s_center + u;
        threadgroup float* ptr_p3_neg = s_p3 + s_center - u;

        for (int k = -p.nrad; k <= p.nrad; ++k) {
            float val_3p = ptr_m3_pos[k];
            float val_1p_neg = ptr_m1_neg[k];
            float val_1p_pos = ptr_m1_pos[k];
            float val_1n_neg = ptr_p1_neg[k];
            float val_1n_pos = ptr_p1_pos[k];
            float val_3n = ptr_p3_neg[k];

            s0 += abs(val_3p - val_1p_neg) + abs(val_1p_pos - val_1n_neg) +
                  abs(val_1n_pos - val_3n);
        }

        float final_s = s0;

        if (p.cost3) {
            int u2 = u * 2;
            float s1 = -FLT_MAX;
            float s2 = -FLT_MAX;

            bool s1_valid = (u >= 0 && x >= u2) || (u <= 0 && x < width + u2);
            if (s1_valid) {
                float temp_s1 = 0.0f;
                threadgroup float* ptr_m3_c = s_m3 + s_center;
                threadgroup float* ptr_m1_neg2 = s_m1 + s_center - u2;
                threadgroup float* ptr_m1_c = s_m1 + s_center;
                threadgroup float* ptr_p1_neg2 = s_p1 + s_center - u2;
                threadgroup float* ptr_p1_c = s_p1 + s_center;
                threadgroup float* ptr_p3_neg2 = s_p3 + s_center - u2;

                for (int k = -p.nrad; k <= p.nrad; ++k) {
                   float v_3p = ptr_m3_c[k];
                   float v_1p = ptr_m1_neg2[k];
                   float v_1p_c = ptr_m1_c[k];
                   float v_1n = ptr_p1_neg2[k];
                   float v_1n_c = ptr_p1_c[k];
                   float v_3n = ptr_p3_neg2[k];

                    temp_s1 += abs(v_3p - v_1p) + abs(v_1p_c - v_1n) +
                               abs(v_1n_c - v_3n);
                }
                s1 = temp_s1;
            }

            bool s2_valid = (u <= 0 && x >= -u2) || (u >= 0 && x < width + u2);
            if (s2_valid) {
                float temp_s2 = 0.0f;
                threadgroup float* ptr_m3_pos2 = s_m3 + s_center + u2;
                threadgroup float* ptr_m1_c = s_m1 + s_center;
                threadgroup float* ptr_m1_pos2 = s_m1 + s_center + u2;
                threadgroup float* ptr_p1_c = s_p1 + s_center;
                threadgroup float* ptr_p1_pos2 = s_p1 + s_center + u2;
                threadgroup float* ptr_p3_c = s_p3 + s_center;

                for (int k = -p.nrad; k <= p.nrad; ++k) {
                    float v_3p = ptr_m3_pos2[k];
                    float v_1p = ptr_m1_c[k];
                    float v_1p_c = ptr_m1_pos2[k];
                    float v_1n = ptr_p1_c[k];
                    float v_1n_c = ptr_p1_pos2[k];
                    float v_3n = ptr_p3_c[k];

                    temp_s2 += abs(v_3p - v_1p) + abs(v_1p_c - v_1n) +
                               abs(v_1n_c - v_3n);
                }
                s2 = temp_s2;
            }

            float val_s1 = (s1 > -FLT_MAX) ? s1 : ((s2 > -FLT_MAX) ? s2 : s0);
            float val_s2 =
                (s2 > -FLT_MAX) ? s2 : ((val_s1 > -FLT_MAX) ? val_s1 : s0);

            final_s += val_s1 + val_s2;
        }

        float ip_p1 = s_m1[s_center + u];
        float ip_n1 = s_p1[s_center - u];
        float ip = (ip_p1 + ip_n1) * 0.5f;

        float src_p = s_m1[s_center];
        float src_n = s_p1[s_center];

        float v = abs(src_p - ip) + abs(src_n - ip);

        dst_ptr[u_idx] =
            p.alpha * final_s + p.beta * abs(u) + p.remainingWeight * v;
    }
}

kernel void viterbi_scan(device float* cost_buffer [[buffer(0)]],
                         device int* dmap_global [[buffer(1)]],
                         device int8_t* pbackt_global [[buffer(2)]],
                         constant EEDI3Params& p [[buffer(3)]],
                         device bool* bmask_global [[buffer(4)]],
                         uint2 gid [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]]) {
    int y_idx = gid.y;

    // Up to 3 states per thread.
    constexpr int STATES_PER_THREAD = 3;
    float s_prev[STATES_PER_THREAD];
    float s_curr[STATES_PER_THREAD];

    for (int k = 0; k < STATES_PER_THREAD; ++k) {
        s_prev[k] = 0.0f;
    }

    device int8_t* row_backt_ptr = pbackt_global + (y_idx * p.width * p.tpitch);
    device float* row_cost_ptr = cost_buffer + (y_idx * p.width * p.tpitch);
    device bool* bmask = (p.has_mclip && bmask_global)
                             ? (bmask_global + y_idx * p.width)
                             : nullptr;

    for (int x = 0; x < p.width; ++x) {
        bool use_eedi3 = (!bmask || bmask[x]);

        float cost_data[STATES_PER_THREAD];
        for (int k = 0; k < STATES_PER_THREAD; ++k) {
            int u_idx = tid + k * 32;
            if (u_idx < p.tpitch) {
                cost_data[k] = row_cost_ptr[x * p.tpitch + u_idx];
            } else {
                cost_data[k] = FLT_MAX;
            }
        }

        float min_val[STATES_PER_THREAD];
        int best_offset[STATES_PER_THREAD];

        if (x == 0) {
            for (int k = 0; k < STATES_PER_THREAD; ++k) {
                int u_idx = tid + k * 32;
                if (u_idx < p.tpitch) {
                    min_val[k] = 0.0f;
                } else {
                    min_val[k] = FLT_MAX;
                }
                best_offset[k] = 0;
            }
        } else if (use_eedi3) {
            for (int k = 0; k < STATES_PER_THREAD; ++k) {
                int u_idx = tid + k * 32;
                float c = s_prev[k];
                float l = FLT_MAX;
                float r = FLT_MAX;

                if (tid > 0) {
                    l = simd_shuffle_up(c, 1);
                } else {
                    if (k > 0) {
                        l = simd_shuffle(s_prev[k - 1], 31);
                    }
                }

                if (tid < 31) {
                    r = simd_shuffle_down(c, 1);
                } else {
                    if (k < STATES_PER_THREAD - 1) {
                        r = simd_shuffle(s_prev[k + 1], 0);
                    }
                }

                if (u_idx >= p.tpitch) {
                    min_val[k] = FLT_MAX;
                    best_offset[k] = 0;
                    continue;
                }

                float best_c = c;
                int best_d = 0;

                if (u_idx - 1 >= 0) {
                    if (l < FLT_MAX) {
                        float val = l + p.gamma;
                        if (val < best_c) {
                            best_c = val;
                            best_d = -1;
                        }
                    }
                }

                if (u_idx + 1 < p.tpitch) {
                    if (r < FLT_MAX) {
                        float val = r + p.gamma;
                        if (val < best_c) {
                            best_c = val;
                            best_d = 1;
                        }
                    }
                }

                min_val[k] = best_c;
                best_offset[k] = best_d;
            }
        } else {
            for (int k = 0; k < STATES_PER_THREAD; ++k) {
                int u_idx = tid + k * 32;
                if (u_idx < p.tpitch) {
                    min_val[k] = s_prev[k];
                } else {
                    min_val[k] = FLT_MAX;
                }
                best_offset[k] = 0;
            }
        }

        float row_min = FLT_MAX;

        for (int k = 0; k < STATES_PER_THREAD; ++k) {
            int u_idx = tid + k * 32;

            if (u_idx >= p.tpitch) {
                s_curr[k] = FLT_MAX;
                continue;
            }

            float total_cost;
            if (use_eedi3) {
                total_cost = min_val[k] + cost_data[k];
                total_cost = min(total_cost, FLT_MAX * 0.9f);
            } else {
                total_cost = min_val[k];
            }

            s_curr[k] = total_cost;
            row_backt_ptr[x * p.tpitch + u_idx] = (int8_t)best_offset[k];

            row_min = min(row_min, total_cost);
        }

        // Normalize
        row_min = simd_min(row_min);

        for (int k = 0; k < STATES_PER_THREAD; ++k) {
            int u_idx = tid + k * 32;
            if (u_idx < p.tpitch) {
                if (s_curr[k] < FLT_MAX * 0.9f) {
                    s_curr[k] -= row_min;
                }
                s_prev[k] = s_curr[k];
            } else {
                s_prev[k] = FLT_MAX;
            }
        }
    }

    // Backtracking
    if (tid == 0) {
        device int* row_dmap = dmap_global + (y_idx * p.width);

        int best_idx = p.mdis;
        row_dmap[p.width - 1] = 0;

        for (int x = p.width - 2; x >= 0; --x) {
            int8_t offset = row_backt_ptr[(x + 1) * p.tpitch + best_idx];
            best_idx += offset;
            row_dmap[x] = best_idx - p.mdis;
        }
    }
}

kernel void interpolate(device const float* src_buf [[buffer(0)]],
                        device float* dst_buf [[buffer(1)]],
                        device int* dmap_global [[buffer(2)]],
                        constant EEDI3Params& p [[buffer(3)]],
                        device const bool* bmask_global [[buffer(4)]],
                        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= (uint)p.width)
        return;

    int y = p.field + gid.y * 2;
    if (y >= p.height)
        return;

    int x = gid.x;

    int ref_field = 1 - p.field;
    int src_h = p.height;
    int stride = p.stride;

    int r_m1 = get_clamped_y_row_index(y - 1, src_h, ref_field, src_h, p.dh);
    int r_p1 = get_clamped_y_row_index(y + 1, src_h, ref_field, src_h, p.dh);

    device const float* row_m1 = src_buf + r_m1 * stride;
    device const float* row_p1 = src_buf + r_p1 * stride;

    int r_m3 = 0, r_p3 = 0;
    device const float* row_m3 = nullptr;
    device const float* row_p3 = nullptr;

    if (p.ucubic) {
        r_m3 = get_clamped_y_row_index(y - 3, src_h, ref_field, src_h, p.dh);
        r_p3 = get_clamped_y_row_index(y + 3, src_h, ref_field, src_h, p.dh);
        row_m3 = src_buf + r_m3 * stride;
        row_p3 = src_buf + r_p3 * stride;
    }

    device const bool* bmask = (p.has_mclip && bmask_global)
                                   ? (bmask_global + gid.y * p.width)
                                   : nullptr;

    int dir;
    int dmap_idx = gid.y * p.width + x;

    if (bmask && !bmask[x]) {
        dir = 0;
    } else {
        dir = dmap_global[dmap_idx];
    }

    dmap_global[dmap_idx] = dir;

    float val = 0.0f;

    float p1 = row_m1[mirror_x(x + dir, p.width)];
    float p2 = row_p1[mirror_x(x - dir, p.width)];

    int dir3 = dir * 3;
    int absDir3 = abs(dir3);

    if (p.ucubic && x >= absDir3 && x <= p.width - 1 - absDir3) {
        float p0 = row_m3[mirror_x(x + dir3, p.width)];
        float p3 = row_p3[mirror_x(x - dir3, p.width)];

        float sum_inner = p1 + p2;
        float sum_outer = p0 + p3;
        val = fma(0.5625f, sum_inner, -0.0625f * sum_outer);
    } else {
        val = 0.5f * (p1 + p2);
    }

    dst_buf[y * stride + x] = val;
}

kernel void copy_buffer(device const uchar* src [[buffer(0)]],
                        device uchar* dst [[buffer(1)]],
                        constant uint& src_stride [[buffer(2)]],
                        constant uint& dst_stride [[buffer(3)]],
                        constant uint& width_bytes [[buffer(4)]],
                        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= width_bytes)
        return;

    dst[gid.y * dst_stride + gid.x] = src[gid.y * src_stride + gid.x];
}