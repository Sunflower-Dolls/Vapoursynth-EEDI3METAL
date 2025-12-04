#ifndef EEDI3_H
#define EEDI3_H

// NOLINTBEGIN(cppcoreguidelines-init-variables)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

#include <VSHelper4.h>
#include <VapourSynth4.h>

static constexpr int MARGIN_H =
    12; // Left and right margins for the virtual source frame
static constexpr int MARGIN_V = 4; // Top and bottom margins

struct EEDI3Data final {
    VSNode *node, *sclip, *mclip;
    VSVideoInfo vi;
    VSVideoFormat padFormat;
    int field, nrad, mdis, vcheck;
    std::array<bool, 3> process;
    bool dh, ucubic, cost3;
    float alpha, beta, gamma, vthresh2;
    float remainingWeight, rcpVthresh0, rcpVthresh1, rcpVthresh2;
    int tpitch;
};

inline void copyPad(const VSFrame* src, VSFrame* dst, const int plane,
                    const bool dh, const int off, const VSAPI* vsapi) noexcept {
    const int srcWidth = vsapi->getFrameWidth(src, plane);
    const int dstWidth = vsapi->getFrameWidth(dst, 0);
    const int srcHeight = vsapi->getFrameHeight(src, plane);
    const int dstHeight = vsapi->getFrameHeight(dst, 0);
    const ptrdiff_t srcStride =
        vsapi->getStride(src, plane) / static_cast<ptrdiff_t>(sizeof(float));
    const ptrdiff_t dstStride =
        vsapi->getStride(dst, 0) / static_cast<ptrdiff_t>(sizeof(float));
    const auto* srcp =
        reinterpret_cast<const float*>(vsapi->getReadPtr(src, plane));
    auto* VS_RESTRICT dstp =
        reinterpret_cast<float*>(vsapi->getWritePtr(dst, 0));

    if (!dh) {
        vsh::bitblt(dstp + (dstStride * (MARGIN_V + off)) + MARGIN_H,
                    vsapi->getStride(dst, 0) * 2, srcp + (srcStride * off),
                    vsapi->getStride(src, plane) * 2, srcWidth * sizeof(float),
                    srcHeight / 2);
    } else {
        vsh::bitblt(dstp + (dstStride * (MARGIN_V + off)) + MARGIN_H,
                    vsapi->getStride(dst, 0) * 2, srcp,
                    vsapi->getStride(src, plane), srcWidth * sizeof(float),
                    srcHeight);
    }

    dstp += dstStride * (MARGIN_V + off);

    for (int y = MARGIN_V + off; y < dstHeight - MARGIN_V; y += 2) {
        for (int x = 0; x < MARGIN_H; x++) {
            dstp[x] = dstp[(MARGIN_H * 2) - x];
        }

        for (int x = dstWidth - MARGIN_H, c = 2; x < dstWidth; x++, c += 2) {
            dstp[x] = dstp[x - c];
        }

        dstp += dstStride * 2;
    }

    dstp = reinterpret_cast<float*>(vsapi->getWritePtr(dst, 0));

    for (int y = off; y < MARGIN_V; y += 2) {
        memcpy(dstp + (dstStride * y), dstp + (dstStride * (MARGIN_V * 2 - y)),
               dstWidth * sizeof(float));
    }

    for (int y = dstHeight - MARGIN_V + off, c = 2 + (2 * off); y < dstHeight;
         y += 2, c += 4) {
        memcpy(dstp + (dstStride * y), dstp + (dstStride * (y - c)),
               dstWidth * sizeof(float));
    }
}

inline void vCheck(const float* srcp, const float* scpp,
                   float* VS_RESTRICT dstp, const int* dmap,
                   float* VS_RESTRICT tline, const int field_n, const int width,
                   const int height, const ptrdiff_t srcStride,
                   const ptrdiff_t dstStride,
                   const EEDI3Data* VS_RESTRICT d) noexcept {
    for (int y = MARGIN_V + field_n; y < height - MARGIN_V; y += 2) {
        if (y >= 6 && y < height - 6) {
            const auto* dst3p = srcp - (srcStride * 3) + MARGIN_H;
            auto* dst2p = dstp - (dstStride * 2);
            auto* dst1p = dstp - dstStride;
            auto* dst1n = dstp + dstStride;
            auto* dst2n = dstp + (dstStride * 2);
            const auto* dst3n = srcp + (srcStride * 3) + MARGIN_H;

            for (int x = 0; x < width; x++) {
                const int dirc = dmap[x];
                float cint = 0.0F;
                if (scpp != nullptr) {
                    cint = scpp[x];
                } else {
                    cint = (0.5625F * (dst1p[x] + dst1n[x])) -
                           (0.0625F * (dst3p[x] + dst3n[x]));
                }

                if (dirc == 0) {
                    tline[x] = cint;
                    continue;
                }

                const int dirt = dmap[x - width];
                const int dirb = dmap[x + width];

                if (std::max(dirc * dirt, dirc * dirb) < 0 ||
                    (dirt == dirb && dirt == 0)) {
                    tline[x] = cint;
                    continue;
                }

                const float it = (dst2p[x + dirc] + dstp[x - dirc]) / 2.0F;
                const float ib = (dstp[x + dirc] + dst2n[x - dirc]) / 2.0F;
                const float vt = std::abs(dst2p[x + dirc] - dst1p[x + dirc]) +
                                 std::abs(dstp[x + dirc] - dst1p[x + dirc]);
                const float vb = std::abs(dst2n[x - dirc] - dst1n[x - dirc]) +
                                 std::abs(dstp[x - dirc] - dst1n[x - dirc]);
                const float vc =
                    std::abs(dstp[x] - dst1p[x]) + std::abs(dstp[x] - dst1n[x]);

                const float d0 = std::abs(it - dst1p[x]);
                const float d1 = std::abs(ib - dst1n[x]);
                const float d2 = std::abs(vt - vc);
                const float d3 = std::abs(vb - vc);

                float mdiff0 = 0.0F;
                if (d->vcheck == 1) {
                    mdiff0 = std::min(d0, d1);
                } else if (d->vcheck == 2) {
                    mdiff0 = (d0 + d1) / 2.0F;
                } else {
                    mdiff0 = std::max(d0, d1);
                }

                float mdiff1 = 0.0F;
                if (d->vcheck == 1) {
                    mdiff1 = std::min(d2, d3);
                } else if (d->vcheck == 2) {
                    mdiff1 = (d2 + d3) / 2.0F;
                } else {
                    mdiff1 = std::max(d2, d3);
                }

                const float a0 = mdiff0 * d->rcpVthresh0;
                const float a1 = mdiff1 * d->rcpVthresh1;
                const float a2 = std::max(
                    (d->vthresh2 - static_cast<float>(std::abs(dirc))) *
                        d->rcpVthresh2,
                    0.0F);
                const float a = std::min(std::max({a0, a1, a2}), 1.0F);

                tline[x] = ((1.0F - a) * dstp[x]) + (a * cint);
            }

            memcpy(dstp, tline, width * sizeof(float));
        }

        srcp += srcStride * 2;
        if (scpp != nullptr) {
            scpp += dstStride * 2;
        }
        dstp += dstStride * 2;
        dmap += width;
    }
}

// NOLINTEND(cppcoreguidelines-init-variables)

#endif // EEDI3_H
