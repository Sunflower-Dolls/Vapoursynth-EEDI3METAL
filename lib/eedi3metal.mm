#include <Metal/Metal.h>
#include <VSHelper4.h>
#include <VapourSynth4.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <ranges>
#include <vector>

#include "./EEDI3.h"
#include "EEDI3Metal_shared.h"

#include <atomic>
#include <deque>
#include <mutex>
#include <thread>

using namespace std::literals;

struct ticket_semaphore {
    std::atomic<intptr_t> ticket;
    std::atomic<intptr_t> current;

    void acquire() noexcept {
        intptr_t tk = ticket.fetch_add(1, std::memory_order::acquire);
        while (true) {
            intptr_t curr = current.load(std::memory_order::acquire);
            if (tk <= curr) {
                return;
            }
            current.wait(curr, std::memory_order::relaxed);
        }
    }

    void release() noexcept {
        current.fetch_add(1, std::memory_order::release);
        current.notify_all();
    }
};

struct Metal_Resource {
    id<MTLCommandQueue> commandQueue = nil;
    id<MTLBuffer> paramsBuffer = nil;
    id<MTLBuffer> pbacktBuffer = nil;
    id<MTLBuffer> dmapBuffer = nil;
    id<MTLBuffer> bmaskBuffer = nil;
    id<MTLBuffer> costBuffer = nil;

    ~Metal_Resource() {
        commandQueue = nil;
        paramsBuffer = nil;
        pbacktBuffer = nil;
        dmapBuffer = nil;
        bmaskBuffer = nil;
        costBuffer = nil;
    }

    Metal_Resource() = default;

    Metal_Resource(Metal_Resource &&other) noexcept
        : commandQueue(std::move(other.commandQueue)),
          paramsBuffer(std::move(other.paramsBuffer)),
          pbacktBuffer(std::move(other.pbacktBuffer)),
          dmapBuffer(std::move(other.dmapBuffer)),
          bmaskBuffer(std::move(other.bmaskBuffer)),
          costBuffer(std::move(other.costBuffer)) {
        other.commandQueue = nil;
        other.paramsBuffer = nil;
        other.pbacktBuffer = nil;
        other.dmapBuffer = nil;
        other.bmaskBuffer = nil;
        other.costBuffer = nil;
    }

    Metal_Resource &operator=(Metal_Resource &&other) noexcept {
        if (this != &other) {
            commandQueue = std::move(other.commandQueue);
            paramsBuffer = std::move(other.paramsBuffer);
            pbacktBuffer = std::move(other.pbacktBuffer);
            dmapBuffer = std::move(other.dmapBuffer);
            bmaskBuffer = std::move(other.bmaskBuffer);
            costBuffer = std::move(other.costBuffer);
            other.commandQueue = nil;
            other.paramsBuffer = nil;
            other.pbacktBuffer = nil;
            other.dmapBuffer = nil;
            other.bmaskBuffer = nil;
            other.costBuffer = nil;
        }
        return *this;
    }

    Metal_Resource(const Metal_Resource &) = delete;
    Metal_Resource &operator=(const Metal_Resource &) = delete;
};

struct EEDI3MetalData {
    EEDI3Data d; // CPU data for vCheck and params

    id<MTLDevice> device;
    id<MTLComputePipelineState> calc_costs_pso;
    id<MTLComputePipelineState> viterbi_scan_pso;
    id<MTLComputePipelineState> interpolate_pso;
    id<MTLComputePipelineState> copy_field_pso;
    id<MTLComputePipelineState> dilate_mask_pso;
    id<MTLComputePipelineState> copy_pso;

    ticket_semaphore semaphore;
    std::deque<Metal_Resource> resources;
    std::mutex resources_lock;

    size_t metal_stride;
    int metal_stride_pixels;
};

struct ResourceHolder {
    EEDI3MetalData *d;
    Metal_Resource resource;

    ResourceHolder(EEDI3MetalData *_d) : d(_d) {
        d->semaphore.acquire();
        std::lock_guard<std::mutex> lock(d->resources_lock);
        // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
        resource = std::move(d->resources.front());
        d->resources.pop_front();
    }

    ~ResourceHolder() {
        std::lock_guard<std::mutex> lock(d->resources_lock);
        d->resources.push_back(std::move(resource));
        d->semaphore.release();
    }

    ResourceHolder(const ResourceHolder &) = delete;
    ResourceHolder &operator=(const ResourceHolder &) = delete;
    ResourceHolder(ResourceHolder &&) = delete;
    ResourceHolder &operator=(ResourceHolder &&) = delete;
};

struct MetalCaptureScope {
    MTLCaptureManager *manager = nil;
    bool active = false;

    MetalCaptureScope(id<MTLDevice> device, int n) {
        if (std::getenv("eedi3_METAL_CAPTURE") != nullptr) {
            manager = [MTLCaptureManager sharedCaptureManager];
            if ((manager != nullptr) && !manager.isCapturing) {
                MTLCaptureDescriptor *desc =
                    [[MTLCaptureDescriptor alloc] init];
                desc.captureObject = device;
                desc.destination = MTLCaptureDestinationGPUTraceDocument;

                std::string filename =
                    "eedi3_capture_" + std::to_string(n) + ".gputrace";
                desc.outputURL = [NSURL
                    fileURLWithPath:[NSString
                                        stringWithUTF8String:filename.c_str()]];

                NSError *error = nil;
                if ([manager startCaptureWithDescriptor:desc error:&error]) {
                    active = true;
                    std::cout << "Started Metal Capture for frame " << n
                              << "\n";
                } else {
                    std::cerr << "Failed to start Metal Capture: " <<
                        [[error localizedDescription] UTF8String] << "\n";
                    std::cerr << "Please run with environment variable "
                              << "METAL_CAPTURE_ENABLED=1" << "\n";
                }
            }
        }
    }

    ~MetalCaptureScope() {
        if (active && (manager != nullptr)) {
            [manager stopCapture];
            std::cout << "Stopped Metal Capture" << "\n";
        }
    }

    MetalCaptureScope(const MetalCaptureScope &) = delete;
    MetalCaptureScope &operator=(const MetalCaptureScope &) = delete;
    MetalCaptureScope(MetalCaptureScope &&) = delete;
    MetalCaptureScope &operator=(MetalCaptureScope &&) = delete;
};

static inline void encode_copy_from_vs(id<MTLComputeCommandEncoder> encoder,
                                       id<MTLDevice> device,
                                       const void *src_ptr, size_t src_len,
                                       uint src_stride, id<MTLBuffer> dst_buf,
                                       size_t dst_offset, uint dst_stride,
                                       uint width_bytes, uint height) {
    id<MTLBuffer> src_buf =
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        [device newBufferWithBytesNoCopy:const_cast<void *>(src_ptr)
                                  length:src_len
                                 options:MTLResourceStorageModeShared
                             deallocator:nil];

    [encoder setBuffer:src_buf offset:0 atIndex:0];
    [encoder setBuffer:dst_buf offset:dst_offset atIndex:1];
    [encoder setBytes:&src_stride length:sizeof(uint) atIndex:2];
    [encoder setBytes:&dst_stride length:sizeof(uint) atIndex:3];
    [encoder setBytes:&width_bytes length:sizeof(uint) atIndex:4];

    MTLSize grid = MTLSizeMake(width_bytes, height, 1);
    MTLSize group = MTLSizeMake(std::min((int)width_bytes, 32),
                                std::min((int)height, 32), 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:group];
}

static inline void encode_copy_to_vs(id<MTLComputeCommandEncoder> encoder,
                                     id<MTLDevice> device,
                                     id<MTLBuffer> src_buf, size_t src_offset,
                                     uint src_stride, void *dst_ptr,
                                     size_t dst_len, uint dst_stride,
                                     uint width_bytes, uint height) {
    id<MTLBuffer> dst_buf =
        [device newBufferWithBytesNoCopy:dst_ptr
                                  length:dst_len
                                 options:MTLResourceStorageModeShared
                             deallocator:nil];

    [encoder setBuffer:src_buf offset:src_offset atIndex:0];
    [encoder setBuffer:dst_buf offset:0 atIndex:1];
    [encoder setBytes:&src_stride length:sizeof(uint) atIndex:2];
    [encoder setBytes:&dst_stride length:sizeof(uint) atIndex:3];
    [encoder setBytes:&width_bytes length:sizeof(uint) atIndex:4];

    MTLSize grid = MTLSizeMake(width_bytes, height, 1);
    MTLSize group = MTLSizeMake(std::min((int)width_bytes, 32),
                                std::min((int)height, 32), 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:group];
}

static const VSFrame *VS_CC eedi3GetFrame(int n, int activationReason,
                                          void *instanceData,
                                          [[maybe_unused]] void **frameData,
                                          VSFrameContext *frameCtx,
                                          VSCore *core, const VSAPI *vsapi) {
    auto *metal_d = static_cast<EEDI3MetalData *>(instanceData);
    auto *d = &metal_d->d;

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(d->field > 1 ? n / 2 : n, d->node, frameCtx);

        if (d->vcheck > 0 && (d->sclip != nullptr)) {
            vsapi->requestFrameFilter(n, d->sclip, frameCtx);
        }

        if (d->mclip != nullptr) {
            vsapi->requestFrameFilter(d->field > 1 ? n / 2 : n, d->mclip,
                                      frameCtx);
        }
    } else if (activationReason == arAllFramesReady) {
        ResourceHolder rh(metal_d);
        Metal_Resource &res = rh.resource;

        MetalCaptureScope capture_scope(metal_d->device, n);

        const VSFrame *src =
            vsapi->getFrameFilter(d->field > 1 ? n / 2 : n, d->node, frameCtx);
        const VSFrame *mclip = nullptr;
        if (d->mclip != nullptr) {
            mclip = vsapi->getFrameFilter(d->field > 1 ? n / 2 : n, d->mclip,
                                          frameCtx);
        }

        VSFrame *dst = vsapi->newVideoFrame(&d->vi.format, d->vi.width,
                                            d->vi.height, src, core);

        int field = d->field;
        if (field > 1) {
            field -= 2;
        }

        int err = 0;
        const int fieldBased = vsapi->mapGetIntSaturated(
            vsapi->getFramePropertiesRO(src), "_FieldBased", 0, &err);
        if (fieldBased == 1) {
            field = 0;
        } else if (fieldBased == 2) {
            field = 1;
        }

        int field_n = 0;
        if (d->field > 1) {
            if ((n & 1) != 0) {
                field_n = static_cast<int>(field == 0);
            } else {
                field_n = static_cast<int>(field == 1);
            }
        } else {
            field_n = field;
        }

        for (int plane = 0; plane < d->vi.format.numPlanes; plane++) {
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
            if (!d->process[plane]) {
                const uint8_t *srcp = vsapi->getReadPtr(src, plane);
                uint8_t *dstp = vsapi->getWritePtr(dst, plane);
                const int src_stride =
                    static_cast<int>(vsapi->getStride(src, plane));
                const int dst_stride =
                    static_cast<int>(vsapi->getStride(dst, plane));
                const int height = vsapi->getFrameHeight(src, plane);
                const int row_size = static_cast<int>(
                    vsapi->getFrameWidth(src, plane) * sizeof(float));

                for (int y = 0; y < height; y++) {
                    std::memcpy(dstp + (y * dst_stride),
                                srcp + (y * src_stride), row_size);
                }
                continue;
            }

            const int plane_width = vsapi->getFrameWidth(src, plane);
            const int plane_height = vsapi->getFrameHeight(src, plane);
            int src_stride = static_cast<int>(vsapi->getStride(src, plane));

            // Create Buffers
            size_t plane_size_bytes = metal_d->metal_stride * plane_height;

            id<MTLBuffer> src_buf_metal = [metal_d->device
                newBufferWithLength:plane_size_bytes
                            options:MTLResourceStorageModePrivate];
            id<MTLBuffer> mclip_buf_metal = nil;

            id<MTLCommandBuffer> cmdBuf = [res.commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

            [enc setComputePipelineState:metal_d->copy_pso];

            // Upload Source
            encode_copy_from_vs(enc, metal_d->device,
                                vsapi->getReadPtr(src, plane),
                                src_stride * plane_height, src_stride,
                                src_buf_metal, 0, metal_d->metal_stride,
                                plane_width * sizeof(float), plane_height);

            // Upload Mclip
            if (mclip != nullptr) {
                int mclip_stride =
                    static_cast<int>(vsapi->getStride(mclip, plane));
                mclip_buf_metal = [metal_d->device
                    newBufferWithLength:plane_size_bytes
                                options:MTLResourceStorageModePrivate];

                [enc setComputePipelineState:metal_d->copy_pso];
                encode_copy_from_vs(enc, metal_d->device,
                                    vsapi->getReadPtr(mclip, plane),
                                    mclip_stride * plane_height, mclip_stride,
                                    mclip_buf_metal, 0, metal_d->metal_stride,
                                    plane_width * sizeof(float), plane_height);
            }

            // Create plane-specific destination buffer
            const int dst_width = vsapi->getFrameWidth(dst, plane);
            const int dst_height = vsapi->getFrameHeight(dst, plane);

            size_t dst_size_bytes = metal_d->metal_stride * dst_height;
            id<MTLBuffer> dstBuffer = [metal_d->device
                newBufferWithLength:dst_size_bytes
                            options:MTLResourceStorageModePrivate];

            EEDI3Params params;
            params.width = dst_width;
            params.height = dst_height;
            params.mdis = d->mdis;
            params.tpitch = d->tpitch;
            params.alpha = d->alpha;
            params.beta = d->beta;
            params.gamma = d->gamma;
            params.remainingWeight = d->remainingWeight;
            params.cost3 = d->cost3 ? 1 : 0;
            params.ucubic = d->ucubic ? 1 : 0;
            params.has_mclip = (mclip != nullptr) ? 1 : 0;
            params.nrad = d->nrad;
            params.field = field_n;
            params.dh = d->dh ? 1 : 0;
            params.stride = metal_d->metal_stride_pixels;

            std::memcpy([res.paramsBuffer contents], &params,
                        sizeof(EEDI3Params));

            const int field_height = (dst_height - field_n + 1) / 2;

            [enc setComputePipelineState:metal_d->copy_field_pso];
            [enc setBuffer:src_buf_metal offset:0 atIndex:0];
            [enc setBuffer:dstBuffer offset:0 atIndex:1];
            [enc setBuffer:res.paramsBuffer offset:0 atIndex:2];

            int copy_field_height = (dst_height - (1 - field_n) + 1) / 2;
            [enc dispatchThreads:MTLSizeMake(dst_width, copy_field_height, 1)
                threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];

            if (mclip != nullptr) {
                [enc setComputePipelineState:metal_d->dilate_mask_pso];
                [enc setBuffer:mclip_buf_metal offset:0 atIndex:0];
                [enc setBuffer:res.bmaskBuffer offset:0 atIndex:1];
                [enc setBuffer:res.paramsBuffer offset:0 atIndex:2];
                // 1D dispatch, 1 threadgroup per line
                [enc dispatchThreadgroups:MTLSizeMake(field_height, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            }

            [enc setComputePipelineState:metal_d->calc_costs_pso];
            [enc setBuffer:src_buf_metal offset:0 atIndex:0];
            [enc setBuffer:res.costBuffer offset:0 atIndex:1];
            [enc setBuffer:res.paramsBuffer offset:0 atIndex:2];
            [enc setBuffer:res.bmaskBuffer offset:0 atIndex:3];

            [enc dispatchThreads:MTLSizeMake(dst_width, field_height, 1)
                threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];

            [enc setComputePipelineState:metal_d->viterbi_scan_pso];
            [enc setBuffer:res.costBuffer offset:0 atIndex:0];
            [enc setBuffer:res.dmapBuffer offset:0 atIndex:1];
            [enc setBuffer:res.pbacktBuffer offset:0 atIndex:2];
            [enc setBuffer:res.paramsBuffer offset:0 atIndex:3];
            [enc setBuffer:res.bmaskBuffer offset:0 atIndex:4];

            [enc dispatchThreadgroups:MTLSizeMake(1, field_height, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

            [enc setComputePipelineState:metal_d->interpolate_pso];
            [enc setBuffer:src_buf_metal offset:0 atIndex:0];
            [enc setBuffer:dstBuffer offset:0 atIndex:1];

            [enc setBuffer:res.dmapBuffer offset:0 atIndex:2];
            [enc setBuffer:res.paramsBuffer offset:0 atIndex:3];
            [enc setBuffer:res.bmaskBuffer offset:0 atIndex:4];

            [enc dispatchThreads:MTLSizeMake(dst_width, field_height, 1)
                threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];

            // Download Destination
            [enc setComputePipelineState:metal_d->copy_pso];
            int dst_stride = static_cast<int>(vsapi->getStride(dst, plane));
            encode_copy_to_vs(
                enc, metal_d->device, dstBuffer, 0, metal_d->metal_stride,
                vsapi->getWritePtr(dst, plane), dst_stride * dst_height,
                dst_stride, dst_width * sizeof(float), dst_height);

            [enc endEncoding];
            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];

            auto *dst_ptr =
                reinterpret_cast<float *>(vsapi->getWritePtr(dst, plane));
            const int dst_stride_bytes =
                static_cast<int>(vsapi->getStride(dst, plane));
            const int dst_stride_pixels =
                static_cast<int>(dst_stride_bytes / sizeof(float));

            int *dmap_ptr = static_cast<int *>([res.dmapBuffer contents]);

            if (d->vcheck > 0) {
                VSFrame *pad = vsapi->newVideoFrame(
                    &d->padFormat, dst_width + (MARGIN_H * 2),
                    dst_height + (MARGIN_V * 2), nullptr, core);

                copyPad(src, pad, plane, d->dh, 1 - field_n, vsapi);

                const auto *src_ptr_base =
                    reinterpret_cast<const float *>(vsapi->getReadPtr(pad, 0));
                const int pad_stride_pixels =
                    static_cast<int>(vsapi->getStride(pad, 0) / sizeof(float));

                const int padded_height = vsapi->getFrameHeight(pad, 0);

                const float *aligned_src_ptr =
                    src_ptr_base + (pad_stride_pixels * (MARGIN_V + field_n));

                float *aligned_dst_ptr =
                    dst_ptr + (dst_stride_pixels * field_n);

                const int *aligned_dmap_ptr = dmap_ptr;

                const VSFrame *scp = nullptr;
                if (d->sclip != nullptr) {
                    scp = vsapi->getFrameFilter(n, d->sclip, frameCtx);
                }

                const float *aligned_scpp = nullptr;
                if (scp != nullptr) {
                    aligned_scpp = reinterpret_cast<const float *>(
                                       vsapi->getReadPtr(scp, plane)) +
                                   (dst_stride_pixels * field_n);
                }

                std::vector<float> tline(dst_width);

                vCheck(aligned_src_ptr, aligned_scpp, aligned_dst_ptr,
                       aligned_dmap_ptr, tline.data(), field_n, dst_width,
                       padded_height, pad_stride_pixels, dst_stride_pixels, d);

                vsapi->freeFrame(pad);
                if (scp != nullptr) {
                    vsapi->freeFrame(scp);
                }
            }
        }

        VSMap *props = vsapi->getFramePropertiesRW(dst);
        vsapi->mapSetInt(props, "_FieldBased", 0, maReplace);

        if (d->field > 1) {
            int errNum = 0;
            int errDen = 0;
            int64_t durationNum =
                vsapi->mapGetInt(props, "_DurationNum", 0, &errNum);
            int64_t durationDen =
                vsapi->mapGetInt(props, "_DurationDen", 0, &errDen);
            if ((errNum == 0) && (errDen == 0)) {
                vsh::muldivRational(&durationNum, &durationDen, 1, 2);
                vsapi->mapSetInt(props, "_DurationNum", durationNum, maReplace);
                vsapi->mapSetInt(props, "_DurationDen", durationDen, maReplace);
            }
        }

        vsapi->freeFrame(src);
        if (mclip != nullptr) {
            vsapi->freeFrame(mclip);
        }

        return dst;
    }

    return nullptr;
}

static void VS_CC eedi3Free(void *instanceData, [[maybe_unused]] VSCore *core,
                            const VSAPI *vsapi) {
    auto metal_d = std::unique_ptr<EEDI3MetalData>(
        static_cast<EEDI3MetalData *>(instanceData));
    auto *d = &metal_d->d;

    @autoreleasepool {
        metal_d->resources.clear();
        metal_d->device = nil;
    }

    vsapi->freeNode(d->node);
    vsapi->freeNode(d->sclip);
    vsapi->freeNode(d->mclip);
}

static void VS_CC eedi3Create(const VSMap *in, VSMap *out,
                              [[maybe_unused]] void *userData, VSCore *core,
                              const VSAPI *vsapi) {
    auto metal_d = std::make_unique<EEDI3MetalData>();
    auto *d = &metal_d->d;
    int err = 0;

    try {
        metal_d->device = MTLCreateSystemDefaultDevice();
        if (metal_d->device == nullptr) {
            throw "Failed to create Metal device"s;
        }

        NSError *error = nil;
        // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
        const char metal_source_chars[] = {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"
#embed "kernel.metal"
#pragma clang diagnostic pop
        };
        NSString *metalSource = [[NSString alloc]
            initWithBytes:static_cast<const void *>(metal_source_chars)
                   length:sizeof(metal_source_chars)
                 encoding:NSUTF8StringEncoding];

        id<MTLLibrary> library =
            [metal_d->device newLibraryWithSource:metalSource
                                          options:nil
                                            error:&error];
        if (library == nullptr) {
            throw "Failed to compile Metal library: "s +
                std::string([error.localizedDescription UTF8String]);
        }

        id<MTLFunction> calc_costs_func =
            [library newFunctionWithName:@"calc_costs"];
        if (calc_costs_func == nullptr) {
            throw "Failed to find function 'calc_costs'"s;
        }

        metal_d->calc_costs_pso =
            [metal_d->device newComputePipelineStateWithFunction:calc_costs_func
                                                           error:&error];
        if (metal_d->calc_costs_pso == nullptr) {
            throw "Failed to create Calc Costs PSO: "s +
                std::string([error.localizedDescription UTF8String]);
        }

        id<MTLFunction> viterbi_scan_func =
            [library newFunctionWithName:@"viterbi_scan"];
        if (viterbi_scan_func == nullptr) {
            throw "Failed to find function 'viterbi_scan'"s;
        }

        metal_d->viterbi_scan_pso = [metal_d->device
            newComputePipelineStateWithFunction:viterbi_scan_func
                                          error:&error];
        if (metal_d->viterbi_scan_pso == nullptr) {
            throw "Failed to create Viterbi Scan PSO: "s +
                std::string([error.localizedDescription UTF8String]);
        }

        id<MTLFunction> interpolate_func =
            [library newFunctionWithName:@"interpolate"];
        if (interpolate_func == nullptr) {
            throw "Failed to find function 'interpolate'"s;
        }

        metal_d->interpolate_pso = [metal_d->device
            newComputePipelineStateWithFunction:interpolate_func
                                          error:&error];
        if (metal_d->interpolate_pso == nullptr) {
            throw "Failed to create Interpolate PSO: "s +
                std::string([error.localizedDescription UTF8String]);
        }

        id<MTLFunction> copy_func = [library newFunctionWithName:@"copy_field"];
        if (copy_func == nullptr) {
            throw "Failed to find function 'copy_field'"s;
        }
        metal_d->copy_field_pso =
            [metal_d->device newComputePipelineStateWithFunction:copy_func
                                                           error:&error];
        if (metal_d->copy_field_pso == nullptr) {
            throw "Failed to create Copy Field PSO: "s +
                std::string([error.localizedDescription UTF8String]);
        }

        id<MTLFunction> dilate_func =
            [library newFunctionWithName:@"dilate_mask"];
        if (dilate_func == nullptr) {
            throw "Failed to find function 'dilate_mask'"s;
        }
        metal_d->dilate_mask_pso =
            [metal_d->device newComputePipelineStateWithFunction:dilate_func
                                                           error:&error];
        if (metal_d->dilate_mask_pso == nullptr) {
            throw "Failed to create Dilate Mask PSO: "s +
                std::string([error.localizedDescription UTF8String]);
        }

        id<MTLFunction> memcpy_func =
            [library newFunctionWithName:@"copy_buffer"];
        if (memcpy_func == nullptr) {
            throw "Failed to find function 'copy_buffer'"s;
        }
        metal_d->copy_pso =
            [metal_d->device newComputePipelineStateWithFunction:memcpy_func
                                                           error:&error];
        if (metal_d->copy_pso == nullptr) {
            throw "Failed to create Memcpy PSO: "s +
                std::string([error.localizedDescription UTF8String]);
        }

        d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
        d->sclip = vsapi->mapGetNode(in, "sclip", 0, &err);
        d->mclip = vsapi->mapGetNode(in, "mclip", 0, &err);
        d->vi = *vsapi->getVideoInfo(d->node);

        if (!vsh::isConstantVideoFormat(&d->vi) ||
            d->vi.format.sampleType != stFloat ||
            d->vi.format.bitsPerSample != 32) {
            throw "only constant format 32 bit float input supported for Metal"s;
        }

        vsapi->queryVideoFormat(&d->padFormat, cfGray, d->vi.format.sampleType,
                                d->vi.format.bitsPerSample, 0, 0, core);

        d->field = vsapi->mapGetIntSaturated(in, "field", 0, nullptr);
        d->dh = !(vsapi->mapGetInt(in, "dh", 0, &err) == 0);

        const int m = vsapi->mapNumElements(in, "planes");
        std::ranges::fill(d->process, (m <= 0));

        for (int i = 0; i < m; i++) {
            const int n = vsapi->mapGetIntSaturated(in, "planes", i, nullptr);
            if (n < 0 || n >= d->vi.format.numPlanes) {
                throw "plane index out of range"s;
            }
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
            if (d->process[n]) {
                throw "plane specified twice"s;
            }
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
            d->process[n] = true;
        }

        d->alpha = vsapi->mapGetFloatSaturated(in, "alpha", 0, &err);
        if (err != 0) {
            d->alpha = 0.2F;
        }

        d->beta = vsapi->mapGetFloatSaturated(in, "beta", 0, &err);
        if (err != 0) {
            d->beta = 0.25F;
        }

        d->gamma = vsapi->mapGetFloatSaturated(in, "gamma", 0, &err);
        if (err != 0) {
            d->gamma = 20.0F;
        }

        d->nrad = vsapi->mapGetIntSaturated(in, "nrad", 0, &err);
        if (err != 0) {
            d->nrad = 2;
        }

        d->mdis = vsapi->mapGetIntSaturated(in, "mdis", 0, &err);
        if (err != 0) {
            d->mdis = 20;
        }

        d->ucubic = !(vsapi->mapGetInt(in, "ucubic", 0, &err) == 0);
        if (err != 0) {
            d->ucubic = true;
        }

        d->cost3 = !(vsapi->mapGetInt(in, "cost3", 0, &err) == 0);
        if (err != 0) {
            d->cost3 = true;
        }

        d->vcheck = vsapi->mapGetIntSaturated(in, "vcheck", 0, &err);
        if (err != 0) {
            d->vcheck = 2;
        }

        float vthresh0 = vsapi->mapGetFloatSaturated(in, "vthresh0", 0, &err);
        if (err != 0) {
            vthresh0 = 32.0F;
        }

        float vthresh1 = vsapi->mapGetFloatSaturated(in, "vthresh1", 0, &err);
        if (err != 0) {
            vthresh1 = 64.0F;
        }

        d->vthresh2 = vsapi->mapGetFloatSaturated(in, "vthresh2", 0, &err);
        if (err != 0) {
            d->vthresh2 = 4.0F;
        }

        if (d->field < 0 || d->field > 3) {
            throw "field must be 0, 1, 2, or 3"s;
        }
        if (!d->dh) {
            const auto *frame = vsapi->getFrame(0, d->node, nullptr, 0);

            for (int plane = 0; plane < d->vi.format.numPlanes; plane++) {
                // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
                if (d->process[plane] &&
                    ((vsapi->getFrameHeight(frame, plane) & 1) != 0)) {
                    vsapi->freeFrame(frame);
                    throw "plane's height must be mod 2 when dh=False"s;
                }
            }

            vsapi->freeFrame(frame);
        }
        if (d->dh && d->field > 1) {
            throw "field must be 0 or 1 when dh=True"s;
        }
        if (d->alpha < 0.0F || d->alpha > 1.0F) {
            throw "alpha must be between 0.0 and 1.0"s;
        }
        if (d->beta < 0.0F || d->beta > 1.0F) {
            throw "beta must be between 0.0 and 1.0"s;
        }
        if (d->alpha + d->beta > 1.0F) {
            throw "alpha+beta must be between 0.0 and 1.0"s;
        }
        if (d->gamma < 0.0F) {
            throw "gamma must be >= 0.0"s;
        }
        if (d->nrad < 0 || d->nrad > 3) {
            throw "nrad must be between 0 and 3"s;
        }
        if (d->mdis < 1 || d->mdis > 40) {
            throw "mdis must be between 1 and 40"s;
        }
        if (d->vcheck < 0 || d->vcheck > 3) {
            throw "vcheck must be 0, 1, 2, or 3"s;
        }

        if (d->vcheck > 0 &&
            (vthresh0 <= 0.0F || vthresh1 <= 0.0F || d->vthresh2 <= 0.0F)) {
            throw "vthresh0, vthresh1 and vthresh2 must be greater than 0.0"s;
        }

        if (d->mclip != nullptr) {
            if (!vsh::isSameVideoInfo(vsapi->getVideoInfo(d->mclip), &d->vi)) {
                throw "mclip's format and dimensions don't match"s;
            }

            if (vsapi->getVideoInfo(d->mclip)->numFrames != d->vi.numFrames) {
                throw "mclip's number of frames doesn't match"s;
            }
        }

        if (d->vcheck > 0 && (d->sclip != nullptr)) {
            if (!vsh::isSameVideoInfo(vsapi->getVideoInfo(d->sclip), &d->vi)) {
                throw "sclip's format and dimensions don't match"s;
            }

            if (vsapi->getVideoInfo(d->sclip)->numFrames != d->vi.numFrames) {
                throw "sclip's number of frames doesn't match"s;
            }
        }

        d->remainingWeight = 1.0F - d->alpha - d->beta;

        if (d->cost3) {
            d->alpha /= 3.0F;
        }

        d->beta /= 255.0F;
        d->gamma /= 255.0F;
        vthresh0 /= 255.0F;
        vthresh1 /= 255.0F;

        d->tpitch = (d->mdis * 2) + 1;
        d->rcpVthresh0 = 1.0F / vthresh0;
        d->rcpVthresh1 = 1.0F / vthresh1;
        d->rcpVthresh2 = 1.0F / d->vthresh2;

        if (d->field > 1) {
            d->vi.numFrames *= 2;
            vsh::muldivRational(&d->vi.fpsNum, &d->vi.fpsDen, 2, 1);
        }
        if (d->dh) {
            d->vi.height *= 2;
        }

        metal_d->metal_stride =
            ((size_t)d->vi.width * sizeof(float) + 63) & ~63;
        metal_d->metal_stride_pixels =
            static_cast<int>(metal_d->metal_stride / sizeof(float));

        int num_threads = 8;
        metal_d->semaphore.current.store(num_threads - 1,
                                         std::memory_order::relaxed);

        int max_width = d->vi.width;
        int max_height = d->vi.height;
        int max_field_height = (max_height + 1) / 2;

        for (int i = 0; i < num_threads; ++i) {
            Metal_Resource res;
            res.commandQueue = [metal_d->device newCommandQueue];
            if (res.commandQueue == nil) {
                throw "Failed to create command queue"s;
            }

            res.paramsBuffer = [metal_d->device
                newBufferWithLength:sizeof(EEDI3Params)
                            options:MTLResourceStorageModeShared];

            size_t pbackt_size = static_cast<size_t>(max_field_height) *
                                 max_width * d->tpitch * sizeof(int8_t);
            res.pbacktBuffer = [metal_d->device
                newBufferWithLength:pbackt_size
                            options:MTLResourceStorageModePrivate];

            size_t dmap_size =
                static_cast<size_t>(max_width) * max_field_height * sizeof(int);
            res.dmapBuffer = [metal_d->device
                newBufferWithLength:dmap_size
                            options:MTLResourceStorageModeShared];

            if (d->mclip != nullptr) {
                size_t bmask_size = static_cast<size_t>(max_width) *
                                    max_field_height * sizeof(bool);
                res.bmaskBuffer = [metal_d->device
                    newBufferWithLength:bmask_size
                                options:MTLResourceStorageModePrivate];
            }

            size_t cost_size = static_cast<size_t>(max_width) *
                               max_field_height * d->tpitch * sizeof(float);
            res.costBuffer = [metal_d->device
                newBufferWithLength:cost_size
                            options:MTLResourceStorageModePrivate];

            metal_d->resources.push_back(std::move(res));
        }

    } catch (const std::string &error) {
        vsapi->mapSetError(out, ("EEDI3Metal: " + error).c_str());
        vsapi->freeNode(d->node);
        vsapi->freeNode(d->sclip);
        vsapi->freeNode(d->mclip);
        return;
    }

    std::vector<VSFilterDependency> deps = {
        {d->node, d->field > 1 ? rpGeneral : rpStrictSpatial}};
    if (d->sclip != nullptr) {
        deps.push_back({d->sclip, rpStrictSpatial});
    }
    if (d->mclip != nullptr) {
        deps.push_back({d->mclip, d->field > 1 ? rpGeneral : rpStrictSpatial});
    }

    vsapi->createVideoFilter(
        out, "EEDI3Metal", &d->vi, eedi3GetFrame, eedi3Free, fmParallel,
        deps.data(), static_cast<int>(deps.size()), metal_d.release(), core);
}

VS_EXTERNAL_API(void)
VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
    vspapi->configPlugin("com.Sunflower-Dolls.eedi3metal", "eedi3metal",
                         "EEDI3 Metal Port", VS_MAKE_VERSION(1, 0),
                         VAPOURSYNTH_API_VERSION, 0, plugin);
    vspapi->registerFunction("EEDI3",
                             "clip:vnode;"
                             "field:int;"
                             "dh:int:opt;"
                             "planes:int[]:opt;"
                             "alpha:float:opt;"
                             "beta:float:opt;"
                             "gamma:float:opt;"
                             "nrad:int:opt;"
                             "mdis:int:opt;"
                             "hp:int:opt;"
                             "ucubic:int:opt;"
                             "cost3:int:opt;"
                             "vcheck:int:opt;"
                             "vthresh0:float:opt;"
                             "vthresh1:float:opt;"
                             "vthresh2:float:opt;"
                             "sclip:vnode:opt;"
                             "mclip:vnode:opt;",
                             "clip:vnode;", eedi3Create, nullptr, plugin);
}
