#include "llaisys/models/qwen2.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>


struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;      // 保存模型元数据
    LlaisysQwen2Weights weights; // 保存权重指针


};


extern "C" {

    // --- 创建模型 ---
    __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
        const LlaisysQwen2Meta *meta, 
        llaisysDeviceType_t device, 
        int *device_ids, 
        int ndevice
    ) {
        printf("[C++ Backend] Creating Qwen2 Model...\n");
        printf("  - Layers: %zu\n", meta->nlayer);
        printf("  - Hidden: %zu\n", meta->hs);
        fflush(stdout);

        
        // 1. 申请模型结构体内存
        auto model = new LlaisysQwen2Model();
        
        // TODO()

        return model;
    }




    // --- 销毁模型 ---
    __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
        if (!model) return;
        
        printf("[C++ Backend] Destroying Qwen2 Model...\n");

        // TODO()

        delete model;
    }




    // --- 获取权重接口 ---
    __export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model) {
        if (!model) return nullptr;
        return &model->weights;
    }




    // --- 推理接口 (Stub) ---
    __export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken) {
        printf("[C++ Backend] Inference triggered on %zu tokens.\n", ntoken);

        
        return 0;
    }

} // extern "C"