#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h" // Change this line
#include "model_data_64_ftf.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <math.h>
#include <algorithm>

#define TAG "ML"
#define NUM_CLASSES 3

constexpr int kTensorArenaSize = 120 * 1024;
static uint8_t* aligned_arena = nullptr;

static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input = nullptr;
static TfLiteTensor* output = nullptr;

static tflite::MicroMutableOpResolver<15> resolver;

extern "C" void gesture_model_init() {
    ESP_LOGI(TAG, "=== MODEL INIT START ===");
    
    if (aligned_arena == nullptr) {
        uint8_t* raw_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize + 16, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
        
        if (raw_arena == nullptr) {
            ESP_LOGE(TAG, "Failed to allocate arena! Free heap: %lu", (unsigned long)xPortGetFreeHeapSize());
            return;
        }
        

        aligned_arena = (uint8_t*)(((uintptr_t)raw_arena + 15) & ~15);
        ESP_LOGI(TAG, "Arena allocated. Raw: %p, Aligned: %p", raw_arena, aligned_arena);
    }
    
    model = tflite::GetModel(gesture_model_esp32_2_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema mismatch!");
        return;
    }

    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected();
    resolver.AddMaxPool2D();
    resolver.AddAveragePool2D();
    resolver.AddReshape();
    resolver.AddAdd();
    resolver.AddRelu6();
    resolver.AddTranspose();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddMean();
    resolver.AddSoftmax();
    resolver.AddMul();
    resolver.AddPad();
    

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, aligned_arena, kTensorArenaSize);
    
    interpreter = &static_interpreter;

    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed! Status: %d", allocate_status);
        return;
    }
    
    input = interpreter->input(0);
    output = interpreter->output(0);

    if (input && output) {
        ESP_LOGI(TAG, "=== INIT SUCCESS ===");
        ESP_LOGI(TAG, "Arena used: %zu bytes", interpreter->arena_used_bytes());
    } else {
        ESP_LOGE(TAG, "=== INIT FAILED: Tensors not found ===");
    }
}

extern "C" int gesture_model_predict(uint8_t* image_data) {
    if (!interpreter || !input) return -1;

    float sum = 0.0f;
    for (int i = 0; i < 784; i++) sum += (float)image_data[i];
    float mean = sum / 784.0f;

    float variance = 0.0f;
    for (int i = 0; i < 784; i++) {
        float diff = (float)image_data[i] - mean;
        variance += diff * diff;
    }

    float std = sqrtf(variance / 784.0f) + 1e-5f;

    float in_scale = input->params.scale;
    int in_zero = input->params.zero_point;

    for (int i = 0; i < 784; i++) {

        float standardized = ((float)image_data[i] - mean) / std;
        
        int32_t quantized = (int32_t)roundf(standardized / in_scale + in_zero);
        
        if (quantized > 127) quantized = 127;
        if (quantized < -128) quantized = -128;
        
        input->data.int8[i] = (int8_t)quantized;
    }

    if (interpreter->Invoke() != kTfLiteOk) return -1;

    const float TEMP = 0.6364f; // Temperature scaling for model calibration 
    float out_scale = output->params.scale;
    int out_zero = output->params.zero_point;
    
    float logits[NUM_CLASSES];
    float max_logit = -1e9f;

    for (int i = 0; i < NUM_CLASSES; i++) {
        logits[i] = (output->data.int8[i] - out_zero) * out_scale;
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    float sum_exp = 0.0f;
    float probs[NUM_CLASSES];
    for (int i = 0; i < NUM_CLASSES; i++) {
        probs[i] = expf((logits[i] - max_logit) / TEMP);
        sum_exp += probs[i];
    }

    int best_class = 0;
    float max_prob = 0.0f;
    for (int i = 0; i < NUM_CLASSES; i++) {
        probs[i] /= sum_exp;
        if (probs[i] > max_prob) {
            max_prob = probs[i];
            best_class = i;
        }
    }

    ESP_LOGI(TAG, "Percentages: [A: %.1f%%, Q: %.1f%%, T: %.1f%%]", 
             probs[0]*100, probs[1]*100, probs[2]*100);

    return best_class;
}