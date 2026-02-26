#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include "esp_log.h"
#include "../utils/utils.h"
#include "../ml/model_bridge.h"
#include "../servo/servo_duty.h"
#include "../poten/potentio.h"

#define TAG "TAG"
#define INIT_TAG "INFERENCE"
#define SERVO_TAG "SERVO"

void inference_task(void *pvParameters); 
QueueHandle_t inference_queue = NULL;

void init_inference_system(void) {
    inference_queue = xQueueCreate(2, sizeof(inference_data_t));
    if (inference_queue == NULL) {
        ESP_LOGE("APP", "Failed to create inference queue");
        return;
    }
    
    ESP_LOGI(TAG, "Inference queue created at: %p", (void*)inference_queue);
    
    BaseType_t result = xTaskCreatePinnedToCore(
        inference_task,
        "inference",
        16*1024,
        NULL,
        4,
        NULL,
        0
    );
    
    if (result != pdPASS) {
        ESP_LOGE(TAG, "Failed to create inference task");
    }
}

void inference_task(void *pvParameters) {
    inference_data_t received_data;
    
    ESP_LOGI(INIT_TAG, "Task started on core %d", xPortGetCoreID());
    int model_ret;
    
    while (1) {
        if (xQueueReceive(inference_queue, &received_data, portMAX_DELAY) == pdTRUE) {
            ESP_LOGI(INIT_TAG, "THRESHOLD : %f", pot_get_threshold());

            model_ret = -1;
            // ESP_LOGI(INIT_TAG, "Received %d bytes", received_data.length);
            
            if (received_data.length == 786) {
                // ESP_LOGI(INIT_TAG, "Skipping 2-byte header");
                model_ret = gesture_model_predict(received_data.data + 2);  // Skip first 2 bytes
            }
            else if (received_data.length == 784) {
                // ESP_LOGI(INIT_TAG, "No header, using raw data");
                model_ret = gesture_model_predict(received_data.data);
            }
            else {
                // ESP_LOGW(INIT_TAG, "Invalid data length: %d (expected 784 or 786)", 
                        //  received_data.length);
                continue;
            }
            
            ESP_LOGI(INIT_TAG, "Prediction complete");

            // servo action
            switch (model_ret)
            {
                case 0:
                    ESP_LOGI(SERVO_TAG, "0 degree");
                    servo_set_angle(0,2000);
                    break;
                case 1:
                    ESP_LOGI(SERVO_TAG, "90 degree");
                    servo_set_angle(90,2000);
                    break;
                case 2:
                    ESP_LOGI(SERVO_TAG, "180 degree");
                    servo_set_angle(180,2000);
                    break;
                case 3:
                    ESP_LOGI(SERVO_TAG, "0->180 degree");
                    servo_set_angle(0,1000);
                    servo_set_angle(180,1000);
                    break;
                default:
                    ESP_LOGI(SERVO_TAG, "null degree");
                    break;
            }
        }
    }
}