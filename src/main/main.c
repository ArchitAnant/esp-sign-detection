#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_err.h"
#include "wifi/wifi.h"
#include "server/server.h"
#include "ml/model_bridge.h"
#include "ml/inference.h"
#include "utils/utils.h"
#include "servo/servo_duty.h"

#define TAG "APP"

void app_main(void)
{   
    ESP_LOGI(TAG, "ML Model init");

    // ML init
    gesture_model_init();
    ESP_LOGI(TAG, "ML Model init Completed!");

    init_inference_system();
    print_ram_usage();

    // Wifi init
    wifi_init_sta();
    ESP_LOGI(TAG, "ESP32 IP acquired");

    /*
    servo init.
    we do a quick 90 flick to check if the servo is alive and 
    working!
    */
    servo_init();
    servo_set_angle(0);
    servo_set_angle(90);
    servo_set_angle(0);
    print_ram_usage();

    
    ESP_LOGI(TAG, "Starting Server");
    // Start TCP server task
    xTaskCreatePinnedToCore(
        udp_rx_task,
        "server_task",
        16*1024,
        NULL,
        5,
        NULL,
        0
    );
}
