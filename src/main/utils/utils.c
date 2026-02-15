#include "driver/gpio.h"
#include "utils.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_heap_caps.h"


void flash_led(){

    gpio_reset_pin(LED_PIN);
    gpio_set_direction(LED_PIN, GPIO_MODE_OUTPUT);
    
    gpio_set_level(LED_PIN,1);
    vTaskDelay(pdMS_TO_TICKS(200));
    gpio_set_level(LED_PIN,0);
    vTaskDelay(pdMS_TO_TICKS(200));

}

void turn_led_on(){

    gpio_reset_pin(LED_PIN);
    gpio_set_direction(LED_PIN, GPIO_MODE_OUTPUT);
    
    gpio_set_level(LED_PIN, 1);
    vTaskDelay(pdMS_TO_TICKS(200));
}

void turn_led_off(){
    
    gpio_reset_pin(LED_PIN);
    gpio_set_direction(LED_PIN, GPIO_MODE_OUTPUT);

    gpio_set_level(LED_PIN, 0);
    vTaskDelay(pdMS_TO_TICKS(200));
}



void print_ram_usage() {
    // 1. Total Free RAM (Internal SRAM)
    uint32_t free_ram = heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    
    // 2. Largest contiguous block (Crucial for Tensor Arena!)
    // Even if you have 100KB free, if it's split into small pieces, 
    // you cannot allocate a 90KB arena.
    uint32_t largest_block = heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    
    // 3. The "Low Water Mark" 
    // This shows the minimum amount of RAM that has been free since the ESP32 started.
    // If this gets close to 0, your ESP32 is about to crash.
    uint32_t min_free = heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

    printf("\n--- RAM USAGE REPORT ---\n");
    printf("Total Free RAM:      %6.2f KB\n", (float)free_ram / 1024.0f);
    printf("Largest Free Block:  %6.2f KB\n", (float)largest_block / 1024.0f);
    printf("Min Free Ever (LWM): %6.2f KB\n", (float)min_free / 1024.0f);
    printf("------------------------\n\n");
}