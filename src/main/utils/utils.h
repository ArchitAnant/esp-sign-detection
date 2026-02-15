#ifndef UTILS_H
#define UTILS_H

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

#define LED_PIN GPIO_NUM_2
#define MAX_INFERENCE_DATA_SIZE 786

// Function declarations
void flash_led();
void parse_json(char buf[]);
void turn_led_on();
void turn_led_off();
void print_ram_usage();

// Data structure for inference
typedef struct {
    uint8_t data[MAX_INFERENCE_DATA_SIZE];
    size_t length;
} inference_data_t;

extern QueueHandle_t inference_queue;

#endif 