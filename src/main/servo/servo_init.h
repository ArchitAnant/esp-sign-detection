#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/ledc.h"
#include "esp_err.h"


#define SERVO_GPIO        18
#define SERVO_FREQ_HZ     50
#define SERVO_TIMER       LEDC_TIMER_0
#define SERVO_MODE        LEDC_LOW_SPEED_MODE
#define SERVO_CHANNEL     LEDC_CHANNEL_0
#define SERVO_RESOLUTION  LEDC_TIMER_16_BIT

#define MIN_PULSEWIDTH_US 500
#define MAX_PULSEWIDTH_US 2500
#define MAX_DEGREE        180


uint32_t servo_angle_to_duty(uint32_t angle);
void servo_init(void);