#include "servo_init.h"

uint32_t servo_angle_to_duty(uint32_t angle)
{
    uint32_t pulsewidth_us = MIN_PULSEWIDTH_US +
        ((MAX_PULSEWIDTH_US - MIN_PULSEWIDTH_US) * angle) / MAX_DEGREE;

    uint32_t duty = (pulsewidth_us * (1 << 16)) / (1000000 / SERVO_FREQ_HZ);
    return duty;
}


void servo_init(void)
{
    // Timer config
    ledc_timer_config_t timer_conf = {
        .speed_mode       = SERVO_MODE,
        .duty_resolution  = SERVO_RESOLUTION,
        .timer_num        = SERVO_TIMER,
        .freq_hz          = SERVO_FREQ_HZ,
        .clk_cfg          = LEDC_AUTO_CLK
    };
    ledc_timer_config(&timer_conf);

    // Channel config
    ledc_channel_config_t channel_conf = {
        .gpio_num       = SERVO_GPIO,
        .speed_mode     = SERVO_MODE,
        .channel        = SERVO_CHANNEL,
        .timer_sel      = SERVO_TIMER,
        .duty           = 0,
        .hpoint         = 0
    };
    ledc_channel_config(&channel_conf);
}