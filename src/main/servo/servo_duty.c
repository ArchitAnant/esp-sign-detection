#include "servo_duty.h"
#include "esp_log.h"

void servo_set_angle(uint32_t angle)
{
    ESP_LOGI("SERVO", "CALLED");
    if (angle > MAX_DEGREE) angle = MAX_DEGREE;

    uint32_t duty = servo_angle_to_duty(angle);

    ledc_set_duty(SERVO_MODE, SERVO_CHANNEL, duty);
    ledc_update_duty(SERVO_MODE, SERVO_CHANNEL);
    vTaskDelay(pdMS_TO_TICKS(2000));
}