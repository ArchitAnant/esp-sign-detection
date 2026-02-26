#ifndef POTENTIOMETER_H
#define POTENTIOMETER_H

#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize the potentiometer on GPIO 34 (ADC1 Channel 6).
 * Sets up 12-bit resolution and 12dB attenuation for 3.3V range.
 */
esp_err_t pot_init(void);

/**
 * @brief Get the raw 12-bit reading (0 to 4095).
 */
int pot_get_raw(void);

/**
 * @brief Get a scaled threshold for ML confidence.
 * Maps 0.0V -> 0.50 (Very sensitive)
 * Maps 3.3V -> 0.99 (Very strict)
 */
float pot_get_threshold(void);

#ifdef __cplusplus
}
#endif

#endif // POTENTIOMETER_H