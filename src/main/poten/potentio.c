#include "potentio.h"
#include "esp_adc/adc_oneshot.h"
#include "esp_log.h"

static const char *TAG = "POTENTIOMETER";

// Handle for the ADC unit
static adc_oneshot_unit_handle_t adc1_handle = NULL;
static bool is_initialized = false;

// GPIO 34 is ADC1_CHANNEL_6
#define POT_CHANNEL ADC_CHANNEL_6

esp_err_t pot_init(void) {
    if (is_initialized) return ESP_OK;

    // 1. Configure ADC Unit 1
    adc_oneshot_unit_init_cfg_t init_config1 = {
        .unit_id = ADC_UNIT_1,
        .ulp_mode = ADC_ULP_MODE_DISABLE,
    };
    ESP_ERROR_CHECK(adc_oneshot_new_unit(&init_config1, &adc1_handle));

    // 2. Configure the specific channel (GPIO 34)
    adc_oneshot_chan_cfg_t config = {
        .bitwidth = ADC_BITWIDTH_DEFAULT, // 12-bit
        .atten = ADC_ATTEN_DB_12,         // For 0V - 3.3V range
    };
    ESP_ERROR_CHECK(adc_oneshot_config_channel(adc1_handle, POT_CHANNEL, &config));

    is_initialized = true;
    ESP_LOGI(TAG, "Potentiometer initialized on GPIO 34");
    return ESP_OK;
}

int pot_get_raw(void) {
    if (!is_initialized) return 0;
    
    int raw_val = 0;
    // Read the voltage
    ESP_ERROR_CHECK(adc_oneshot_read(adc1_handle, POT_CHANNEL, &raw_val));
    return raw_val;
}

float pot_get_threshold(void) {
    int raw = pot_get_raw();
    
    /* 
     * Mapping math:
     * We want the knob to control sensitivity.
     * Fully Counter-Clockwise (0V / 0 raw) -> 0.50 threshold
     * Fully Clockwise (3.3V / 4095 raw)    -> 0.99 threshold
     */
    float normalized = (float)raw / 4095.0f;
    float threshold = 0.50f + (normalized * 0.49f);
    
    return threshold;
}