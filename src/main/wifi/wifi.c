#include "wifi.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "esp_netif.h"
#include "../utils/utils.h"

// header for wifi ssid and password
#include "secret.h"

#define TAG "WIFI"

// Global flag to indicate IP is ready
bool got_ip = false;

static void wifi_event_handler(
    void *arg,
    esp_event_base_t event_base,
    int32_t event_id,
    void *event_data
)
{
    if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP)
    {
        ip_event_got_ip_t *event = (ip_event_got_ip_t*) event_data;

        ESP_LOGI(TAG, "Got IP: " IPSTR, IP2STR(&event->ip_info.ip));
        
        got_ip = true;  // signal main task
        flash_led();
        flash_led();
    }
    else {
        flash_led();
    }
}

void wifi_init_sta(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES ||
        ret == ESP_ERR_NVS_NEW_VERSION_FOUND
    )
    {
        nvs_flash_erase();
        nvs_flash_init();
    }

    esp_netif_init();
    esp_event_loop_create_default();
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    esp_wifi_init(&cfg);

    // Register events
    esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID,
                               &wifi_event_handler, NULL);
    esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP,
                               &wifi_event_handler, NULL);

    // Wi-Fi config
    wifi_config_t wifi_config = {
        .sta = {
            .ssid = WIFI_SSID,
            .password = WIFI_PASS,
        }
    };

    esp_wifi_set_mode(WIFI_MODE_STA);
    esp_wifi_set_config(WIFI_IF_STA, &wifi_config);
    esp_wifi_start();

    esp_wifi_connect();

    ESP_LOGI(TAG, "WiFi init done");
}
