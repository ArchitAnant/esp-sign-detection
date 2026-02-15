#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include "esp_log.h"
#include "../utils/utils.h"

#define UDP_PORT 5005
#define RX_BUF_SIZE 1024

#define TAG "UDP_RX"

void udp_rx_task(void *pvParameters)
{
    int sock;
    struct sockaddr_in local_addr;
    struct sockaddr_in source_addr;
    socklen_t socklen = sizeof(source_addr);
    uint8_t rx_buffer[RX_BUF_SIZE];
    
    // Create socket
    sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (sock < 0) {
        ESP_LOGE(TAG, "Unable to create socket");
        vTaskDelete(NULL);
        return;
    }
    
    // Bind socket
    local_addr.sin_family = AF_INET;
    local_addr.sin_port = htons(UDP_PORT);
    local_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    
    if (bind(sock, (struct sockaddr*)&local_addr, sizeof(local_addr)) < 0) {
        ESP_LOGE(TAG, "Socket bind failed");
        close(sock);
        vTaskDelete(NULL);
        return;
    }
    
    ESP_LOGI(TAG, "Listening on port %d, queue handle: %p", UDP_PORT, (void*)inference_queue);
    
    // Verify queue is valid
    if (inference_queue == NULL) {
        ESP_LOGE(TAG, "FATAL: inference_queue is NULL! Did init_inference_system() run?");
        close(sock);
        vTaskDelete(NULL);
        return;
    }
    
    while (1) {
        int len = recvfrom(sock, rx_buffer, sizeof(rx_buffer) - 1, 0, 
                          (struct sockaddr *)&source_addr, &socklen);
        
        if (len > 0) {
            ESP_LOGI(TAG, "Received %d bytes from %s:%d", 
                     len,
                     inet_ntoa(source_addr.sin_addr),
                     ntohs(source_addr.sin_port));
            
            // Validate size
            if (len > MAX_INFERENCE_DATA_SIZE) {
                ESP_LOGW(TAG, "Packet too large (%d > %d), dropping", 
                         len, MAX_INFERENCE_DATA_SIZE);
                continue;
            }
            
            // Prepare data
            inference_data_t data_to_send;
            memcpy(data_to_send.data, rx_buffer, len);
            data_to_send.length = len;
            
            // Send to queue
            ESP_LOGI(TAG, "Queueing data, queue handle: %p", (void*)inference_queue);
            
            if (xQueueSend(inference_queue, &data_to_send, pdMS_TO_TICKS(100)) != pdTRUE) {
                ESP_LOGW(TAG, "Queue full or send failed, dropping packet");
            } else {
                ESP_LOGI(TAG, "Data successfully queued");
            }
        }
    }
    
    close(sock);
    vTaskDelete(NULL);
}