#ifndef MODEL_BRIDGE_H
#define MODEL_BRIDGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
    #endif

    void gesture_model_init();

    int gesture_model_predict(uint8_t* img_data);

    #ifdef __cplusplus
}
#endif
#endif