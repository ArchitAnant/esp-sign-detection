#ifndef WIFI_H
#define WIFI_H

#include <stdbool.h>

void wifi_init_sta(void);

// Flag to indicate IP is ready
extern bool got_ip;

#endif
