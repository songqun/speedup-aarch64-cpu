#include <sys/time.h>
#include <stdlib.h>

double get_current_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec*1000.0 + tv.tv_usec/1000.0);
}
