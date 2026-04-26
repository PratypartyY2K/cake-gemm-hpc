#include "timer.h"
#include <chrono>

double now()
{
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}