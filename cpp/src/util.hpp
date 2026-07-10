#pragma once

#include <cmath>

namespace omr {

// round() de Python 3 : arrondi au plus proche, pair en cas d'égalité.
inline long long py_round(double x) {
    double floor_val = std::floor(x);
    double diff = x - floor_val;
    long long fl = static_cast<long long>(floor_val);
    if (diff < 0.5) return fl;
    if (diff > 0.5) return fl + 1;
    return (fl % 2 == 0) ? fl : fl + 1;
}

// % et // de Python : reste et division de signe celui du diviseur (floor).
inline int py_mod(int a, int b) {
    int r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) r += b;
    return r;
}

inline int py_floordiv(int a, int b) {
    int q = a / b;
    int r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) q -= 1;
    return q;
}

}  // namespace omr
