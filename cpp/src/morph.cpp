#include "morph.hpp"

namespace omr {

StructElem disk(int radius) {
    int n = 2 * radius + 1;
    StructElem se(n, n);
    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            se.set(y + radius, x + radius, x * x + y * y <= radius * radius);
        }
    }
    return se;
}

StructElem rect(int height, int width) {
    return StructElem(width, height, 1);
}

Mask erode(const Mask& mask, const StructElem& se) {
    int sh = se.h, sw = se.w;
    int cy = sh / 2, cx = sw / 2;
    int h = mask.h, w = mask.w;
    Mask out(w, h, 1);
    for (int dy = 0; dy < sh; ++dy) {
        for (int dx = 0; dx < sw; ++dx) {
            if (!se.at(dy, dx)) continue;
            int oy = dy - cy, ox = dx - cx;
            for (int y = 0; y < h; ++y) {
                int sy = y + oy;
                for (int x = 0; x < w; ++x) {
                    if (!out.at(y, x)) continue;
                    int sx = x + ox;
                    bool v = (sy >= 0 && sy < h && sx >= 0 && sx < w) && mask.at(sy, sx);
                    if (!v) out.set(y, x, false);
                }
            }
        }
    }
    return out;
}

Mask dilate(const Mask& mask, const StructElem& se) {
    int sh = se.h, sw = se.w;
    int cy = sh / 2, cx = sw / 2;
    int h = mask.h, w = mask.w;
    Mask out(w, h, 0);
    for (int dy = 0; dy < sh; ++dy) {
        for (int dx = 0; dx < sw; ++dx) {
            if (!se.at(dy, dx)) continue;
            int oy = dy - cy, ox = dx - cx;
            for (int y = 0; y < h; ++y) {
                int sy = y + oy;
                if (sy < 0 || sy >= h) continue;
                for (int x = 0; x < w; ++x) {
                    if (out.at(y, x)) continue;
                    int sx = x + ox;
                    if (sx >= 0 && sx < w && mask.at(sy, sx)) out.set(y, x, true);
                }
            }
        }
    }
    return out;
}

Mask opening(const Mask& mask, const StructElem& se) { return dilate(erode(mask, se), se); }
Mask closing(const Mask& mask, const StructElem& se) { return erode(dilate(mask, se), se); }

}  // namespace omr
