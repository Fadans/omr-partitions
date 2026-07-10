#include "segment.hpp"

#include <algorithm>
#include <limits>
#include <unordered_map>

namespace omr {

namespace {

struct UnionFind {
    std::vector<int> parent{0};

    int make() {
        parent.push_back(static_cast<int>(parent.size()));
        return static_cast<int>(parent.size()) - 1;
    }

    int find(int a) {
        int root = a;
        while (parent[root] != root) root = parent[root];
        while (parent[a] != root) {
            int next = parent[a];
            parent[a] = root;
            a = next;
        }
        return root;
    }

    void unite(int a, int b) {
        int ra = find(a), rb = find(b);
        if (ra != rb) parent[std::max(ra, rb)] = std::min(ra, rb);
    }
};

}  // namespace

std::vector<int32_t> label_components(const Mask& mask) {
    int h = mask.h, w = mask.w;
    std::vector<int32_t> labels(static_cast<size_t>(h) * w, 0);
    auto lab_at = [&](int y, int x) -> int32_t& { return labels[static_cast<size_t>(y) * w + x]; };

    UnionFind uf;
    int next_label = 1;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (!mask.at(y, x)) continue;
            int32_t left = x > 0 ? lab_at(y, x - 1) : 0;
            int32_t up = y > 0 ? lab_at(y - 1, x) : 0;
            int32_t ul = (x > 0 && y > 0) ? lab_at(y - 1, x - 1) : 0;
            int32_t ur = (y > 0 && x + 1 < w) ? lab_at(y - 1, x + 1) : 0;

            int32_t m = 0;
            for (int32_t n : {left, up, ul, ur}) {
                if (n != 0 && (m == 0 || n < m)) m = n;
            }
            if (m == 0) {
                m = next_label++;
                uf.make();
            }
            lab_at(y, x) = m;
            for (int32_t n : {left, up, ul, ur}) {
                if (n != 0) uf.unite(static_cast<int>(m), static_cast<int>(n));
            }
        }
    }

    std::unordered_map<int, int32_t> remap;
    int32_t out_next = 1;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int32_t& v = lab_at(y, x);
            if (v == 0) continue;
            int root = uf.find(static_cast<int>(v));
            auto it = remap.find(root);
            int32_t out_label;
            if (it == remap.end()) {
                out_label = out_next++;
                remap.emplace(root, out_label);
            } else {
                out_label = it->second;
            }
            v = out_label;
        }
    }
    return labels;
}

std::vector<Component> components_from_labels(const std::vector<int32_t>& labels, int w, int h) {
    int32_t n = 0;
    for (int32_t v : labels) n = std::max(n, v);
    if (n == 0) return {};

    std::vector<int> x1(n + 1, std::numeric_limits<int>::max());
    std::vector<int> y1(n + 1, std::numeric_limits<int>::max());
    std::vector<int> x2(n + 1, std::numeric_limits<int>::min());
    std::vector<int> y2(n + 1, std::numeric_limits<int>::min());
    std::vector<int> area(n + 1, 0);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int32_t lab = labels[static_cast<size_t>(y) * w + x];
            if (lab == 0) continue;
            x1[lab] = std::min(x1[lab], x);
            y1[lab] = std::min(y1[lab], y);
            x2[lab] = std::max(x2[lab], x);
            y2[lab] = std::max(y2[lab], y);
            area[lab] += 1;
        }
    }

    std::vector<Component> out;
    out.reserve(n);
    for (int32_t lab = 1; lab <= n; ++lab) {
        if (area[lab] == 0) continue;
        out.push_back({lab, x1[lab], y1[lab], x2[lab] + 1, y2[lab] + 1, area[lab]});
    }
    return out;
}

std::vector<Component> segment(const Mask& mask, int min_area) {
    std::vector<int32_t> labels = label_components(mask);
    std::vector<Component> comps = components_from_labels(labels, mask.w, mask.h);
    comps.erase(std::remove_if(comps.begin(), comps.end(),
                                [&](const Component& c) { return c.area < min_area; }),
                comps.end());
    std::stable_sort(comps.begin(), comps.end(), [](const Component& a, const Component& b) {
        return a.x1 < b.x1;
    });
    return comps;
}

}  // namespace omr
