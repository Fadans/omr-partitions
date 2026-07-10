#include "npy.hpp"

#include <cctype>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace omr {

std::vector<double> NpyArray::as_f64() const {
    size_t n = raw.size() / sizeof(double);
    std::vector<double> out(n);
    std::memcpy(out.data(), raw.data(), n * sizeof(double));
    return out;
}

std::vector<int64_t> NpyArray::as_i64() const {
    size_t n = raw.size() / sizeof(int64_t);
    std::vector<int64_t> out(n);
    std::memcpy(out.data(), raw.data(), n * sizeof(int64_t));
    return out;
}

std::vector<std::string> NpyArray::as_unicode() const {
    size_t u_pos = descr.find('U');
    if (u_pos == std::string::npos) throw std::runtime_error("dtype non unicode : " + descr);
    int n_chars = std::stoi(descr.substr(u_pos + 1));
    size_t item_bytes = static_cast<size_t>(n_chars) * 4;
    size_t n = item_bytes ? raw.size() / item_bytes : 0;

    std::vector<std::string> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        std::string s;
        for (int c = 0; c < n_chars; ++c) {
            size_t off = i * item_bytes + static_cast<size_t>(c) * 4;
            uint32_t cp = static_cast<uint32_t>(raw[off]) | (static_cast<uint32_t>(raw[off + 1]) << 8) |
                          (static_cast<uint32_t>(raw[off + 2]) << 16) |
                          (static_cast<uint32_t>(raw[off + 3]) << 24);
            if (cp == 0) break;
            s += static_cast<char>(cp & 0xFF);
        }
        out.push_back(std::move(s));
    }
    return out;
}

namespace {

std::string header_value(const std::string& header, const std::string& key) {
    size_t p = header.find(key);
    if (p == std::string::npos) throw std::runtime_error("en-tête npy sans clé : " + key);
    size_t q1 = header.find('\'', p + key.size());
    size_t q2 = header.find('\'', q1 + 1);
    return header.substr(q1 + 1, q2 - q1 - 1);
}

std::vector<size_t> header_shape(const std::string& header) {
    size_t p = header.find("'shape':");
    p = header.find('(', p);
    size_t end = header.find(')', p);
    std::string inner = header.substr(p + 1, end - p - 1);
    std::vector<size_t> shape;
    std::string cur;
    for (char c : inner) {
        if (c == ',') {
            if (!cur.empty()) {
                shape.push_back(static_cast<size_t>(std::stoul(cur)));
                cur.clear();
            }
        } else if (!std::isspace(static_cast<unsigned char>(c))) {
            cur += c;
        }
    }
    if (!cur.empty()) shape.push_back(static_cast<size_t>(std::stoul(cur)));
    return shape;
}

NpyArray parse_npy(const std::vector<uint8_t>& buf) {
    static const uint8_t magic[6] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    if (buf.size() < 10 || std::memcmp(buf.data(), magic, 6) != 0) {
        throw std::runtime_error("en-tête .npy invalide");
    }
    uint8_t major = buf[6];
    size_t header_len, header_start;
    if (major == 1) {
        header_len = buf[8] | (static_cast<size_t>(buf[9]) << 8);
        header_start = 10;
    } else {
        header_len = buf[8] | (static_cast<size_t>(buf[9]) << 8) | (static_cast<size_t>(buf[10]) << 16) |
                     (static_cast<size_t>(buf[11]) << 24);
        header_start = 12;
    }
    std::string header(reinterpret_cast<const char*>(&buf[header_start]), header_len);

    NpyArray arr;
    arr.descr = header_value(header, "'descr':");
    arr.shape = header_shape(header);
    if (header.find("'fortran_order': True") != std::string::npos) {
        throw std::runtime_error("tableau fortran_order non supporté");
    }
    size_t data_start = header_start + header_len;
    arr.raw.assign(buf.begin() + data_start, buf.end());
    return arr;
}

}  // namespace

std::unordered_map<std::string, NpyArray> read_npz(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("impossible d'ouvrir : " + path);
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    auto u16 = [&](size_t o) {
        return static_cast<uint32_t>(buf[o]) | (static_cast<uint32_t>(buf[o + 1]) << 8);
    };
    auto u32 = [&](size_t o) {
        return static_cast<uint32_t>(buf[o]) | (static_cast<uint32_t>(buf[o + 1]) << 8) |
               (static_cast<uint32_t>(buf[o + 2]) << 16) | (static_cast<uint32_t>(buf[o + 3]) << 24);
    };
    auto u64 = [&](size_t o) {
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(buf[o + i]) << (8 * i);
        return v;
    };

    std::unordered_map<std::string, NpyArray> out;
    size_t pos = 0;
    while (pos + 30 <= buf.size() && buf[pos] == 0x50 && buf[pos + 1] == 0x4B &&
           buf[pos + 2] == 0x03 && buf[pos + 3] == 0x04) {
        uint16_t method = static_cast<uint16_t>(u16(pos + 8));
        uint32_t comp_size32 = u32(pos + 18);
        uint32_t uncomp_size32 = u32(pos + 22);
        uint16_t name_len = static_cast<uint16_t>(u16(pos + 26));
        uint16_t extra_len = static_cast<uint16_t>(u16(pos + 28));
        size_t name_start = pos + 30;
        size_t extra_start = name_start + name_len;
        std::string name(reinterpret_cast<const char*>(&buf[name_start]), name_len);

        if (method != 0) throw std::runtime_error("entrée ZIP compressée non supportée : " + name);

        // ZIP64 : tailles 32 bits à 0xFFFFFFFF -> lues dans le champ extra.
        uint64_t comp_size = comp_size32;
        if (comp_size32 == 0xFFFFFFFFu || uncomp_size32 == 0xFFFFFFFFu) {
            size_t e = extra_start, e_end = extra_start + extra_len;
            while (e + 4 <= e_end) {
                uint16_t id = static_cast<uint16_t>(u16(e));
                uint16_t sz = static_cast<uint16_t>(u16(e + 2));
                if (id == 0x0001) {
                    size_t p = e + 4;
                    if (uncomp_size32 == 0xFFFFFFFFu) p += 8;  // taille non compressée, ignorée ici
                    if (comp_size32 == 0xFFFFFFFFu) comp_size = u64(p);
                    break;
                }
                e += 4 + sz;
            }
        }

        size_t data_start = extra_start + extra_len;
        std::vector<uint8_t> data(buf.begin() + data_start,
                                   buf.begin() + data_start + comp_size);
        std::string key = name;
        if (key.size() > 4 && key.substr(key.size() - 4) == ".npy") key.resize(key.size() - 4);
        out.emplace(key, parse_npy(data));

        pos = data_start + comp_size;
    }
    return out;
}

}  // namespace omr
