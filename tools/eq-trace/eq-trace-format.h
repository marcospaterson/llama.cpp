#pragma once

// File format, record codecs and token sampling hash for llama-eq-trace.
// Header-only, no llama/ggml dependencies. All integers little-endian.
// See README.md in this directory for the exact specification.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace eq_trace {

constexpr uint32_t VERSION     = 1;
constexpr size_t   HEADER_SIZE = 32;

struct header {
    char     magic[4]  = {'E', 'Q', 'T', 'R'}; // "EQTR" routes, "EQTH" hidden
    uint32_t version   = VERSION;
    uint32_t n_layer   = 0; // number of MoE layers traced
    uint32_t n_expert  = 0;
    uint32_t top_k     = 0;
    uint32_t n_embd    = 0;
    uint64_t n_records = 0;
};

// ---- little-endian primitives ----

inline void put_u16(uint8_t * p, uint16_t v) {
    p[0] = (uint8_t) (v & 0xff);
    p[1] = (uint8_t) (v >> 8);
}
inline void put_u32(uint8_t * p, uint32_t v) {
    for (int i = 0; i < 4; i++) {
        p[i] = (uint8_t) (v >> (8 * i));
    }
}
inline void put_u64(uint8_t * p, uint64_t v) {
    for (int i = 0; i < 8; i++) {
        p[i] = (uint8_t) (v >> (8 * i));
    }
}
inline uint16_t get_u16(const uint8_t * p) {
    return (uint16_t) (p[0] | (p[1] << 8));
}
inline uint32_t get_u32(const uint8_t * p) {
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
        v |= (uint32_t) p[i] << (8 * i);
    }
    return v;
}
inline uint64_t get_u64(const uint8_t * p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) {
        v |= (uint64_t) p[i] << (8 * i);
    }
    return v;
}
inline void put_f32(uint8_t * p, float f) {
    uint32_t v;
    std::memcpy(&v, &f, 4);
    put_u32(p, v);
}
inline float get_f32(const uint8_t * p) {
    uint32_t v = get_u32(p);
    float    f;
    std::memcpy(&f, &v, 4);
    return f;
}

// ---- IEEE binary16 ----

// round to nearest even; NaN stays NaN, overflow becomes inf
inline uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t e    = (x >> 23) & 0xffu;
    uint32_t       m    = x & 0x7fffffu;
    if (e == 0xffu) {
        return (uint16_t) (sign | 0x7c00u | (m ? (0x200u | (m >> 13)) : 0u));
    }
    const int exp = (int) e - 127 + 15;
    if (exp >= 31) {
        return (uint16_t) (sign | 0x7c00u);
    }
    if (exp <= 0) {
        if (exp < -10) {
            return (uint16_t) sign;
        }
        m |= 0x800000u;
        const int      shift = 14 - exp;
        uint32_t       h     = m >> shift;
        const uint32_t rem   = m & ((1u << shift) - 1u);
        const uint32_t half  = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1u))) {
            h++;
        }
        return (uint16_t) (sign | h);
    }
    uint32_t       h   = ((uint32_t) exp << 10) | (m >> 13);
    const uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) {
        h++; // a carry into the exponent is correct (up to inf)
    }
    return (uint16_t) (sign | h);
}

inline float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    const uint32_t e    = (h >> 10) & 0x1fu;
    const uint32_t m    = h & 0x3ffu;
    uint32_t       x;
    if (e == 0) {
        if (m == 0) {
            x = sign;
        } else {
            // subnormal: m * 2^-24
            float v = (float) m * (1.0f / 16777216.0f);
            std::memcpy(&x, &v, 4);
            x |= sign;
        }
    } else if (e == 31) {
        x = sign | 0x7f800000u | (m << 13);
    } else {
        x = sign | ((e - 15 + 127) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

// ---- header ----

inline void encode_header(const header & h, uint8_t out[HEADER_SIZE]) {
    std::memcpy(out, h.magic, 4);
    put_u32(out + 4, h.version);
    put_u32(out + 8, h.n_layer);
    put_u32(out + 12, h.n_expert);
    put_u32(out + 16, h.top_k);
    put_u32(out + 20, h.n_embd);
    put_u64(out + 24, h.n_records);
}

// false if the magic is neither EQTR nor EQTH or the version is unsupported
inline bool decode_header(const uint8_t in[HEADER_SIZE], header & h) {
    std::memcpy(h.magic, in, 4);
    if (std::memcmp(h.magic, "EQTR", 4) != 0 && std::memcmp(h.magic, "EQTH", 4) != 0) {
        return false;
    }
    h.version   = get_u32(in + 4);
    h.n_layer   = get_u32(in + 8);
    h.n_expert  = get_u32(in + 12);
    h.top_k     = get_u32(in + 16);
    h.n_embd    = get_u32(in + 20);
    h.n_records = get_u64(in + 24);
    return h.version == VERSION;
}

// ---- routes record: u32 prompt_id, u32 token_pos, u16 layer, u16 ids[top_k], f32 logits[n_expert] ----

inline size_t route_record_size(uint32_t top_k, uint32_t n_expert) {
    return 4 + 4 + 2 + 2 * (size_t) top_k + 4 * (size_t) n_expert;
}

inline void encode_route_record(uint8_t * out, uint32_t prompt_id, uint32_t token_pos, uint16_t layer,
                                const uint16_t * ids, uint32_t top_k, const float * logits, uint32_t n_expert) {
    put_u32(out, prompt_id);
    put_u32(out + 4, token_pos);
    put_u16(out + 8, layer);
    uint8_t * p = out + 10;
    for (uint32_t i = 0; i < top_k; i++, p += 2) {
        put_u16(p, ids[i]);
    }
    for (uint32_t i = 0; i < n_expert; i++, p += 4) {
        put_f32(p, logits[i]);
    }
}

struct route_record {
    uint32_t              prompt_id = 0;
    uint32_t              token_pos = 0;
    uint16_t              layer     = 0;
    std::vector<uint16_t> ids;
    std::vector<float>    logits;
};

inline route_record decode_route_record(const uint8_t * in, uint32_t top_k, uint32_t n_expert) {
    route_record r;
    r.prompt_id = get_u32(in);
    r.token_pos = get_u32(in + 4);
    r.layer     = get_u16(in + 8);
    const uint8_t * p = in + 10;
    r.ids.resize(top_k);
    for (uint32_t i = 0; i < top_k; i++, p += 2) {
        r.ids[i] = get_u16(p);
    }
    r.logits.resize(n_expert);
    for (uint32_t i = 0; i < n_expert; i++, p += 4) {
        r.logits[i] = get_f32(p);
    }
    return r;
}

// ---- hidden record: u32 prompt_id, u32 token_pos, u16 layer, f16 hidden[n_embd] ----

inline size_t hidden_record_size(uint32_t n_embd) {
    return 4 + 4 + 2 + 2 * (size_t) n_embd;
}

// hidden_f16 holds IEEE half bit patterns (see f32_to_f16)
inline void encode_hidden_record(uint8_t * out, uint32_t prompt_id, uint32_t token_pos, uint16_t layer,
                                 const uint16_t * hidden_f16, uint32_t n_embd) {
    put_u32(out, prompt_id);
    put_u32(out + 4, token_pos);
    put_u16(out + 8, layer);
    uint8_t * p = out + 10;
    for (uint32_t i = 0; i < n_embd; i++, p += 2) {
        put_u16(p, hidden_f16[i]);
    }
}

struct hidden_record {
    uint32_t              prompt_id = 0;
    uint32_t              token_pos = 0;
    uint16_t              layer     = 0;
    std::vector<uint16_t> hidden_f16;
};

inline hidden_record decode_hidden_record(const uint8_t * in, uint32_t n_embd) {
    hidden_record r;
    r.prompt_id = get_u32(in);
    r.token_pos = get_u32(in + 4);
    r.layer     = get_u16(in + 8);
    r.hidden_f16.resize(n_embd);
    const uint8_t * p = in + 10;
    for (uint32_t i = 0; i < n_embd; i++, p += 2) {
        r.hidden_f16[i] = get_u16(p);
    }
    return r;
}

// ---- sampling hash ----

inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// h = splitmix64(seed); h = splitmix64(h ^ prompt_id); h = splitmix64(h ^ token_pos)
inline uint64_t token_hash(uint64_t seed, uint32_t prompt_id, uint32_t token_pos) {
    uint64_t h = splitmix64(seed);
    h = splitmix64(h ^ (uint64_t) prompt_id);
    h = splitmix64(h ^ (uint64_t) token_pos);
    return h;
}

// u = (h >> 11) * 2^-53 in [0, 1); the token is recorded iff u < fraction
// (fraction >= 1 records everything, fraction <= 0 nothing)
inline bool token_sampled(uint64_t seed, uint32_t prompt_id, uint32_t token_pos, double fraction) {
    const double u = (double) (token_hash(seed, prompt_id, token_pos) >> 11) * (1.0 / 9007199254740992.0);
    return u < fraction;
}

} // namespace eq_trace
