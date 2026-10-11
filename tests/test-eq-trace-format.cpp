#include "../tools/eq-trace/eq-trace-format.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

using namespace eq_trace;

static void test_header() {
    header h;
    std::memcpy(h.magic, "EQTH", 4);
    h.n_layer   = 16;
    h.n_expert  = 64;
    h.top_k     = 8;
    h.n_embd    = 2048;
    h.n_records = 0x0102030405060708ull;
    uint8_t buf[HEADER_SIZE];
    encode_header(h, buf);
    static const uint8_t expect[HEADER_SIZE] = {
        'E', 'Q', 'T', 'H', 1, 0, 0, 0, 16, 0, 0, 0, 64, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0,
        8, 7, 6, 5, 4, 3, 2, 1,
    };
    CHECK(std::memcmp(buf, expect, HEADER_SIZE) == 0);
    header d;
    CHECK(decode_header(buf, d));
    CHECK(std::memcmp(d.magic, "EQTH", 4) == 0 && d.version == 1 && d.n_layer == 16 && d.n_expert == 64 &&
          d.top_k == 8 && d.n_embd == 2048 && d.n_records == h.n_records);
    buf[0] = 'X';
    CHECK(!decode_header(buf, d));
    buf[0] = 'E';
    buf[4] = 2;
    CHECK(!decode_header(buf, d));
}

static void test_route_record() {
    const uint32_t top_k = 3, n_expert = 5;
    const uint16_t ids[3] = {4, 0, 258};
    const float    lg[5]  = {1.5f, -2.25f, 0.0f, 1e-30f, -3e8f};
    CHECK(route_record_size(top_k, n_expert) == 10 + 6 + 20);
    std::vector<uint8_t> buf(route_record_size(top_k, n_expert));
    encode_route_record(buf.data(), 0xdeadbeefu, 513, 7, ids, top_k, lg, n_expert);
    // spot check byte order
    CHECK(buf[0] == 0xef && buf[3] == 0xde && buf[4] == 0x01 && buf[5] == 0x02 && buf[8] == 7 && buf[9] == 0);
    CHECK(buf[10] == 4 && buf[14] == 0x02 && buf[15] == 0x01); // ids[2] = 258
    CHECK(buf[16] == 0x00 && buf[17] == 0x00 && buf[18] == 0xc0 && buf[19] == 0x3f); // 1.5f
    const route_record r = decode_route_record(buf.data(), top_k, n_expert);
    CHECK(r.prompt_id == 0xdeadbeefu && r.token_pos == 513 && r.layer == 7);
    for (uint32_t i = 0; i < top_k; i++) CHECK(r.ids[i] == ids[i]);
    for (uint32_t i = 0; i < n_expert; i++) CHECK(std::memcmp(&r.logits[i], &lg[i], 4) == 0);
}

static void test_f16() {
    CHECK(f32_to_f16(0.0f) == 0x0000);
    CHECK(f32_to_f16(-0.0f) == 0x8000);
    CHECK(f32_to_f16(1.0f) == 0x3c00);
    CHECK(f32_to_f16(-2.0f) == 0xc000);
    CHECK(f32_to_f16(0.5f) == 0x3800);
    CHECK(f32_to_f16(65504.0f) == 0x7bff);
    CHECK(f32_to_f16(65520.0f) == 0x7c00); // rounds up to inf
    CHECK(f32_to_f16(1e9f) == 0x7c00);
    CHECK(f32_to_f16(-1e9f) == 0xfc00);
    CHECK(f32_to_f16(INFINITY) == 0x7c00);
    CHECK((f32_to_f16(NAN) & 0x7c00) == 0x7c00 && (f32_to_f16(NAN) & 0x3ff) != 0);
    CHECK(f32_to_f16(5.9604645e-8f) == 0x0001);  // smallest subnormal
    CHECK(f32_to_f16(2.9802322e-8f) == 0x0000);  // tie to even -> 0
    CHECK(f32_to_f16(1e-10f) == 0x0000);
    CHECK(f32_to_f16(1.0f + 1.0f / 2048) == 0x3c00); // tie to even
    CHECK(f32_to_f16(1.0f + 3.0f / 2048) == 0x3c02); // tie to even (up)
    // exhaustive round trip over every non-NaN half
    for (uint32_t h = 0; h < 0x10000; h++) {
        if ((h & 0x7c00) == 0x7c00 && (h & 0x3ff)) continue;
        const float f = f16_to_f32((uint16_t) h);
        CHECK(f32_to_f16(f) == h);
    }
    CHECK(f16_to_f32(0x3c00) == 1.0f && f16_to_f32(0xc000) == -2.0f && f16_to_f32(0x0001) == 5.9604645e-8f);
    // nearest: midpoint perturbations around each normal half
    for (uint32_t h = 0x0400; h < 0x7bff; h += 7) {
        const double a = f16_to_f32((uint16_t) h), b = f16_to_f32((uint16_t) (h + 1));
        const float below = (float) (a + (b - a) * 0.4), above = (float) (a + (b - a) * 0.6);
        CHECK(f32_to_f16(below) == h);
        CHECK(f32_to_f16(above) == h + 1);
    }
}

static void test_hidden_record() {
    const uint32_t n_embd = 4;
    const uint16_t hv[4]  = {0x3c00, 0xc000, 0x0001, 0x7bff};
    CHECK(hidden_record_size(n_embd) == 18);
    std::vector<uint8_t> buf(hidden_record_size(n_embd));
    encode_hidden_record(buf.data(), 9, 70000, 31, hv, n_embd);
    CHECK(buf[4] == 0x70 && buf[5] == 0x11 && buf[6] == 0x01 && buf[8] == 31);
    CHECK(buf[10] == 0x00 && buf[11] == 0x3c);
    const hidden_record r = decode_hidden_record(buf.data(), n_embd);
    CHECK(r.prompt_id == 9 && r.token_pos == 70000 && r.layer == 31);
    for (uint32_t i = 0; i < n_embd; i++) CHECK(r.hidden_f16[i] == hv[i]);
}

static void test_sampling() {
    // splitmix64 reference: state 0 gives these outputs (first two of the reference generator)
    CHECK(splitmix64(0) == 0xe220a8397b1dcdafull);
    CHECK(splitmix64(0x9E3779B97F4A7C15ull) == 0x6e789e6aa1b965f4ull);
    // pinned value of the combined hash, python port must match
    CHECK(token_hash(0, 0, 0) == splitmix64(splitmix64(splitmix64(0) ^ 0) ^ 0));
    CHECK(token_hash(1, 2, 3) == splitmix64(splitmix64(splitmix64(1) ^ 2) ^ 3));
    // deterministic, fraction 1 -> all, 0 -> none
    for (uint32_t p = 0; p < 50; p++) {
        for (uint32_t t = 0; t < 50; t++) {
            CHECK(token_sampled(7, p, t, 1.0));
            CHECK(!token_sampled(7, p, t, 0.0));
            CHECK(token_sampled(7, p, t, 0.1) == token_sampled(7, p, t, 0.1));
        }
    }
    // fraction close to F
    const int n = 200000;
    int hit = 0, hit_other_seed = 0, differ = 0;
    for (int i = 0; i < n; i++) {
        const bool a = token_sampled(0, i / 1000, i % 1000, 0.1);
        const bool b = token_sampled(1, i / 1000, i % 1000, 0.1);
        hit += a;
        hit_other_seed += b;
        differ += a != b;
    }
    CHECK(std::fabs((double) hit / n - 0.1) < 0.005);
    CHECK(std::fabs((double) hit_other_seed / n - 0.1) < 0.005);
    CHECK(differ > 0);
}

int main() {
    test_header();
    test_route_record();
    test_f16();
    test_hidden_record();
    test_sampling();
    std::printf("test-eq-trace-format: OK\n");
    return 0;
}
