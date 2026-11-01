// ============================================================================
//  DragonZip — kuchli, yo'qotishsiz (lossless) arxivator
//
//  Siqish yadrosi: "context mixing" (CM) — har bir bit uchun o'nlab
//  kontekst modellarining bashoratlari neyron-mikser orqali birlashtiriladi,
//  so'ng SSE/APM bosqichlari bilan aniqlashtiriladi va arifmetik kodlovchi
//  bilan yoziladi. Barcha hisoblar butun sonlarda — natija har qanday
//  platformada bitma-bit bir xil (deterministik).
//
//  Asosiy qismlar:
//    * 20 ta kontekst: order 0-6, 8; so'z (word) modellari; siyrak (sparse)
//      kontekstlar; matn ustuni (column), yozuv (record) va bilvosita
//      (indirect) modellar
//    * Har bir kontekst uchun 1 baytli bit-tarix holati (n0, n1) va
//      holat -> ehtimollik adaptiv xaritasi; order-1/2 uchun aniq hisoblagichlar
//    * Uzun takrorlarni topuvchi "match model"
//    * 5 ta tanlovchi kontekstli neyron mikser + yakuniy mikser
//    * 4 ta APM/SSE bosqichi
//    * 32-bit arifmetik kodlovchi (16-bit ehtimollik aniqligi)
//    * x86 dasturlari uchun E8/E9 filtri
//    * Allaqachon siqilgan fayllarni aniqlash (vaqtni tejash)
//    * Solid arxiv, ko'p oqimli (multi-thread) ishlash, CRC32 tekshiruvi
//
//  Yig'ish:  g++ -O2 -std=c++17 -pthread dragonzip.cpp -o dragonzip
// ============================================================================

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#include <io.h>
#ifdef _MSC_VER
#pragma comment(lib, "shell32.lib")
#endif
#define DZ_ISATTY(fd) _isatty(fd)
#else
#include <unistd.h>
#include <sys/mman.h>
#define DZ_ISATTY(fd) isatty(fd)
#endif

#if defined(__GNUC__) || defined(__clang__)
#define DZ_PREFETCH(p) __builtin_prefetch(p)
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <xmmintrin.h>
#define DZ_PREFETCH(p) _mm_prefetch((const char*)(p), _MM_HINT_T0)
#else
#define DZ_PREFETCH(p) ((void)0)
#endif

namespace fs = std::filesystem;

typedef uint8_t U8;
typedef uint16_t U16;
typedef uint32_t U32;
typedef uint64_t U64;
typedef int64_t I64;

static const char* DZ_VERSION = "1.0";

// ============================================================================
//  Jadval va yordamchi funksiyalar
// ============================================================================

static int SQ_T[4095];     // squash: stretch-domen (-2047..2047) -> 12-bit ehtimollik
static short ST_T[4096];   // stretch: 12-bit ehtimollik -> ln(p/(1-p)) * 256
static int DT16[1024];     // StateMap tezligi 1/(n+1.5), 16-bit
static U32 CRC_T[256];

static inline int squash(int d) {
  if (d > 2047) d = 2047;
  if (d < -2047) d = -2047;
  return SQ_T[d + 2047];
}
static inline int stretch(int p) { return ST_T[p]; }

static void initTables() {
  // Logistik funksiya 4096/(1+e^-x), x = -8..8 qadam 0.5 (butun sonlarda,
  // shuning uchun har qanday kompilyator/platformada aynan bir xil).
  static const int L[33] = {1,    2,    4,    6,    10,   17,   27,   45,   74,
                            120,  194,  311,  488,  747,  1102, 1546, 2048, 2550,
                            2994, 3349, 3608, 3785, 3902, 3976, 4022, 4051, 4069,
                            4079, 4086, 4090, 4092, 4094, 4095};
  for (int d = -2047; d <= 2047; ++d) {
    int w = d & 127, i = (d >> 7) + 16;
    SQ_T[d + 2047] = (L[i] * (128 - w) + L[i + 1] * w + 64) >> 7;
  }
  int pi = 0;
  for (int x = -2047; x <= 2047; ++x) {
    int v = SQ_T[x + 2047];
    for (int j = pi; j <= v && j < 4096; ++j) ST_T[j] = (short)x;
    pi = v + 1;
  }
  for (int j = pi; j < 4096; ++j) ST_T[j] = 2047;
  for (int n = 0; n < 1024; ++n) DT16[n] = 131072 / (2 * n + 3);
  for (U32 i = 0; i < 256; ++i) {
    U32 c = i;
    for (int k = 0; k < 8; ++k) c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
    CRC_T[i] = c;
  }
}

static U32 crc32Update(U32 crc, const U8* p, size_t n) {
  crc = ~crc;
  while (n--) crc = CRC_T[(crc ^ *p++) & 255] ^ (crc >> 8);
  return ~crc;
}

static inline U64 hmix(U64 x) {
  x ^= x >> 31;
  x *= 0x7FB5D329728EA185ULL;
  x ^= x >> 27;
  x *= 0x81DADEF4BC2DD44DULL;
  x ^= x >> 33;
  return x;
}
static inline U64 hctx(U64 id, U64 v) {
  return hmix(v * 0x9E3779B97F4A7C15ULL + id * 0xD6E8FEB86659FD93ULL + id);
}

static inline int ceilLog2(U64 x) {
  int r = 0;
  while ((U64(1) << r) < x) ++r;
  return r;
}

// ============================================================================
//  Adaptiv hisoblagichlar
// ============================================================================

// Bit-tarix holat mashinasi: har bir holat (n0, n1) — nol va birlar soni
// (taxminiy). Qarama-qarshi bit kelganda eski hisob kamaytiriladi
// (nostatsionar diskontlash), shunda model o'zgaruvchan ma'lumotga moslashadi.
// Jadval ishga tushishda deterministik tarzda (BFS) quriladi.
static U8 ST_NEXT[256][2];
static U8 ST_TOTAL[256];
static int ST_COUNT = 0;

static int stCap(int o) { return o == 0 ? 60 : o == 1 ? 36 : o == 2 ? 20 : o == 3 ? 12 : o < 6 ? 7 : 5; }
static int stDisc(int n) { return n <= 1 ? n : (n + 1) / 2; }

static void initStates() {
  static int idx[64][64];
  int n0s[256], n1s[256];
  for (int a = 0; a < 64; ++a)
    for (int b = 0; b < 64; ++b) idx[a][b] = -1;
  idx[0][0] = 0;
  n0s[0] = n1s[0] = 0;
  ST_COUNT = 1;
  for (int st = 0; st < ST_COUNT; ++st) {
    for (int y = 0; y < 2; ++y) {
      int n0 = n0s[st], n1 = n1s[st];
      if (y) {
        if (n1 < stCap(n0)) ++n1;
        n0 = stDisc(n0);
        if (n1 > stCap(n0)) n1 = stCap(n0);
      } else {
        if (n0 < stCap(n1)) ++n0;
        n1 = stDisc(n1);
        if (n0 > stCap(n1)) n0 = stCap(n1);
      }
      if (idx[n0][n1] < 0) {
        if (ST_COUNT >= 256) { fprintf(stderr, "ichki xato: holatlar ko'p\n"); exit(9); }
        idx[n0][n1] = ST_COUNT;
        n0s[ST_COUNT] = n0;
        n1s[ST_COUNT] = n1;
        ++ST_COUNT;
      }
      ST_NEXT[st][y] = (U8)idx[n0][n1];
    }
  }
  for (int st = 0; st < 256; ++st) ST_TOTAL[st] = st < ST_COUNT ? (U8)std::min(255, n0s[st] + n1s[st]) : 0;
}

// Slot (8 bit): bit-tarix holati (yuqoridagi mashina bo'yicha). 0 = yangi.
static inline void slotUpdate(U8* s, int y) { *s = ST_NEXT[*s][y]; }

// StateMap yozuvi (32 bit): [31..10] 22-bit ehtimollik, [9..0] hisob
static const U32 SM_INIT = 1u << 31;

static inline void smUpdate(U32* e, int y, int limit) {
  U32 v = *e;
  int n = (int)(v & 1023), p = (int)(v >> 10);
  p += (int)(((I64)((y << 22) - p) * DT16[n]) >> 16);
  if (n < limit) ++n;
  *e = ((U32)p << 10) | (U32)n;
}

// ============================================================================
//  Xesh jadval: 16 baytlik "bucket" = 8-bit tekshiruv + 15 ta holat
//  (yarim baytning (nibble) barcha 15 ta bit-tuguni). 3 yo'lli (3-way)
//  qidiruv — barcha nomzodlar bitta 64 baytlik kesh qatorida.
// ============================================================================

class HashTable {
  U8* raw_;
  U8* t_;
  int bb_;  // bucket soni = 2^bb_
 public:
  explicit HashTable(int tbits) {
    bb_ = tbits - 4;
    size_t bytes = (size_t)16 << bb_;
    raw_ = (U8*)calloc(bytes + 64, 1);
    if (!raw_) throw std::bad_alloc();
    t_ = (U8*)(((uintptr_t)raw_ + 63) & ~(uintptr_t)63);
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    madvise(t_, bytes & ~(size_t)4095, MADV_HUGEPAGE);
#endif
  }
  ~HashTable() { free(raw_); }
  HashTable(const HashTable&) = delete;
  HashTable& operator=(const HashTable&) = delete;

  inline void prefetch(U64 k) const {
    DZ_PREFETCH(t_ + ((size_t)(k >> (64 - bb_)) << 4));
  }
  inline U8* find(U64 k, bool& found) {
    size_t i = (size_t)(k >> (64 - bb_));
    U8 chk = (U8)(k >> 8);
    if (!chk) chk = 1;
    U8* b0 = t_ + (i << 4);
    U8* b1 = t_ + ((i ^ 1) << 4);
    U8* b2 = t_ + ((i ^ 2) << 4);
    found = true;
    if (b0[0] == chk) return b0;
    if (b1[0] == chk) return b1;
    if (b2[0] == chk) return b2;
    U8* r = b0;
    if (ST_TOTAL[b1[1]] < ST_TOTAL[r[1]]) r = b1;
    if (ST_TOTAL[b2[1]] < ST_TOTAL[r[1]]) r = b2;
    r[0] = chk;
    memset(r + 1, 0, 15);
    found = false;
    return r;
  }
};

// ============================================================================
//  Neyron mikser (logistik aralashtirish, onlayn gradient tushish)
// ============================================================================

// Og'irliklar 16-bit (1.0 = 8192), kirishlar 16-bit. SIMD (AVX2/SSE2) va oddiy
// C++ variantlari bitma-bit bir xil natija beradi:
//   dot   = sum over pairs ((x0*w0 + x1*w1) >> 8)
//   train = w += ((sat(2x) * err) >> 16 + 1) >> 1   (to'yinuvchi qo'shish)
static const int MIX_ALIGN = 16;  // kirishlar soni shunga karrali bo'lishi kerak

static inline short sat16(int v) { return (short)(v > 32767 ? 32767 : v < -32768 ? -32768 : v); }

#if defined(__AVX2__) && !defined(DZ_NO_SIMD)
#include <immintrin.h>
static inline int dotProduct(const short* x, const short* w, int n) {
  __m256i s = _mm256_setzero_si256();
  for (int i = 0; i < n; i += 16) {
    __m256i p = _mm256_madd_epi16(_mm256_loadu_si256((const __m256i*)(x + i)), _mm256_loadu_si256((const __m256i*)(w + i)));
    s = _mm256_add_epi32(s, _mm256_srai_epi32(p, 8));
  }
  __m128i a = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
  a = _mm_add_epi32(a, _mm_srli_si128(a, 8));
  a = _mm_add_epi32(a, _mm_srli_si128(a, 4));
  return _mm_cvtsi128_si32(a);
}
static inline void train(const short* x, short* w, int n, int err) {
  const __m256i e = _mm256_set1_epi16((short)err), one = _mm256_set1_epi16(1);
  for (int i = 0; i < n; i += 16) {
    __m256i t = _mm256_loadu_si256((const __m256i*)(x + i));
    t = _mm256_adds_epi16(t, t);
    t = _mm256_mulhi_epi16(t, e);
    t = _mm256_srai_epi16(_mm256_adds_epi16(t, one), 1);
    __m256i v = _mm256_loadu_si256((const __m256i*)(w + i));
    _mm256_storeu_si256((__m256i*)(w + i), _mm256_adds_epi16(v, t));
  }
}
#elif (defined(__SSE2__) || defined(_M_X64)) && !defined(DZ_NO_SIMD)
#include <emmintrin.h>
static inline int dotProduct(const short* x, const short* w, int n) {
  __m128i s = _mm_setzero_si128();
  for (int i = 0; i < n; i += 8) {
    __m128i p = _mm_madd_epi16(_mm_loadu_si128((const __m128i*)(x + i)), _mm_loadu_si128((const __m128i*)(w + i)));
    s = _mm_add_epi32(s, _mm_srai_epi32(p, 8));
  }
  s = _mm_add_epi32(s, _mm_srli_si128(s, 8));
  s = _mm_add_epi32(s, _mm_srli_si128(s, 4));
  return _mm_cvtsi128_si32(s);
}
static inline void train(const short* x, short* w, int n, int err) {
  const __m128i e = _mm_set1_epi16((short)err), one = _mm_set1_epi16(1);
  for (int i = 0; i < n; i += 8) {
    __m128i t = _mm_loadu_si128((const __m128i*)(x + i));
    t = _mm_adds_epi16(t, t);
    t = _mm_mulhi_epi16(t, e);
    t = _mm_srai_epi16(_mm_adds_epi16(t, one), 1);
    __m128i v = _mm_loadu_si128((const __m128i*)(w + i));
    _mm_storeu_si128((__m128i*)(w + i), _mm_adds_epi16(v, t));
  }
}
#else
static inline int dotProduct(const short* x, const short* w, int n) {
  int s = 0;
  for (int i = 0; i < n; i += 2) s += (x[i] * w[i] + x[i + 1] * w[i + 1]) >> 8;
  return s;
}
static inline void train(const short* x, short* w, int n, int err) {
  for (int i = 0; i < n; ++i) {
    int t = sat16(x[i] * 2);
    t = (t * err) >> 16;
    t = sat16(t + 1) >> 1;
    w[i] = sat16(w[i] + t);
  }
}
#endif

class Mixer {
  std::vector<short> w_;
  int n_ = 0, ctx_ = 0, pr_ = 2048, lr_ = 1;
  const short* x_ = nullptr;
 public:
  void init(int n, int nctx, int lr, int w0) {
    n_ = n;
    lr_ = lr;
    w_.assign((size_t)n * nctx, (short)w0);
  }
  inline int mix(const short* x, int cx) {
    x_ = x;
    ctx_ = cx;
    int d = dotProduct(x, &w_[(size_t)cx * n_], n_) >> 5;
    if (d > 2047) d = 2047;
    if (d < -2047) d = -2047;
    pr_ = squash(d);
    return d;
  }
  inline void setLR(int lr8) { lr_ = lr8; }
  inline void update(int y) {
    int err = (((y << 12) - pr_) * lr_) >> 3;  // lr_ 1/8 birliklarda
    train(x_, &w_[(size_t)ctx_ * n_], n_, err);
  }
};

// ============================================================================
//  APM / SSE: ehtimollikni kontekst bo'yicha qayta xaritalash (interpolyatsiya)
// ============================================================================

class APM {
  std::vector<U16> t_;
  size_t idx_ = 0;
  int rate_ = 7;
 public:
  void init(int n, int rate) {
    rate_ = rate;
    t_.resize((size_t)n * 33);
    for (int j = 0; j < 33; ++j) t_[j] = (U16)(squash((j - 16) * 128) * 16);
    for (int i = 1; i < n; ++i) memcpy(&t_[(size_t)i * 33], &t_[0], 33 * sizeof(U16));
  }
  // d: stretch-domen (-2047..2047). Qaytaradi: 16-bit ehtimollik.
  inline int pp(int d, int cx) {
    int s = d + 2048;
    int lo = s >> 7, w = s & 127;
    size_t b = (size_t)cx * 33 + lo;
    idx_ = b + (w >> 6);
    return (t_[b] * (128 - w) + t_[b + 1] * w) >> 7;
  }
  inline void prefetchRow(int cx) const {
    const U16* r = &t_[(size_t)cx * 33];
    DZ_PREFETCH(r);
    DZ_PREFETCH(r + 32);
  }
  inline void update(int y) {
    int g = (y << 16) + (y << rate_) - y - y;
    int v = t_[idx_];
    t_[idx_] = (U16)(v + ((g - v) >> rate_));
  }
};

// ============================================================================
//  Arifmetik kodlovchi (32-bit, 16-bit ehtimollik)
// ============================================================================

class Encoder {
  std::vector<U8>& o_;
  U32 x1_ = 0, x2_ = 0xffffffffu;
 public:
  explicit Encoder(std::vector<U8>& out) : o_(out) {}
  inline void encode(int bit, int p16) {  // p16 = P(bit=1) * 65536
    U32 xm = x1_ + (U32)(((U64)(x2_ - x1_) * (U32)p16) >> 16);
    if (bit) x2_ = xm; else x1_ = xm + 1;
    while (((x1_ ^ x2_) & 0xff000000u) == 0) {
      o_.push_back((U8)(x2_ >> 24));
      x1_ <<= 8;
      x2_ = (x2_ << 8) | 255;
    }
  }
  void flush() { o_.push_back((U8)(x1_ >> 24)); }
};

class Decoder {
  const U8* p_;
  size_t n_, i_ = 0;
  U32 x1_ = 0, x2_ = 0xffffffffu, x_ = 0;
  inline U32 next() { return i_ < n_ ? p_[i_++] : 255u; }
 public:
  Decoder(const U8* in, size_t n) : p_(in), n_(n) {
    for (int k = 0; k < 4; ++k) x_ = (x_ << 8) | next();
  }
  inline int decode(int p16) {
    U32 xm = x1_ + (U32)(((U64)(x2_ - x1_) * (U32)p16) >> 16);
    int bit = x_ <= xm;
    if (bit) x2_ = xm; else x1_ = xm + 1;
    while (((x1_ ^ x2_) & 0xff000000u) == 0) {
      x1_ <<= 8;
      x2_ = (x2_ << 8) | 255;
      x_ = (x_ << 8) | next();
    }
    return bit;
  }
};

// ============================================================================
//  Bashorat modeli
// ============================================================================

static inline int lenQ(U32 l) {
  if (l < 16) return (int)l;
  if (l < 32) return 16 + (int)((l - 16) >> 2);
  if (l < 64) return 20 + (int)((l - 32) >> 3);
  if (l < 128) return 24 + (int)((l - 64) >> 4);
  if (l < 512) return 28 + (int)((l - 128) >> 7);
  return 31;
}

static inline U32 hashBytes(const U8* p, int k) {
  U32 h = 0;
  for (int i = 0; i < k; ++i) h = (h + p[i] + 1) * 0x9E3779B1u;
  h ^= h >> 15;
  h *= 0x2C1B3C6Du;
  h ^= h >> 12;
  return h;
}

class Model {
  enum {
    NH = 18,                   // xeshlangan kontekstlar soni
    NC = NH + 2,               // + order-0 va order-1 (to'g'ridan-to'g'ri jadval)
    NI = NC * 2 + 2 + 2 + 1,   // mikser kirishlari: kontekstlar, match, direct o1/o2, bias
    NIP = (NI + MIX_ALIGN - 1) / MIX_ALIGN * MIX_ALIGN,
    NM = 5,                    // birinchi qavat mikserlari
    MINS = 6,                  // match model: qisqa minimal uzunlik
    MINL = 20                  // match model: uzun minimal uzunlik
  };

  U8* buf_;
  size_t n_, pos_ = 0;
  int c0_ = 1, bitpos_ = 0, nib_ = 1, c1_ = 0, c2_ = 0;
  U32 c4_ = 0, c8_ = 0;

  // kontekst modellari
  HashTable T_;
  std::vector<U8> o0t_, o1t_;  // order-0/1 holatlari (to'g'ridan-to'g'ri)
  std::vector<U32> ssm_;       // holat -> ehtimollik (kontekst va bit o'rni bo'yicha)
  U8* bk_[NH];
  U8* sp_[NC];
  int sidx_[NC];
  U64 ch_[NH];
  int order_ = 0;

  // order-1/2 aniq ehtimollik hisoblagichlari
  std::vector<U32> d1_, d2_;
  int d1i_ = 0, d2i_ = 0;
  U32 d2h_ = 0;

  // bilvosita (indirect) kontekstlar: shu kontekstdan keyin kelgan oxirgi baytlar
  U16 t1_[256];
  std::vector<U16> t2_;
  // so'z modeli
  U64 w0_ = 0, w1_ = 0, w2_ = 0;
  // matn ustuni
  size_t lineStart_ = 0, prevLineStart_ = 0;
  // yozuv (record) modeli
  U32 rPos_[256], rDist_[256];
  U8 rRep_[256];
  U32 rCand_ = 0, rlen_ = 0;
  int rCnt_ = 0;

  // match model
  std::vector<U32> mhS_, mhL_;
  int mbits_;
  size_t mptr_ = 0;
  U32 mlen_ = 0;
  int mexp_ = 0, midx_ = -1, mbit_ = 0;
  bool mmiss_ = false;
  std::vector<U32> msm_;

  // aralashtirish
  short x_[NIP];
  Mixer mx_[NM];
  Mixer fm_;
  short xf_[MIX_ALIGN];
  APM a0_, a1_, a2_, a3_;

 public:
  Model(U8* buf, size_t n, int tbits)
      : buf_(buf), n_(n), T_(tbits), o0t_(256, 0), o1t_(65536, 0), ssm_((size_t)NC * 8 * 256, SM_INIT),
        d1_(65536, SM_INIT), d2_((size_t)1 << 22, SM_INIT), t2_(65536, 0), msm_(64, SM_INIT) {
    memset(t1_, 0, sizeof(t1_));
    memset(rPos_, 0, sizeof(rPos_));
    memset(rDist_, 0, sizeof(rDist_));
    memset(rRep_, 0, sizeof(rRep_));
    mbits_ = std::max(12, std::min(22, ceilLog2(n + 1) - 1));
    mhS_.assign((size_t)1 << mbits_, 0);
    mhL_.assign((size_t)1 << mbits_, 0);
    mx_[0].init(NIP, 65, 48, 1500);
    mx_[1].init(NIP, 256, 48, 1500);
    mx_[2].init(NIP, 256, 48, 1500);
    mx_[3].init(NIP, 64, 48, 1500);
    mx_[4].init(NIP, 256, 48, 1500);
    fm_.init(MIX_ALIGN, 8, 8, 8192 / NM);
    a0_.init(256, 6);
    a1_.init(65536, 6);
    a2_.init(65536, 6);
    a3_.init(65536 + 256, 6);
    for (int i = 0; i < NIP; ++i) x_[i] = 0;
    for (int i = 0; i < MIX_ALIGN; ++i) xf_[i] = 0;
    computeContexts();
    findBuckets(0);
    setSlots();
  }

  // Keyingi bit 1 bo'lish ehtimolligi (16-bit)
  int p() {
    int k = 0;
    for (int i = 0; i < NC; ++i) {
      int s = *sp_[i];
      int si = ((i << 3 | bitpos_) << 8) | s;
      sidx_[i] = si;
      int ps = (int)(ssm_[si] >> 20);
      x_[k++] = (short)stretch(ps);
      x_[k++] = s ? (short)((ps - 2048) >> 2) : 0;
    }
    midx_ = -1;
    int ms = 0;
    if (mlen_ > 0 && !mmiss_) {
      int eb = (mexp_ >> (7 - bitpos_)) & 1;
      mbit_ = eb;
      midx_ = lenQ(mlen_) * 2 + eb;
      ms = 1 + midx_;
      x_[k++] = (short)stretch((int)(msm_[midx_] >> 20));
      int conf = (int)std::min<U32>(mlen_, 32) * 32;
      x_[k++] = (short)(eb ? conf : -conf);
    } else {
      x_[k++] = 0;
      x_[k++] = 0;
    }
    d1i_ = (c1_ << 8) | c0_;
    d2i_ = (int)((d2h_ << 8) | (U32)c0_);
    x_[k++] = (short)stretch((int)(d1_[d1i_] >> 20));
    x_[k++] = (short)stretch((int)(d2_[d2i_] >> 20));
    x_[k++] = 256;

    xf_[0] = (short)mx_[0].mix(x_, ms);
    xf_[1] = (short)mx_[1].mix(x_, c0_);
    xf_[2] = (short)mx_[2].mix(x_, c1_);
    xf_[3] = (short)mx_[3].mix(x_, order_ * 8 + bitpos_);
    xf_[4] = (short)mx_[4].mix(x_, c2_);
    xf_[5] = 256;
    int d = fm_.mix(xf_, bitpos_);

    int pm = squash(d) * 16;
    int p0 = a0_.pp(d, c0_);
    int p1 = a1_.pp(d, c0_ | c1_ << 8);
    int p2 = a2_.pp(d, (int)((((c4_ & 0xffff) | (U32)c0_ << 16) * 0x9E3779B1u) >> 16));
    int p3 = a3_.pp(d, midx_ >= 0 ? 256 + ((mexp_ << 8) | c0_) : c0_);
    int pr = (pm + p0 + p1 * 2 + p2 * 2 + p3 * 2) >> 3;
    if (pr < 32) pr = 32;
    if (pr > 65535 - 32) pr = 65535 - 32;
    return pr;
  }

  void update(int y) {
    for (int m = 0; m < NM; ++m) mx_[m].update(y);
    fm_.update(y);
    a0_.update(y);
    a1_.update(y);
    a2_.update(y);
    a3_.update(y);
    for (int i = 0; i < NC; ++i) {
      slotUpdate(sp_[i], y);
      smUpdate(&ssm_[sidx_[i]], y, 1023);
    }
    smUpdate(&d1_[d1i_], y, 255);
    smUpdate(&d2_[d2i_], y, 255);
    if (midx_ >= 0) {
      smUpdate(&msm_[midx_], y, 1023);
      if (mbit_ != y) mmiss_ = true;
    }
    c0_ = (c0_ << 1) | y;
    nib_ = (nib_ << 1) | y;
    if (++bitpos_ == 8) {
      int c = c0_ & 255;
      buf_[pos_++] = (U8)c;
      if ((pos_ & 255) == 0) {  // mikser o'rganish tezligi asta-sekin kamayadi
        int lr = 16 + 32 * 64 / (64 + (int)(pos_ >> 10));
        for (int m = 0; m < NM; ++m) mx_[m].setLR(lr);
      }
      byteUpdate(c);
      c0_ = 1;
      bitpos_ = 0;
      nib_ = 1;
      findBuckets(0);
    } else if (bitpos_ == 4) {
      nib_ = 1;
      findBuckets(c0_);
    }
    setSlots();
    // keyingi bit uchun kerakli jadval qatorlarini oldindan keshga yuklash
    DZ_PREFETCH(&d2_[(d2h_ << 8) | (U32)c0_]);
    a1_.prefetchRow(c0_ | c1_ << 8);
    a2_.prefetchRow((int)((((c4_ & 0xffff) | (U32)c0_ << 16) * 0x9E3779B1u) >> 16));
  }

 private:
  inline void setSlots() {
    sp_[0] = &o0t_[c0_];
    sp_[1] = &o1t_[(c1_ << 8) | c0_];
    for (int i = 0; i < NH; ++i) sp_[2 + i] = bk_[i] + nib_;
  }

  void findBuckets(int nc) {
    U64 add = (U64)nc * 0x9E3779B97F4A7C15ULL;
    U64 keys[NH];
    for (int i = 0; i < NH; ++i) {
      keys[i] = hmix(ch_[i] + add);
      T_.prefetch(keys[i]);
    }
    order_ = 0;
    for (int i = 0; i < NH; ++i) {
      bool f;
      bk_[i] = T_.find(keys[i], f);
      if (i < 6 && f) ++order_;
    }
  }

  void byteUpdate(int c) {
    t1_[c1_] = (U16)(t1_[c1_] << 8 | c);
    t2_[c4_ & 0xffff] = (U16)(t2_[c4_ & 0xffff] << 8 | c);
    c8_ = (c8_ << 8) | (c4_ >> 24);
    c4_ = (c4_ << 8) | (U32)c;
    c2_ = c1_;
    c1_ = c;
    // so'zlar (harflar va UTF-8 baytlari; katta-kichik harf farqlanmaydi)
    bool isw = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 128;
    if (isw) {
      int lc = (c >= 'A' && c <= 'Z') ? c + 32 : c;
      w0_ = (w0_ + (U64)lc + 1) * 0x2F0F3ACD1B3ULL + 0x9E37ULL;
    } else if (w0_) {
      w2_ = w1_;
      w1_ = w0_;
      w0_ = 0;
    }
    // qatorlar
    if (c == '\n') {
      prevLineStart_ = lineStart_;
      lineStart_ = pos_;
    }
    recordUpdate(c);
    matchUpdate();
    computeContexts();
  }

  // Takrorlanuvchi yozuv uzunligini (jadvallar, tuzilmali binar) aniqlash
  void recordUpdate(int c) {
    U32 p1 = (U32)(pos_ - 1);
    U32 d = p1 - rPos_[c];
    if (d == rDist_[c]) {
      if (rRep_[c] < 255) ++rRep_[c];
    } else {
      rDist_[c] = d;
      rRep_[c] = 0;
    }
    rPos_[c] = p1;
    if (rRep_[c] >= 2 && d >= 2 && d <= 0x10000) {
      if (d == rCand_) {
        if (rCnt_ < 64) ++rCnt_;
      } else if (--rCnt_ <= 0) {
        rCand_ = d;
        rCnt_ = 1;
      }
      if (rCnt_ >= 8) rlen_ = rCand_;
    }
  }

  // Oldingi eng uzun mos keluvchi ketma-ketlikni topish
  void matchUpdate() {
    if (mlen_ > 0) {
      if (buf_[mptr_] == (U8)c1_) {
        ++mptr_;
        if (mlen_ < 65535) ++mlen_;
      } else {
        mlen_ = 0;
      }
    }
    mmiss_ = false;
    if (pos_ >= (size_t)MINL) {
      U32 sh = 32 - mbits_;
      U32 hS = hashBytes(buf_ + pos_ - MINS, MINS) >> sh;
      U32 hL = hashBytes(buf_ + pos_ - MINL, MINL) >> sh;
      if (mlen_ == 0) {
        U32 cand = mhL_[hL];
        if (cand) {
          U32 l = 0;
          while (l < 400 && l < cand && buf_[cand - 1 - l] == buf_[pos_ - 1 - l]) ++l;
          if (l >= (U32)MINL) {
            mlen_ = l;
            mptr_ = cand;
          }
        }
        if (!mlen_) {
          cand = mhS_[hS];
          if (cand) {
            U32 l = 0;
            while (l < 400 && l < cand && buf_[cand - 1 - l] == buf_[pos_ - 1 - l]) ++l;
            if (l >= (U32)MINS) {
              mlen_ = l;
              mptr_ = cand;
            }
          }
        }
      }
      mhL_[hL] = (U32)pos_;
      mhS_[hS] = (U32)pos_;
    }
    mexp_ = mlen_ ? buf_[mptr_] : 0;
  }

  void computeContexts() {
    d2h_ = (U32)(((c4_ & 0xffff) * 0x9E3779B1u) >> 18);
    ch_[0] = hctx(1, c4_ & 0xffff);                                // order-2
    ch_[1] = hctx(2, c4_ & 0xffffff);                              // order-3
    ch_[2] = hctx(3, c4_);                                         // order-4
    ch_[3] = hctx(4, (U64)c4_ | (U64)(c8_ & 0xff) << 32);          // order-5
    ch_[4] = hctx(5, (U64)c4_ | (U64)(c8_ & 0xffff) << 32);        // order-6
    ch_[5] = hctx(6, (U64)c4_ | (U64)c8_ << 32);                   // order-8
    ch_[6] = w0_ ? hctx(7, w0_) : hctx(8, w1_ + (U64)c1_ * 0x100000001B3ULL);  // joriy so'z
    ch_[7] = hctx(9, w0_ + w1_ * 0x9E3779B97F4A7C15ULL);           // so'z + oldingi so'z
    ch_[8] = hctx(10, w0_ + w2_ * 0xC2B2AE3D27D4EB4FULL);          // so'z + undan oldingi
    ch_[9] = hctx(11, c4_ & 0xff00);                               // siyrak: 2-bayt
    ch_[10] = hctx(12, c4_ & 0xffff00);                            // siyrak: 2,3-baytlar
    ch_[11] = hctx(13, c4_ & 0xffff0000);                          // siyrak: 3,4-baytlar
    if (rlen_ && pos_ >= 2 * (size_t)rlen_) {                      // yozuv modeli
      U64 a = buf_[pos_ - rlen_], b = buf_[pos_ - 2 * rlen_];
      ch_[12] = hctx(14, a | b << 8 | (U64)rlen_ << 16);
      ch_[13] = hctx(15, a | (U64)c1_ << 8 | (U64)rlen_ << 16);
    } else {
      ch_[12] = hctx(16, c4_ & 0xff00ff);
      ch_[13] = hctx(17, c4_ & 0xff0000ff);
    }
    size_t col = pos_ - lineStart_;                                // matn ustuni
    size_t ab = prevLineStart_ + col;
    U64 above = ab < lineStart_ ? buf_[ab] : 0;
    ch_[14] = hctx(18, above | (U64)c1_ << 8);
    ch_[15] = hctx(19, above | (U64)std::min<size_t>(col, 255) << 8);
    ch_[16] = hctx(20, (U64)c1_ | (U64)t1_[c1_] << 8);             // bilvosita order-1
    ch_[17] = hctx(21, (U64)(c4_ & 0xffff) | (U64)t2_[c4_ & 0xffff] << 16);  // bilvosita order-2
  }
};

// ============================================================================
//  Blok siqish / ochish
// ============================================================================

static int tableBitsFor(int level, size_t n) {
  int lb = 21 + level;                       // 1 -> 4 MB ... 9 -> 1 GB
  int nb = std::max(20, ceilLog2(n + 1) + 6);
  return std::min(lb, nb);
}

static void cmEncode(U8* data, size_t n, int tbits, std::vector<U8>& out, std::atomic<U64>* prog) {
  Model m(data, n, tbits);
  Encoder enc(out);
  for (size_t i = 0; i < n; ++i) {
    int c = data[i];
    for (int j = 7; j >= 0; --j) {
      int bit = (c >> j) & 1;
      enc.encode(bit, m.p());
      m.update(bit);
    }
    if (prog && ((i + 1) & 0xffff) == 0) *prog += 0x10000;
  }
  enc.flush();
  if (prog) *prog += n & 0xffff;
}

static void cmDecode(const U8* in, size_t inLen, U8* out, size_t n, int tbits, std::atomic<U64>* prog) {
  Model m(out, n, tbits);
  Decoder dec(in, inLen);
  for (size_t i = 0; i < n; ++i) {
    for (int j = 0; j < 8; ++j) m.update(dec.decode(m.p()));
    if (prog && ((i + 1) & 0xffff) == 0) *prog += 0x10000;
  }
  if (prog) *prog += n & 0xffff;
}

// ============================================================================
//  x86 E8/E9 filtri (CALL/JMP manzillarini nisbiydan mutlaqqa aylantirish)
//  [-2^24, 2^24) oralig'ini o'ziga biektiv aylantiradi => to'liq qaytariladi.
// ============================================================================

static inline U32 rd32(const U8* p) { return p[0] | p[1] << 8 | p[2] << 16 | (U32)p[3] << 24; }
static inline void wr32(U8* p, U32 v) {
  p[0] = (U8)v; p[1] = (U8)(v >> 8); p[2] = (U8)(v >> 16); p[3] = (U8)(v >> 24);
}

static void x86Filter(U8* b, size_t n, bool encode) {
  for (size_t i = 0; i + 5 <= n;) {
    if ((b[i] & 0xFE) == 0xE8) {
      U32 a = rd32(b + i + 1);
      if (a + 0x01000000u < 0x02000000u) {
        U32 off = (U32)i;
        if (encode) a = ((a + 0x01000000u + off) & 0x01FFFFFFu) - 0x01000000u;
        else        a = ((a + 0x01000000u - off) & 0x01FFFFFFu) - 0x01000000u;
        wr32(b + i + 1, a);
      }
      i += 5;
    } else {
      ++i;
    }
  }
}

// ============================================================================
//  Arxiv formati
//
//  [Sarlavha 16 bayt] "DRGZ" ver level 0 0  +8 bayt zaxira
//  [Bloklar ...]      'B' method tbits 0 | crc32 | rawSize(8) | compSize(8) | ma'lumot
//  [Katalog bloki]    xuddi shu formatda, 'D' belgisi bilan
//  [Yakun 24 bayt]    dirOffset(8) | blockCount(4) | 0(4) | "DRGZ-END"
// ============================================================================

enum : U8 { METHOD_STORE = 0, METHOD_CM = 1 };
enum : U8 { FLAG_X86 = 1, FLAG_STORED = 2 };
static const U64 BLOCK_HDR = 28;  // 24 bayt ma'lumot + sarlavhaning o'z CRC32 si
static const char TRAILER_MAGIC[9] = "DRGZ-END";

struct Entry {
  std::string path;  // arxivdagi nisbiy yo'l, '/' bilan
  U8 type = 0;       // 0 = fayl, 1 = papka
  U8 flags = 0;
  U64 size = 0;
  I64 mtime = 0;
  U32 mode = 0;
  U32 crc = 0;
  fs::path src;      // faqat siqishda
};

static void putVar(std::vector<U8>& o, U64 v) {
  while (v >= 128) { o.push_back((U8)(v | 128)); v >>= 7; }
  o.push_back((U8)v);
}
static U64 getVar(const std::vector<U8>& d, size_t& i) {
  U64 v = 0;
  for (int s = 0; s < 64; s += 7) {
    if (i >= d.size()) throw std::runtime_error("katalog buzilgan");
    U8 b = d[i++];
    v |= (U64)(b & 127) << s;
    if (!(b & 128)) return v;
  }
  throw std::runtime_error("katalog buzilgan");
}

static std::vector<U8> serializeDir(const std::vector<Entry>& es) {
  std::vector<U8> o;
  o.push_back('D'); o.push_back('1');
  putVar(o, es.size());
  for (const Entry& e : es) {
    o.push_back(e.type);
    o.push_back(e.flags);
    putVar(o, e.path.size());
    o.insert(o.end(), e.path.begin(), e.path.end());
    putVar(o, e.size);
    putVar(o, (U64)((e.mtime << 1) ^ (e.mtime >> 63)));
    putVar(o, e.mode);
    for (int k = 0; k < 4; ++k) o.push_back((U8)(e.crc >> (8 * k)));
  }
  return o;
}

static std::vector<Entry> parseDir(const std::vector<U8>& d) {
  if (d.size() < 3 || d[0] != 'D' || d[1] != '1') throw std::runtime_error("katalog formati noma'lum");
  size_t i = 2;
  U64 cnt = getVar(d, i);
  if (cnt > d.size()) throw std::runtime_error("katalog buzilgan");
  std::vector<Entry> es((size_t)cnt);
  for (Entry& e : es) {
    if (i + 2 > d.size()) throw std::runtime_error("katalog buzilgan");
    e.type = d[i++];
    e.flags = d[i++];
    U64 pl = getVar(d, i);
    if (pl > d.size() - i) throw std::runtime_error("katalog buzilgan");
    e.path.assign((const char*)&d[i], (size_t)pl);
    i += (size_t)pl;
    e.size = getVar(d, i);
    U64 z = getVar(d, i);
    e.mtime = (I64)(z >> 1) ^ -(I64)(z & 1);
    e.mode = (U32)getVar(d, i);
    if (i + 4 > d.size()) throw std::runtime_error("katalog buzilgan");
    e.crc = d[i] | d[i + 1] << 8 | d[i + 2] << 16 | (U32)d[i + 3] << 24;
    i += 4;
  }
  return es;
}

// ============================================================================
//  Yordamchi: vaqt, formatlash, progress
// ============================================================================

// Fayl tizimi soati va tizim soati orasidagi farq (har doim butun soniyalar)
static std::chrono::seconds fileClockOffset() {
  using namespace std::chrono;
  static const seconds off = [] {
    auto f = duration_cast<microseconds>(fs::file_time_type::clock::now().time_since_epoch());
    auto s = duration_cast<microseconds>(system_clock::now().time_since_epoch());
    return round<seconds>(f - s);
  }();
  return off;
}
static I64 toUnixTime(fs::file_time_type ft) {
  using namespace std::chrono;
  return (I64)(floor<seconds>(ft.time_since_epoch()) - fileClockOffset()).count();
}
static fs::file_time_type fromUnixTime(I64 t) {
  using namespace std::chrono;
  return fs::file_time_type(duration_cast<fs::file_time_type::duration>(seconds(t) + fileClockOffset()));
}

static std::string fmtSize(U64 b) {
  char s[64];
  if (b < 1024) snprintf(s, sizeof s, "%llu B", (unsigned long long)b);
  else if (b < 1024 * 1024) snprintf(s, sizeof s, "%.1f KB", b / 1024.0);
  else if (b < (U64)1024 * 1024 * 1024) snprintf(s, sizeof s, "%.2f MB", b / 1048576.0);
  else snprintf(s, sizeof s, "%.2f GB", b / 1073741824.0);
  return s;
}
static std::string fmtTime(double sec) {
  char s[64];
  if (sec < 60) snprintf(s, sizeof s, "%.1fs", sec);
  else if (sec < 3600) snprintf(s, sizeof s, "%dm%02ds", (int)sec / 60, (int)sec % 60);
  else snprintf(s, sizeof s, "%dh%02dm", (int)sec / 3600, ((int)sec / 60) % 60);
  return s;
}

static bool g_quiet = false;

// Ctrl+C bosilganda chala arxiv faylini o'chirish
static char g_tmpPath[4096] = {0};
extern "C" void onSignal(int) {
  if (g_tmpPath[0]) std::remove(g_tmpPath);
  std::_Exit(130);
}

class Progress {
  const char* label_;
  U64 total_;
  std::chrono::steady_clock::time_point t0_;
  bool tty_;
 public:
  std::atomic<U64> done{0};
  Progress(const char* label, U64 total)
      : label_(label), total_(total), t0_(std::chrono::steady_clock::now()), tty_(DZ_ISATTY(2) != 0) {}
  void show(bool final = false) {
    if (g_quiet || total_ == 0) return;
    if (!tty_ && !final) return;
    double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
    U64 d = std::min<U64>(done.load(), total_);
    double frac = (double)d / (double)total_;
    int bars = (int)(frac * 30);
    char bar[32];
    for (int i = 0; i < 30; ++i) bar[i] = i < bars ? '#' : '.';
    bar[30] = 0;
    double speed = el > 0 ? d / el / 1048576.0 : 0;
    double eta = (frac > 0.001 && !final) ? el / frac - el : 0;
    fprintf(stderr, "\r  %s [%s] %5.1f%%  %6.2f MB/s  %s %-8s", label_, bar, frac * 100, speed,
            final ? "vaqt" : "qoldi", fmtTime(final ? el : eta).c_str());
    if (final) fprintf(stderr, "\n");
    fflush(stderr);
  }
};

// Ishlarni parallel bajarish va progressni ko'rsatish
template <class F>
static void runParallel(size_t count, F fn, Progress* prog) {
  std::vector<std::thread> th;
  std::atomic<size_t> finished{0};
  std::vector<std::exception_ptr> errs(count);
  for (size_t i = 0; i < count; ++i) {
    th.emplace_back([&, i]() {
      try { fn(i); } catch (...) { errs[i] = std::current_exception(); }
      ++finished;
    });
  }
  while (finished.load() < count) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    if (prog) prog->show();
  }
  for (auto& t : th) t.join();
  for (auto& e : errs)
    if (e) std::rethrow_exception(e);
}

// ============================================================================
//  Fayllarni tahlil qilish
// ============================================================================

static bool isX86Exe(const U8* h, size_t n) {
  if (n >= 64 && h[0] == 'M' && h[1] == 'Z') return true;
  if (n >= 20 && h[0] == 0x7f && h[1] == 'E' && h[2] == 'L' && h[3] == 'F') {
    int mach = h[18] | h[19] << 8;
    return mach == 3 || mach == 62;
  }
  return false;
}

// Fayl allaqachon siqilganmi (jpg, png, zip, mp4, ...)? Namuna entropiyasi bilan.
static bool looksIncompressible(const fs::path& p, U64 size) {
  const size_t CH = 65536;
  if (size < 2 * CH) return false;
  std::ifstream in(p, std::ios::binary);
  if (!in) return false;
  U64 cnt[256] = {0};
  U64 total = 0, rep = 0, probes = 0;
  std::vector<U8> b(CH);
  std::vector<U8> seen((size_t)1 << 20, 0);  // 4 baytli xeshlar (takrorlarni topish uchun)
  int samples = (int)std::min<U64>(8, size / CH);
  for (int s = 0; s < samples; ++s) {
    U64 off = samples > 1 ? (size - CH) / (U64)(samples - 1) * (U64)s : 0;
    in.clear();
    in.seekg((std::streamoff)off);
    in.read((char*)b.data(), (std::streamsize)CH);
    size_t got = (size_t)in.gcount();
    for (size_t i = 0; i < got; ++i) cnt[b[i]]++;
    total += got;
    for (size_t i = 0; i + 4 <= got; i += 2) {
      U32 h = (rd32(&b[i]) * 0x9E3779B1u) >> 12;
      if (seen[h]) ++rep;
      seen[h] = 1;
      ++probes;
    }
  }
  if (total < CH) return false;
  double H = 0;
  for (int i = 0; i < 256; ++i)
    if (cnt[i]) {
      double q = (double)cnt[i] / (double)total;
      H -= q * std::log2(q);
    }
  // Tasodifiy (siqilgan/shifrlangan) ma'lumot: entropiya ~8 bit/bayt va
  // 4-baytli ketma-ketliklar deyarli takrorlanmaydi.
  double repRate = probes ? (double)rep / (double)probes : 0;
  return H > 7.97 && repRate < 0.15;
}

// ============================================================================
//  Siqish (arxiv yaratish)
// ============================================================================

struct Options {
  int level = 6;
  int threads = 0;
  U64 blockMB = 0;
  bool force = false;
  bool detect = true;
  std::string outDir;
};

class SolidReader {
  std::vector<Entry*> files_;
  size_t fi_ = 0;
  U64 left_ = 0;
  bool open_ = false;
  std::ifstream in_;
  std::vector<U8> mem_;
  size_t memPos_ = 0;
  bool useMem_ = false;
  U32 crc_ = 0;

  void openNext() {
    while (fi_ < files_.size() && files_[fi_]->size == 0) {
      files_[fi_]->crc = 0;
      ++fi_;
    }
    if (fi_ >= files_.size()) throw std::runtime_error("ichki xato: ma'lumot yetishmadi");
    Entry* e = files_[fi_];
    left_ = e->size;
    crc_ = 0;
    useMem_ = (e->flags & FLAG_X86) != 0;
    in_.close();
    in_.clear();
    in_.open(e->src, std::ios::binary);
    if (!in_) throw std::runtime_error("faylni ochib bo'lmadi: " + e->src.u8string());
    if (useMem_) {
      mem_.resize((size_t)e->size);
      in_.read((char*)mem_.data(), (std::streamsize)e->size);
      if ((U64)in_.gcount() != e->size) throw std::runtime_error("fayl o'qishda xato (o'zgargan?): " + e->src.u8string());
      crc_ = crc32Update(0, mem_.data(), mem_.size());
      x86Filter(mem_.data(), mem_.size(), true);
      memPos_ = 0;
      in_.close();
    }
    open_ = true;
  }

 public:
  explicit SolidReader(std::vector<Entry*> files) : files_(std::move(files)) {}
  void read(U8* dst, size_t want) {
    while (want > 0) {
      if (!open_) openNext();
      size_t take = (size_t)std::min<U64>(want, left_);
      if (useMem_) {
        memcpy(dst, mem_.data() + memPos_, take);
        memPos_ += take;
      } else {
        in_.read((char*)dst, (std::streamsize)take);
        if ((size_t)in_.gcount() != take)
          throw std::runtime_error("fayl o'qishda xato (o'zgargan?): " + files_[fi_]->src.u8string());
        crc_ = crc32Update(crc_, dst, take);
      }
      dst += take;
      want -= take;
      left_ -= take;
      if (left_ == 0) {
        files_[fi_]->crc = crc_;
        open_ = false;
        in_.close();
        mem_.clear();
        mem_.shrink_to_fit();
        ++fi_;
      }
    }
  }
  void finish() {
    while (fi_ < files_.size() && files_[fi_]->size == 0) files_[fi_++]->crc = 0;
  }
};

struct Job {
  std::vector<U8> raw, comp;
  U8 method = METHOD_CM;
  int tbits = 20;
  U32 crc = 0;
};

static void writeBlock(std::ofstream& out, char tag, U8 method, int tbits, U32 crc,
                       const std::vector<U8>& raw, const std::vector<U8>* comp) {
  U8 h[BLOCK_HDR];
  h[0] = (U8)tag; h[1] = method; h[2] = (U8)tbits; h[3] = 0;
  wr32(h + 4, crc);
  U64 rs = raw.size(), cs = comp ? comp->size() : raw.size();
  for (int k = 0; k < 8; ++k) { h[8 + k] = (U8)(rs >> (8 * k)); h[16 + k] = (U8)(cs >> (8 * k)); }
  wr32(h + 24, crc32Update(0, h, 24));
  out.write((const char*)h, BLOCK_HDR);
  const std::vector<U8>& d = comp ? *comp : raw;
  out.write((const char*)d.data(), (std::streamsize)d.size());
}

static std::string lowerExt(const std::string& p) {
  size_t s = p.find_last_of('/');
  std::string name = s == std::string::npos ? p : p.substr(s + 1);
  size_t d = name.find_last_of('.');
  if (d == std::string::npos || d == 0) return "";
  std::string e = name.substr(d + 1);
  for (char& c : e) c = (char)tolower((unsigned char)c);
  return e;
}

static int cmdCreate(const std::string& archive, const std::vector<std::string>& inputs, const Options& opt) {
  auto t0 = std::chrono::steady_clock::now();
  fs::path arcPath = fs::u8path(archive);
  if (fs::exists(arcPath) && !opt.force) {
    fprintf(stderr, "Xato: '%s' allaqachon mavjud. Ustidan yozish uchun -f qo'shing.\n", archive.c_str());
    return 1;
  }
  fs::path arcCanon = fs::weakly_canonical(fs::absolute(arcPath));

  // 1) Fayllarni yig'ish
  std::vector<Entry> entries;
  for (const std::string& in : inputs) {
    fs::path p = fs::u8path(in);
    std::error_code ec;
    if (!fs::exists(p, ec)) {
      fprintf(stderr, "Ogohlantirish: topilmadi: %s\n", in.c_str());
      continue;
    }
    fs::path abs = fs::weakly_canonical(fs::absolute(p));
    fs::path base = abs.parent_path();
    auto addOne = [&](const fs::path& f, bool isDir) {
      Entry e;
      e.src = f;
      e.path = fs::relative(f, base).generic_u8string();
      if (e.path.empty() || e.path == ".") return;
      e.type = isDir ? 1 : 0;
      std::error_code ec2;
      e.mtime = toUnixTime(fs::last_write_time(f, ec2));
      e.mode = (U32)fs::status(f, ec2).permissions() & 07777;
      if (!isDir) e.size = fs::file_size(f, ec2);
      if (ec2) {
        fprintf(stderr, "Ogohlantirish: o'qib bo'lmadi: %s\n", f.u8string().c_str());
        return;
      }
      entries.push_back(std::move(e));
    };
    if (fs::is_directory(abs)) {
      addOne(abs, true);
      for (auto it = fs::recursive_directory_iterator(abs, fs::directory_options::skip_permission_denied, ec);
           it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (ec) break;
        const fs::path& f = it->path();
        std::error_code ec3;
        if (it->is_directory(ec3)) addOne(f, true);
        else if (it->is_regular_file(ec3)) {
          if (fs::weakly_canonical(f, ec3) == arcCanon) continue;
          addOne(f, false);
        }
      }
    } else if (fs::is_regular_file(abs)) {
      if (abs == arcCanon) continue;
      addOne(abs, false);
    }
  }
  // takrorlarni olib tashlash
  std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) { return a.path < b.path; });
  entries.erase(std::unique(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) { return a.path == b.path; }),
                entries.end());
  if (entries.empty()) {
    fprintf(stderr, "Xato: arxivlash uchun hech narsa topilmadi.\n");
    return 1;
  }

  // 2) Tasniflash
  U64 cmTotal = 0, stTotal = 0;
  size_t nFiles = 0, nDirs = 0, nX86 = 0, nStored = 0;
  for (Entry& e : entries) {
    if (e.type == 1) { ++nDirs; continue; }
    ++nFiles;
    if (e.size == 0) continue;
    U8 head[64] = {0};
    {
      std::ifstream f(e.src, std::ios::binary);
      f.read((char*)head, sizeof head);
      size_t got = (size_t)f.gcount();
      if (isX86Exe(head, got) && e.size <= ((U64)1 << 30)) { e.flags |= FLAG_X86; ++nX86; }
    }
    if (opt.detect && !(e.flags & FLAG_X86) && looksIncompressible(e.src, e.size)) {
      e.flags |= FLAG_STORED;
      ++nStored;
    }
    if (e.flags & FLAG_STORED) stTotal += e.size; else cmTotal += e.size;
  }

  // 3) Tartiblash: papkalar, siqiladigan fayllar (kengaytma bo'yicha), saqlanadiganlar
  auto rank = [](const Entry& e) { return e.type == 1 ? 0 : (e.flags & FLAG_STORED) ? 2 : 1; };
  std::stable_sort(entries.begin(), entries.end(), [&](const Entry& a, const Entry& b) {
    int ra = rank(a), rb = rank(b);
    if (ra != rb) return ra < rb;
    if (ra == 0) return a.path < b.path;
    std::string ea = lowerExt(a.path), eb = lowerExt(b.path);
    if (ea != eb) return ea < eb;
    return a.path < b.path;
  });

  std::vector<Entry*> cmFiles, stFiles;
  for (Entry& e : entries) {
    if (e.type == 1) continue;
    if (e.flags & FLAG_STORED) stFiles.push_back(&e); else cmFiles.push_back(&e);
  }

  int threads = opt.threads > 0 ? opt.threads : (int)std::min<unsigned>(4, std::max(1u, std::thread::hardware_concurrency()));
  U64 bs;
  if (opt.blockMB) bs = opt.blockMB << 20;
  else {
    U64 per = (cmTotal + threads - 1) / threads;
    bs = std::max<U64>(per, (U64)64 << 20);
    bs = std::min<U64>(bs, (U64)768 << 20);
  }
  bs = std::max<U64>(bs, 1 << 16);
  bs = std::min<U64>(bs, (U64)2047 << 20);

  if (!g_quiet) {
    fprintf(stderr, "DragonZip %s — kuchli yo'qotishsiz arxivator\n", DZ_VERSION);
    fprintf(stderr, "  Fayllar: %zu ta, papkalar: %zu ta, jami: %s\n", nFiles, nDirs, fmtSize(cmTotal + stTotal).c_str());
    if (nStored) fprintf(stderr, "  Allaqachon siqilgan (saqlanadi): %zu ta, %s\n", nStored, fmtSize(stTotal).c_str());
    if (nX86) fprintf(stderr, "  Dastur fayllari (x86 filtri): %zu ta\n", nX86);
    U64 nblocks = cmTotal ? (cmTotal + bs - 1) / bs : 0;
    int useT = (int)std::min<U64>((U64)threads, std::max<U64>(1, nblocks));
    int tb = tableBitsFor(opt.level, (size_t)std::min<U64>(bs, cmTotal));
    fprintf(stderr, "  Daraja: %d, oqimlar: %d, bloklar: %llu x %s, xotira: ~%s\n", opt.level, useT,
            (unsigned long long)nblocks, fmtSize(std::min<U64>(bs, cmTotal)).c_str(),
            fmtSize((U64)useT * (((U64)1 << tb) + 64 * 1048576 + std::min<U64>(bs, cmTotal) * 3 / 2)).c_str());
  }

  // 4) Yozish
  fs::path tmpPath = arcPath;
  tmpPath += ".tmp";
  std::ofstream out(tmpPath, std::ios::binary | std::ios::trunc);
  if (!out) {
    fprintf(stderr, "Xato: arxiv faylini yaratib bo'lmadi: %s\n", archive.c_str());
    return 1;
  }
  {
    std::string tp = tmpPath.string();
    if (tp.size() < sizeof(g_tmpPath)) memcpy(g_tmpPath, tp.c_str(), tp.size() + 1);
    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);
  }
  U8 hdr[16] = {'D', 'R', 'G', 'Z', 1, (U8)opt.level, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  out.write((const char*)hdr, 16);
  U32 blockCount = 0;

  try {
    Progress prog("Siqilmoqda ", cmTotal + stTotal);
    SolidReader rcm(cmFiles);
    U64 remain = cmTotal;
    while (remain > 0) {
      std::vector<Job> jobs;
      while ((int)jobs.size() < threads && remain > 0) {
        Job j;
        size_t sz = (size_t)std::min<U64>(bs, remain);
        j.raw.resize(sz);
        rcm.read(j.raw.data(), sz);
        j.crc = crc32Update(0, j.raw.data(), sz);
        j.tbits = tableBitsFor(opt.level, sz);
        remain -= sz;
        jobs.push_back(std::move(j));
      }
      runParallel(jobs.size(), [&](size_t i) {
        Job& j = jobs[i];
        j.comp.reserve(j.raw.size() / 2 + 1024);
        cmEncode(j.raw.data(), j.raw.size(), j.tbits, j.comp, &prog.done);
      }, &prog);
      for (Job& j : jobs) {
        if (j.comp.size() >= j.raw.size()) writeBlock(out, 'B', METHOD_STORE, 0, j.crc, j.raw, nullptr);
        else writeBlock(out, 'B', METHOD_CM, j.tbits, j.crc, j.raw, &j.comp);
        ++blockCount;
      }
    }
    rcm.finish();
    SolidReader rst(stFiles);
    remain = stTotal;
    while (remain > 0) {
      std::vector<U8> raw((size_t)std::min<U64>(remain, (U64)32 << 20));
      rst.read(raw.data(), raw.size());
      writeBlock(out, 'B', METHOD_STORE, 0, crc32Update(0, raw.data(), raw.size()), raw, nullptr);
      prog.done += raw.size();
      remain -= raw.size();
      ++blockCount;
    }
    rst.finish();
    prog.show(true);

    // katalog
    U64 dirOff = (U64)out.tellp();
    std::vector<U8> dir = serializeDir(entries);
    std::vector<U8> dc;
    int dtb = tableBitsFor(1, dir.size());
    std::vector<U8> dcopy = dir;
    cmEncode(dcopy.data(), dcopy.size(), dtb, dc, nullptr);
    U32 dcrc = crc32Update(0, dir.data(), dir.size());
    if (dc.size() >= dir.size()) writeBlock(out, 'D', METHOD_STORE, 0, dcrc, dir, nullptr);
    else writeBlock(out, 'D', METHOD_CM, dtb, dcrc, dir, &dc);

    U8 tr[24];
    for (int k = 0; k < 8; ++k) tr[k] = (U8)(dirOff >> (8 * k));
    wr32(tr + 8, blockCount);
    wr32(tr + 12, crc32Update(0, tr, 12));
    memcpy(tr + 16, TRAILER_MAGIC, 8);
    out.write((const char*)tr, 24);
    out.close();
    if (!out) throw std::runtime_error("arxivni yozishda xato (disk to'lganmi?)");
    std::error_code ec;
    fs::rename(tmpPath, arcPath, ec);
    if (ec) {
      fs::remove(arcPath, ec);
      fs::rename(tmpPath, arcPath);
    }
    g_tmpPath[0] = 0;
  } catch (...) {
    out.close();
    std::error_code ec;
    fs::remove(tmpPath, ec);
    g_tmpPath[0] = 0;
    throw;
  }

  U64 inSize = cmTotal + stTotal;
  U64 arcSize = fs::file_size(arcPath);
  double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  if (!g_quiet) {
    fprintf(stderr, "Tayyor: %s -> %s", fmtSize(inSize).c_str(), fmtSize(arcSize).c_str());
    if (inSize) fprintf(stderr, "  (%.2f%%, %.3f bit/bayt, %.2fx kichik)", 100.0 * arcSize / inSize,
                        8.0 * arcSize / inSize, (double)inSize / std::max<U64>(arcSize, 1));
    fprintf(stderr, "  vaqt: %s\n", fmtTime(el).c_str());
  }
  return 0;
}

// ============================================================================
//  Arxivni o'qish
// ============================================================================

struct BlockInfo {
  char tag;
  U8 method;
  int tbits;
  U32 crc;
  U64 raw, comp, offset;  // offset = ma'lumot boshlanishi
};

class ArchiveReader {
 public:
  std::ifstream in;
  U64 fileSize = 0, dirOff = 0;
  U32 blockCount = 0;
  std::vector<Entry> entries;

  explicit ArchiveReader(const std::string& name) {
    fs::path p = fs::u8path(name);
    in.open(p, std::ios::binary);
    if (!in) throw std::runtime_error("arxivni ochib bo'lmadi: " + name);
    fileSize = fs::file_size(p);
    if (fileSize < 16 + 24) throw std::runtime_error("bu DragonZip arxivi emas (juda kichik)");
    U8 h[16];
    readAt(0, h, 16);
    if (memcmp(h, "DRGZ", 4) != 0) throw std::runtime_error("bu DragonZip arxivi emas");
    if (h[4] != 1) throw std::runtime_error("arxiv versiyasi qo'llab-quvvatlanmaydi");
    U8 tr[24];
    readAt(fileSize - 24, tr, 24);
    if (memcmp(tr + 16, TRAILER_MAGIC, 8) != 0) throw std::runtime_error("arxiv oxiri buzilgan yoki to'liq emas");
    for (int k = 0; k < 8; ++k) dirOff |= (U64)tr[k] << (8 * k);
    blockCount = rd32(tr + 8);
    if (rd32(tr + 12) != crc32Update(0, tr, 12)) throw std::runtime_error("arxiv oxiri buzilgan (CRC)");
    if (dirOff < 16 || dirOff + BLOCK_HDR > fileSize - 24) throw std::runtime_error("arxiv buzilgan");
    BlockInfo bi = readHeader(dirOff);
    if (bi.tag != 'D') throw std::runtime_error("katalog topilmadi");
    if (bi.raw > std::max<U64>((U64)1 << 20, bi.comp * 4096)) throw std::runtime_error("katalog sarlavhasi buzilgan");
    std::vector<U8> dir = decodeBlock(bi, nullptr);
    entries = parseDir(dir);
  }

  void readAt(U64 off, U8* dst, size_t n) {
    in.clear();
    in.seekg((std::streamoff)off);
    in.read((char*)dst, (std::streamsize)n);
    if ((size_t)in.gcount() != n) throw std::runtime_error("arxivni o'qishda xato (buzilgan?)");
  }

  BlockInfo readHeader(U64 off) {
    U8 h[BLOCK_HDR];
    readAt(off, h, BLOCK_HDR);
    if (rd32(h + 24) != crc32Update(0, h, 24)) throw std::runtime_error("blok sarlavhasi buzilgan (CRC)");
    BlockInfo b;
    b.tag = (char)h[0];
    b.method = h[1];
    b.tbits = h[2];
    b.crc = rd32(h + 4);
    b.raw = b.comp = 0;
    for (int k = 0; k < 8; ++k) {
      b.raw |= (U64)h[8 + k] << (8 * k);
      b.comp |= (U64)h[16 + k] << (8 * k);
    }
    b.offset = off + BLOCK_HDR;
    if ((b.tag != 'B' && b.tag != 'D') || b.method > METHOD_CM || b.comp > fileSize || b.offset + b.comp > fileSize ||
        b.raw > ((U64)4 << 30) || (b.method == METHOD_CM && (b.tbits < 20 || b.tbits > 30)) ||
        (b.method == METHOD_STORE && b.raw != b.comp))
      throw std::runtime_error("blok sarlavhasi buzilgan");
    return b;
  }

  std::vector<U8> decodeBlock(const BlockInfo& b, std::atomic<U64>* prog) {
    std::vector<U8> comp((size_t)b.comp);
    readAt(b.offset, comp.data(), comp.size());
    return decodeData(b, comp, prog);
  }

  static std::vector<U8> decodeData(const BlockInfo& b, const std::vector<U8>& comp, std::atomic<U64>* prog) {
    std::vector<U8> raw;
    if (b.method == METHOD_STORE) {
      raw = comp;
      if (prog) *prog += raw.size();
    } else {
      raw.resize((size_t)b.raw);
      cmDecode(comp.data(), comp.size(), raw.data(), raw.size(), b.tbits, prog);
    }
    if (crc32Update(0, raw.data(), raw.size()) != b.crc) throw std::runtime_error("CRC xatosi: blok buzilgan");
    return raw;
  }
};

static bool safePath(const std::string& p) {
  if (p.empty() || p[0] == '/' || p[0] == '\\') return false;
  if (p.size() >= 2 && p[1] == ':') return false;
  if (p.find('\\') != std::string::npos) return false;
  size_t s = 0;
  while (s <= p.size()) {
    size_t e = p.find('/', s);
    if (e == std::string::npos) e = p.size();
    std::string part = p.substr(s, e - s);
    if (part == ".." || part == "." || part.empty()) return false;
    s = e + 1;
  }
  return true;
}

// Ochilgan oqimni fayllarga taqsimlaydi va CRC ni tekshiradi
class FileSink {
  std::vector<Entry*> files_;
  fs::path root_;
  bool test_;
  size_t fi_ = 0;
  bool open_ = false;
  U64 left_ = 0;
  U32 crc_ = 0;
  std::ofstream out_;
  std::vector<U8> mem_;
 public:
  size_t errors = 0, written = 0;
  FileSink(std::vector<Entry*> files, fs::path root, bool test) : files_(std::move(files)), root_(std::move(root)), test_(test) {}

  fs::path target(const Entry* e) const { return root_ / fs::u8path(e->path); }

  void finishFile() {
    Entry* e = files_[fi_];
    if (e->flags & FLAG_X86) {
      x86Filter(mem_.data(), mem_.size(), false);
      crc_ = crc32Update(0, mem_.data(), mem_.size());
      if (!test_) out_.write((const char*)mem_.data(), (std::streamsize)mem_.size());
      mem_.clear();
      mem_.shrink_to_fit();
    }
    if (!test_) {
      out_.close();
      if (!out_) {
        fprintf(stderr, "\nXato: yozib bo'lmadi: %s\n", e->path.c_str());
        ++errors;
      }
      std::error_code ec;
      fs::last_write_time(target(e), fromUnixTime(e->mtime), ec);
      if (e->mode) fs::permissions(target(e), (fs::perms)e->mode, ec);
    }
    if (crc_ != e->crc) {
      fprintf(stderr, "\nCRC XATOSI: %s\n", e->path.c_str());
      ++errors;
    }
    ++written;
    open_ = false;
    ++fi_;
  }

  void startFile() {
    Entry* e = files_[fi_];
    left_ = e->size;
    crc_ = 0;
    if (!test_) {
      fs::path t = target(e);
      std::error_code ec;
      fs::create_directories(t.parent_path(), ec);
      out_.open(t, std::ios::binary | std::ios::trunc);
      if (!out_) throw std::runtime_error("faylni yaratib bo'lmadi: " + t.u8string());
    }
    if (e->flags & FLAG_X86) mem_.reserve((size_t)e->size);
    open_ = true;
    if (left_ == 0) finishFile();
  }

  void feed(const U8* p, size_t n) {
    while (n > 0) {
      if (!open_) {
        if (fi_ >= files_.size()) throw std::runtime_error("arxivda ortiqcha ma'lumot (buzilgan?)");
        startFile();
        continue;
      }
      size_t take = (size_t)std::min<U64>(n, left_);
      if (files_[fi_]->flags & FLAG_X86) {
        mem_.insert(mem_.end(), p, p + take);
      } else {
        crc_ = crc32Update(crc_, p, take);
        if (!test_) out_.write((const char*)p, (std::streamsize)take);
      }
      p += take;
      n -= take;
      left_ -= take;
      if (left_ == 0) finishFile();
    }
  }

  void finish() {
    while (fi_ < files_.size()) {
      if (!open_) startFile();
      if (open_) throw std::runtime_error("arxiv to'liq emas: ma'lumot yetishmadi");
    }
  }
};

static int cmdExtract(const std::string& archive, const Options& opt, bool test) {
  auto t0 = std::chrono::steady_clock::now();
  ArchiveReader ar(archive);
  fs::path root = fs::u8path(opt.outDir.empty() ? "." : opt.outDir);

  std::vector<Entry*> files;
  U64 total = 0;
  for (Entry& e : ar.entries) {
    if (!safePath(e.path)) throw std::runtime_error("xavfli yo'l arxivda: " + e.path);
    if (e.type == 0) {
      files.push_back(&e);
      total += e.size;
    }
  }
  if (!test) {
    if (!opt.force) {
      size_t exist = 0;
      for (Entry* e : files) {
        std::error_code ec;
        if (fs::exists(root / fs::u8path(e->path), ec)) {
          if (exist < 5) fprintf(stderr, "  mavjud: %s\n", e->path.c_str());
          ++exist;
        }
      }
      if (exist) {
        fprintf(stderr, "Xato: %zu ta fayl allaqachon mavjud. Ustidan yozish uchun -f qo'shing.\n", exist);
        return 1;
      }
    }
    for (Entry& e : ar.entries)
      if (e.type == 1) {
        std::error_code ec;
        fs::create_directories(root / fs::u8path(e.path), ec);
      }
  }

  if (!g_quiet) fprintf(stderr, "DragonZip %s — %s: %s (%zu ta fayl, %s)\n", DZ_VERSION, test ? "tekshirish" : "ochish",
                        archive.c_str(), files.size(), fmtSize(total).c_str());

  int threads = opt.threads > 0 ? opt.threads : (int)std::min<unsigned>(4, std::max(1u, std::thread::hardware_concurrency()));
  FileSink sink(files, root, test);
  Progress prog(test ? "Tekshirilmoqda" : "Ochilmoqda ", total);
  U64 off = 16, rawSeen = 0;
  U32 seen = 0;
  while (off < ar.dirOff) {
    // bir nechta blokni parallel ochamiz
    std::vector<BlockInfo> bis;
    std::vector<std::vector<U8>> comps;
    while ((int)bis.size() < threads && off < ar.dirOff) {
      BlockInfo b = ar.readHeader(off);
      if (b.tag != 'B') throw std::runtime_error("kutilmagan blok turi");
      rawSeen += b.raw;
      if (rawSeen > total) throw std::runtime_error("blok hajmlari katalogga mos emas (buzilgan?)");
      std::vector<U8> c((size_t)b.comp);
      ar.readAt(b.offset, c.data(), c.size());
      off = b.offset + b.comp;
      bis.push_back(b);
      comps.push_back(std::move(c));
      if (b.method == METHOD_STORE) break;  // saqlangan bloklar tez, ketma-ket ishlaymiz
    }
    std::vector<std::vector<U8>> raws(bis.size());
    runParallel(bis.size(), [&](size_t i) {
      raws[i] = ArchiveReader::decodeData(bis[i], comps[i], &prog.done);
      std::vector<U8>().swap(comps[i]);
    }, &prog);
    for (auto& r : raws) {
      sink.feed(r.data(), r.size());
      std::vector<U8>().swap(r);
    }
    seen += (U32)bis.size();
  }
  sink.finish();
  prog.show(true);
  if (seen != ar.blockCount) {
    fprintf(stderr, "Ogohlantirish: bloklar soni mos emas\n");
    sink.errors++;
  }

  // papkalar vaqtini tiklash (fayllar yozilgandan keyin)
  if (!test) {
    for (auto it = ar.entries.rbegin(); it != ar.entries.rend(); ++it)
      if (it->type == 1) {
        std::error_code ec;
        fs::path t = root / fs::u8path(it->path);
        fs::last_write_time(t, fromUnixTime(it->mtime), ec);
        if (it->mode) fs::permissions(t, (fs::perms)it->mode, ec);
      }
  }
  double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  if (sink.errors) {
    fprintf(stderr, "XATOLAR: %zu ta. Arxiv buzilgan bo'lishi mumkin!\n", sink.errors);
    return 2;
  }
  if (!g_quiet)
    fprintf(stderr, "%s: %zu ta fayl, %s, hammasi CRC32 bo'yicha to'g'ri. vaqt: %s\n", test ? "Arxiv butun" : "Tayyor",
            sink.written, fmtSize(total).c_str(), fmtTime(el).c_str());
  return 0;
}

static int cmdList(const std::string& archive) {
  ArchiveReader ar(archive);
  U64 total = 0;
  size_t nf = 0, nd = 0;
  printf("%-19s %14s  %s\n", "Sana", "Hajm", "Nom");
  printf("------------------- --------------  ------------------------------\n");
  for (const Entry& e : ar.entries) {
    char tb[32] = "?";
    time_t t = (time_t)e.mtime;
    struct tm tmv;
#ifdef _WIN32
    if (localtime_s(&tmv, &t) == 0) strftime(tb, sizeof tb, "%Y-%m-%d %H:%M:%S", &tmv);
#else
    if (localtime_r(&t, &tmv)) strftime(tb, sizeof tb, "%Y-%m-%d %H:%M:%S", &tmv);
#endif
    if (e.type == 1) {
      printf("%-19s %14s  %s/\n", tb, "<papka>", e.path.c_str());
      ++nd;
    } else {
      printf("%-19s %14llu  %s%s\n", tb, (unsigned long long)e.size, e.path.c_str(),
             (e.flags & FLAG_STORED) ? "  [saqlangan]" : (e.flags & FLAG_X86) ? "  [x86]" : "");
      total += e.size;
      ++nf;
    }
  }
  printf("------------------- --------------  ------------------------------\n");
  printf("%zu ta fayl, %zu ta papka. Asl hajm: %s, arxiv: %s", nf, nd, fmtSize(total).c_str(), fmtSize(ar.fileSize).c_str());
  if (total) printf(" (%.2f%%)", 100.0 * ar.fileSize / total);
  printf("\n");
  return 0;
}

// Xotirada siqish/ochish sinovi: siqish darajasi, tezlik va to'g'rilik
static int cmdBench(const std::vector<std::string>& files, const Options& opt) {
  U64 tin = 0, tout = 0;
  double tenc = 0, tdec = 0;
  for (const std::string& f : files) {
    std::ifstream in(fs::u8path(f), std::ios::binary);
    if (!in) { fprintf(stderr, "ochilmadi: %s\n", f.c_str()); return 1; }
    std::vector<U8> data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::vector<U8> work = data;
    if (isX86Exe(work.data(), work.size())) x86Filter(work.data(), work.size(), true);
    int tb = tableBitsFor(opt.level, work.size());
    std::vector<U8> comp;
    auto t0 = std::chrono::steady_clock::now();
    cmEncode(work.data(), work.size(), tb, comp, nullptr);
    auto t1 = std::chrono::steady_clock::now();
    std::vector<U8> back(work.size());
    cmDecode(comp.data(), comp.size(), back.data(), back.size(), tb, nullptr);
    auto t2 = std::chrono::steady_clock::now();
    if (isX86Exe(data.data(), data.size())) x86Filter(back.data(), back.size(), false);
    bool ok = back == data;
    double e = std::chrono::duration<double>(t1 - t0).count(), d = std::chrono::duration<double>(t2 - t1).count();
    printf("%-28s %10zu -> %9zu  %6.3f bpb  enc %6.2f MB/s  dec %6.2f MB/s  %s\n", fs::u8path(f).filename().u8string().c_str(),
           data.size(), comp.size(), data.empty() ? 0.0 : 8.0 * comp.size() / data.size(), data.size() / e / 1048576.0,
           data.size() / d / 1048576.0, ok ? "OK" : "XATO!");
    if (!ok) return 2;
    tin += data.size();
    tout += comp.size();
    tenc += e;
    tdec += d;
  }
  printf("%-28s %10llu -> %9llu  %6.3f bpb  enc %6.2f MB/s  dec %6.2f MB/s\n", "JAMI", (unsigned long long)tin,
         (unsigned long long)tout, tin ? 8.0 * tout / tin : 0.0, tin / tenc / 1048576.0, tin / tdec / 1048576.0);
  return 0;
}

// ============================================================================
//  Buyruq satri
// ============================================================================

static void usage() {
  fprintf(stderr,
          "DragonZip %s — kuchli yo'qotishsiz arxivator (context mixing)\n"
          "\n"
          "Foydalanish:\n"
          "  dragonzip a [parametrlar] <arxiv.drz> <fayl/papka>...   arxiv yaratish (siqish)\n"
          "  dragonzip x [parametrlar] <arxiv.drz>                   arxivni ochish\n"
          "  dragonzip t <arxiv.drz>                                 butunligini tekshirish\n"
          "  dragonzip l <arxiv.drz>                                 tarkibini ko'rish\n"
          "  dragonzip bench [-1..-9] <fayl>...                      siqish sinovi (xotirada)\n"
          "\n"
          "Parametrlar:\n"
          "  -1 .. -9     siqish darajasi = model xotirasi (standart: -6 = 128 MB/oqim)\n"
          "  -t N         oqimlar (thread) soni (standart: yadrolar soni, ko'pi bilan 4)\n"
          "  -b N         blok hajmi, MB (katta blok = yaxshiroq siqish, kamroq parallellik)\n"
          "  -o PAPKA     ochish uchun papka (x)\n"
          "  -f           mavjud fayllar ustidan yozish\n"
          "  -q           jim rejim\n"
          "  --no-detect  siqilgan fayllarni aniqlamaslik (hammasini CM bilan siqish)\n"
          "\n"
          "Maksimal siqish uchun:  dragonzip a -9 -t 1 arxiv.drz papka/\n",
          DZ_VERSION);
}

static int realMain(std::vector<std::string> args) {
  initTables();
  initStates();
  if (args.size() < 2) {
    usage();
    return 1;
  }
  std::string cmd = args[1];
  Options opt;
  std::vector<std::string> pos;
  for (size_t i = 2; i < args.size(); ++i) {
    const std::string& a = args[i];
    if (a.size() == 2 && a[0] == '-' && a[1] >= '1' && a[1] <= '9') opt.level = a[1] - '0';
    else if (a == "-t" && i + 1 < args.size()) opt.threads = std::max(1, atoi(args[++i].c_str()));
    else if (a == "-b" && i + 1 < args.size()) opt.blockMB = (U64)std::max(1, atoi(args[++i].c_str()));
    else if (a == "-o" && i + 1 < args.size()) opt.outDir = args[++i];
    else if (a == "-f") opt.force = true;
    else if (a == "-q") g_quiet = true;
    else if (a == "--no-detect") opt.detect = false;
    else if (a == "--") { for (++i; i < args.size(); ++i) pos.push_back(args[i]); }
    else if (a.size() > 1 && a[0] == '-') {
      fprintf(stderr, "Noma'lum parametr: %s\n", a.c_str());
      return 1;
    } else pos.push_back(a);
  }
  try {
    if (cmd == "a" || cmd == "c") {
      if (pos.size() < 2) { usage(); return 1; }
      std::vector<std::string> ins(pos.begin() + 1, pos.end());
      return cmdCreate(pos[0], ins, opt);
    } else if (cmd == "x" || cmd == "e") {
      if (pos.size() != 1) { usage(); return 1; }
      return cmdExtract(pos[0], opt, false);
    } else if (cmd == "t") {
      if (pos.size() != 1) { usage(); return 1; }
      return cmdExtract(pos[0], opt, true);
    } else if (cmd == "l") {
      if (pos.size() != 1) { usage(); return 1; }
      return cmdList(pos[0]);
    } else if (cmd == "bench") {
      return cmdBench(pos, opt);
    } else if (cmd == "-h" || cmd == "--help" || cmd == "h") {
      usage();
      return 0;
    }
    fprintf(stderr, "Noma'lum buyruq: %s\n\n", cmd.c_str());
    usage();
    return 1;
  } catch (const std::bad_alloc&) {
    fprintf(stderr, "\nXato: xotira yetmadi. Pastroq daraja (-4) yoki kamroq oqim (-t 1) bilan urinib ko'ring.\n");
    return 3;
  } catch (const std::exception& e) {
    fprintf(stderr, "\nXato: %s\n", e.what());
    return 2;
  }
}

#ifdef _WIN32
int main() {
  SetConsoleOutputCP(CP_UTF8);
  int argc = 0;
  LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &argc);
  std::vector<std::string> args;
  for (int i = 0; i < argc; ++i) {
    int len = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, nullptr, 0, nullptr, nullptr);
    std::string s(len > 0 ? len - 1 : 0, '\0');
    if (len > 1) WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, &s[0], len, nullptr, nullptr);
    args.push_back(s);
  }
  LocalFree(wargv);
  return realMain(args);
}
#else
int main(int argc, char** argv) { return realMain(std::vector<std::string>(argv, argv + argc)); }
#endif
