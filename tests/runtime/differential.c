// Compares compiled operations with GMP on boundary and random operands.
// Usage: rcgmp-differential [seed [iterations]]
#include "Test.h"
#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <string.h>

void *addi_local(void *, void *);
void *addi_atomic(void *, void *);
void *subi_local(void *, void *);
void *subi_atomic(void *, void *);
void *muli_local(void *, void *);
void *muli_atomic(void *, void *);
void *andi_local(void *, void *);
void *andi_atomic(void *, void *);
void *xori_local(void *, void *);
void *xori_atomic(void *, void *);
void *retain_local(void *);
void *retain_atomic(void *);
void release_local(void *);
void release_atomic(void *);
void *from_signed_i61_local(uint64_t);
void *from_unsigned_i61_local(uint64_t);
void *from_signed_i62_local(uint64_t);
void *from_unsigned_i62_local(uint64_t);
void *from_signed_i63_local(uint64_t);
void *from_unsigned_i63_local(uint64_t);
void *from_signed_i64_local(uint64_t);
void *from_unsigned_i64_local(uint64_t);
void *from_signed_i65_local(uint64_t, uint64_t);
void *from_unsigned_i65_local(uint64_t, uint64_t);
void *from_signed_i128_local(uint64_t, uint64_t);
void *from_unsigned_i128_local(uint64_t, uint64_t);
void *from_signed_i129_local(uint64_t, uint64_t, uint64_t);
void *from_unsigned_i129_local(uint64_t, uint64_t, uint64_t);
void *from_signed_i64_atomic(uint64_t);
void *from_unsigned_i64_atomic(uint64_t);
void *from_signed_i129_atomic(uint64_t, uint64_t, uint64_t);
void *from_unsigned_i129_atomic(uint64_t, uint64_t, uint64_t);
void *constant_local(size_t);
void *constant_atomic(size_t);

// Compiled reference counting, indexed by atomicity. Operands are built by the
// C runtime, so both implementations operate on the same objects.
static void *(*retain[2])(void *) = {retain_local, retain_atomic};
static void (*release[2])(void *) = {release_local, release_atomic};

// Compiled constants, selected by index, and their values in the order of
// differential.mlir.
static void *(*constant[2])(size_t) = {constant_local, constant_atomic};
static const char *constantValues[] = {
    "0",
    "-1",
    "2305843009213693951",
    "-2305843009213693952",
    "2305843009213693952",
    "-2305843009213693953",
    "9223372036854775807",
    "-9223372036854775808",
    "18446744073709551615",
    "-18446744073709551616",
    "340282366920938463463374607431768211455",
    "-340282366920938463463374607431768211457",
};
#define CONSTANT_COUNT (sizeof(constantValues) / sizeof(*constantValues))
static mpz_t constants[CONSTANT_COUNT];

// Each operation lists its compiled and runtime entry points, indexed by
// atomicity, and the GMP function it must agree with.
typedef struct {
  const char *name;
  void *(*apply[2])(void *, void *);
  void *(*runtime[2])(void *, void *, uint32_t);
  void (*oracle)(mpz_ptr, mpz_srcptr, mpz_srcptr);
} BinaryOperation;

static BinaryOperation binaryOperations[] = {
    {"addi",
     {addi_local, addi_atomic},
     {rcgmp_mpz_add, rcgmp_mpz_add_atomic},
     mpz_add},
    {"subi",
     {subi_local, subi_atomic},
     {rcgmp_mpz_sub, rcgmp_mpz_sub_atomic},
     mpz_sub},
    {"muli",
     {muli_local, muli_atomic},
     {rcgmp_mpz_mul, rcgmp_mpz_mul_atomic},
     mpz_mul},
    {"andi",
     {andi_local, andi_atomic},
     {rcgmp_mpz_and, rcgmp_mpz_and_atomic},
     mpz_and},
    {"xori",
     {xori_local, xori_atomic},
     {rcgmp_mpz_xor, rcgmp_mpz_xor_atomic},
     mpz_xor},
};

// Each from_int entry point converts an operand of the given width, passed as
// ceil(width / 64) 64-bit words, least significant first.
typedef struct {
  const char *name;
  unsigned width;
  int isUnsigned;
  uint32_t atomic;
  union {
    void *(*one)(uint64_t);
    void *(*two)(uint64_t, uint64_t);
    void *(*three)(uint64_t, uint64_t, uint64_t);
  } apply;
} FromIntOperation;

// The widths cover each way of deciding whether a value is immediate, and
// operands of one, two, and three words.
static FromIntOperation fromIntOperations[] = {
    {"from_signed_i61", 61, 0, 0, {.one = from_signed_i61_local}},
    {"from_unsigned_i61", 61, 1, 0, {.one = from_unsigned_i61_local}},
    {"from_signed_i62", 62, 0, 0, {.one = from_signed_i62_local}},
    {"from_unsigned_i62", 62, 1, 0, {.one = from_unsigned_i62_local}},
    {"from_signed_i63", 63, 0, 0, {.one = from_signed_i63_local}},
    {"from_unsigned_i63", 63, 1, 0, {.one = from_unsigned_i63_local}},
    {"from_signed_i64", 64, 0, 0, {.one = from_signed_i64_local}},
    {"from_unsigned_i64", 64, 1, 0, {.one = from_unsigned_i64_local}},
    {"from_signed_i65", 65, 0, 0, {.two = from_signed_i65_local}},
    {"from_unsigned_i65", 65, 1, 0, {.two = from_unsigned_i65_local}},
    {"from_signed_i128", 128, 0, 0, {.two = from_signed_i128_local}},
    {"from_unsigned_i128", 128, 1, 0, {.two = from_unsigned_i128_local}},
    {"from_signed_i129", 129, 0, 0, {.three = from_signed_i129_local}},
    {"from_unsigned_i129", 129, 1, 0, {.three = from_unsigned_i129_local}},
    {"from_signed_i64", 64, 0, 1, {.one = from_signed_i64_atomic}},
    {"from_unsigned_i64", 64, 1, 1, {.one = from_unsigned_i64_atomic}},
    {"from_signed_i129", 129, 0, 1, {.three = from_signed_i129_atomic}},
    {"from_unsigned_i129", 129, 1, 1, {.three = from_unsigned_i129_atomic}},
};

// A shared operand has an extra reference held by the caller, which must
// survive with its value. An aliased rhs is a retained reference to the lhs. An
// immortal heap operand has a zero reference count, like a compiled constant.
enum {
  SharedLhs = 1,
  SharedRhs = 2,
  Aliased = 4,
  ImmortalLhs = 8,
  ImmortalRhs = 16
};
typedef struct {
  unsigned flags;
  const char *name;
} OwnershipMode;

static OwnershipMode pairModes[] = {
    {0, "unique operands"},
    {SharedLhs, "shared lhs"},
    {SharedRhs, "shared rhs"},
    {SharedLhs | SharedRhs, "shared operands"},
    {ImmortalLhs, "immortal lhs"},
    {SharedLhs | ImmortalRhs, "shared lhs, immortal rhs"},
    {ImmortalLhs | ImmortalRhs, "immortal operands"},
};
static OwnershipMode aliasModes[] = {
    {Aliased, "aliased operands"},
    {Aliased | SharedLhs, "aliased shared operand"},
    {Aliased | ImmortalLhs, "aliased immortal operand"},
};

static unsigned long seed = 31415;
static unsigned long iterations = 20000;

// Where the run is, so that a failure can be located without a rerun: a phase
// and, within a random stream, the one-based position and the stream length.
static const char *phase;
static unsigned long position, positionCount;

static void print_position(void) {
  if (positionCount)
    fprintf(stderr, "  %s %lu of %lu\n", phase, position, positionCount);
  else
    fprintf(stderr, "  %s\n", phase);
  fprintf(stderr, "  seed = %lu, iterations = %lu\n", seed, iterations);
}

// A heap operand is allocated with spare limbs beyond its magnitude, as results
// of earlier operations often are, so that an operation may reuse it in place.
// Like the stale limbs of such results, the spare limbs are nonzero, so reading
// past the magnitude changes the value. A constant operand is the object that
// compiled code produces for the value, which has no spare limbs.
typedef struct {
  mpz_srcptr value;
  size_t spare;
  void *constant;
} Operand;

typedef struct {
  BinaryOperation *operation;
  uint32_t atomic;
  OwnershipMode *mode;
  Operand lhs, rhs;
  // Each heap operand has room for the result, so a unique one must hold it.
  int roomy;
} BinaryCase;

static int fits_immediate(mpz_srcptr value) {
  return mpz_size(value) <= 1 &&
         mpz_getlimbn(value, 0) <=
             (uintptr_t)RCGMP_IMM_MAX + (mpz_sgn(value) < 0);
}

static void print_operand(const char *name, Operand operand) {
  if (fits_immediate(operand.value))
    gmp_fprintf(stderr, "  %s = %#Zx (immediate)\n", name, operand.value);
  else if (operand.constant)
    gmp_fprintf(stderr, "  %s = %#Zx (constant)\n", name, operand.value);
  else
    gmp_fprintf(stderr, "  %s = %#Zx (%zu spare limbs)\n", name, operand.value,
                operand.spare);
}

static void fail(BinaryCase *test, const char *problem) {
  fprintf(stderr, "%s (%s, %s): %s\n", test->operation->name,
          test->atomic ? "atomic" : "local", test->mode->name, problem);
  print_operand("lhs", test->lhs);
  print_operand("rhs", test->rhs);
  print_position();
  abort();
}

// An immortal heap operand keeps its spare limbs, which would let a counted one
// hold the result in place.
static void *make_operand(Operand operand, uint32_t atomic, int immortal) {
  if (operand.constant)
    return operand.constant;
  void *handle = from_mpz(operand.value, atomic);
  if (rcgmp_is_immediate(handle))
    return handle;
  RCGMPMpz *exact = handle;
  size_t count = mpz_size(operand.value);
  RCGMPMpz *padded = rcgmp_mpz_allocate(count + operand.spare, atomic);
  memcpy(padded->limbs, exact->limbs, count * sizeof(mp_limb_t));
  for (size_t i = 0; i < operand.spare; ++i)
    padded->limbs[count + i] =
        i % 2 ? GMP_NUMB_MAX / 0xff * 0xa5 : GMP_NUMB_MAX;
  padded->size = exact->size;
  rcgmp_mpz_release(exact, atomic);
  if (immortal) {
    if (atomic)
      atomic_store(&padded->refs.atomic, 0);
    else
      padded->refs.local = 0;
  }
  return padded;
}

static uintptr_t references(void *value, uint32_t atomic) {
  RCGMPMpz *heap = value;
  return atomic ? atomic_load(&heap->refs.atomic) : heap->refs.local;
}

// Each operand position holding operand consumed one of its references, and the
// caller's reference kept its value. The result may be another reference to it.
// An immortal operand keeps its zero count.
static void release_shared(BinaryCase *test, void *operand, mpz_srcptr value,
                           void *result, int immortal) {
  if (!represents(operand, value))
    fail(test, "shared operand was modified");
  if (!rcgmp_is_immediate(operand) &&
      references(operand, test->atomic) !=
          (immortal ? 0 : 1 + (operand == result)))
    fail(test, "operand positions did not each consume one reference");
  release[test->atomic](operand);
}

// Immortal operands are neither counted nor written, even when the result is
// the same object. Those made here are then freed.
static void drop_immortal(BinaryCase *test, Operand operand, void *handle) {
  if (!represents(handle, operand.value) ||
      references(handle, test->atomic) != 0)
    fail(test, "immortal operand was modified");
  if (!operand.constant)
    rcgmp_mpz_free(handle, test->atomic);
}

static void retain_operand(BinaryCase *test, void *operand) {
  if (retain[test->atomic](operand) != operand)
    fail(test, "retain returned a different handle");
}

static void check_binary(BinaryCase *test) {
  uint32_t atomic = test->atomic;
  unsigned flags = test->mode->flags;
  mpz_srcptr lhs = test->lhs.value, rhs = test->rhs.value;
  void *x = make_operand(test->lhs, atomic, flags & ImmortalLhs);
  void *y = x;
  if (flags & Aliased)
    retain_operand(test, x);
  else
    y = make_operand(test->rhs, atomic, flags & ImmortalRhs);
  int immortalX =
      !rcgmp_is_immediate(x) && (test->lhs.constant || (flags & ImmortalLhs));
  int immortalY = y == x ? immortalX
                         : !rcgmp_is_immediate(y) &&
                               (test->rhs.constant || (flags & ImmortalRhs));
  if (!represents(x, lhs) || !represents(y, rhs))
    fail(test, "operand construction is not canonical");
  if (flags & SharedLhs)
    retain_operand(test, x);
  if (flags & SharedRhs)
    retain_operand(test, y);
  mpz_t expected;
  mpz_init(expected);
  test->operation->oracle(expected, lhs, rhs);
  size_t before = allocations;
  void *result = test->operation->apply[atomic](x, y);
  if (!represents(result, expected))
    fail(test, "result differs from GMP or is not canonical");
  // General multiplication cannot write over its operands, and immortal ones
  // are never written.
  int reusable = test->operation->oracle != mpz_mul || mpz_size(lhs) <= 1 ||
                 mpz_size(rhs) <= 1;
  int counted = (!rcgmp_is_immediate(x) && !immortalX) ||
                (!rcgmp_is_immediate(y) && !immortalY);
  if (test->roomy && !(flags & (SharedLhs | SharedRhs | Aliased)) && reusable &&
      counted && !rcgmp_is_immediate(result) && allocations != before)
    fail(test, "result did not reuse a unique operand");
  if (flags & SharedLhs)
    release_shared(test, x, lhs, result, immortalX);
  if (flags & SharedRhs)
    release_shared(test, y, rhs, result, immortalY);
  release[atomic](result);
  if (immortalX)
    drop_immortal(test, test->lhs, x);
  if (immortalY && y != x)
    drop_immortal(test, test->rhs, y);
  mpz_clear(expected);
  if (allocations != deallocations)
    fail(test, "allocations and deallocations do not balance");
}

static void check_pair(BinaryOperation *operation, uint32_t atomic, Operand lhs,
                       Operand rhs, int roomy) {
  for (size_t mode = 0; mode < sizeof(pairModes) / sizeof(*pairModes); ++mode) {
    BinaryCase test = {operation, atomic, &pairModes[mode], lhs, rhs, roomy};
    check_binary(&test);
  }
}

static void check_aliased(BinaryOperation *operation, uint32_t atomic,
                          Operand operand) {
  for (size_t mode = 0; mode < sizeof(aliasModes) / sizeof(*aliasModes);
       ++mode) {
    BinaryCase test = {operation, atomic,  &aliasModes[mode],
                       operand,   operand, 0};
    check_binary(&test);
  }
}

// Compiled code calls the runtime only when an operand is on the heap or an
// immediate result overflows, but every runtime entry point accepts any
// canonical operands and must return a canonical result. Reuse is permitted
// only for heap operands.
static OwnershipMode runtimeMode = {0, "runtime entry point"};

static void check_runtime(BinaryOperation *operation, uint32_t atomic,
                          mpz_srcptr lhs, mpz_srcptr rhs) {
  BinaryCase test = {operation,      atomic,         &runtimeMode,
                     {lhs, 0, NULL}, {rhs, 0, NULL}, 0};
  mpz_t expected;
  mpz_init(expected);
  operation->oracle(expected, lhs, rhs);
  void *result = operation->runtime[atomic](from_mpz(lhs, atomic),
                                            from_mpz(rhs, atomic), 0);
  if (!represents(result, expected))
    fail(&test, "result differs from GMP or is not canonical");
  release[atomic](result);
  mpz_clear(expected);
  if (allocations != deallocations)
    fail(&test, "allocations and deallocations do not balance");
}

static void add_small(mpz_t value, long delta) {
  if (delta < 0)
    mpz_sub_ui(value, value, -delta);
  else
    mpz_add_ui(value, value, delta);
}

static long random_delta(gmp_randstate_t random, long bound) {
  return (long)gmp_urandomm_ui(random, 2 * bound + 1) - bound;
}

static void power_of_two(mpz_t value, unsigned exponent) {
  mpz_set_ui(value, 0);
  mpz_setbit(value, exponent);
}

static void random_operand(mpz_t value, gmp_randstate_t random) {
  unsigned word = GMP_NUMB_BITS;
  // Square roots of the immediate bound, so that products land near it; the
  // immediate bound; the signed word bound; and one to four full limbs.
  unsigned edges[] = {(word - 3) / 2, (word - 2) / 2, word - 3, word - 1,
                      word,           2 * word,       3 * word, 4 * word};
  switch (gmp_urandomm_ui(random, 4)) {
  case 0:
    power_of_two(
        value, edges[gmp_urandomm_ui(random, sizeof(edges) / sizeof(*edges))]);
    add_small(value, random_delta(random, 4));
    break;
  case 1:
    mpz_set_si(value, random_delta(random, 4));
    break;
  case 2:
    // Long runs of ones and zeros stress carries, borrows, and sign extension.
    mpz_rrandomb(value, random, 1 + gmp_urandomm_ui(random, 5 * word));
    break;
  default:
    mpz_urandomb(value, random, gmp_urandomm_ui(random, 5 * word + 1));
    break;
  }
  if (gmp_urandomb_ui(random, 1))
    mpz_neg(value, value);
}

// Derive rhs from lhs to reach cancellation, complementary bit patterns, and
// products at either immediate bound, which independent operands rarely hit.
static void related_operand(mpz_t rhs, mpz_srcptr lhs, gmp_randstate_t random) {
  switch (gmp_urandomm_ui(random, 3)) {
  case 0:
    mpz_set(rhs, lhs);
    if (gmp_urandomb_ui(random, 1))
      mpz_neg(rhs, rhs);
    add_small(rhs, random_delta(random, 2));
    break;
  case 1:
    mpz_com(rhs, lhs);
    break;
  default:
    if (!mpz_sgn(lhs)) {
      random_operand(rhs, random);
      break;
    }
    power_of_two(rhs, GMP_NUMB_BITS - 3);
    mpz_tdiv_q(rhs, rhs, lhs);
    if (gmp_urandomb_ui(random, 1))
      mpz_neg(rhs, rhs);
    add_small(rhs, random_delta(random, 1));
    break;
  }
}

// With a one-limb operand, a result needs at most one limb more than the other
// operand, so a one-limb heap operand can hold it in place when its spare limbs
// cover the other operand. Random operands have at most five limbs and derived
// ones at most six, so spares range up to six.
static size_t random_spare(gmp_randstate_t random) {
  return gmp_urandomm_ui(random, 7);
}

// Each operation and atomicity draws from its own stream, keyed by the seed and
// the operation name, so that a larger iteration count extends every stream
// and adding an operation leaves the others unchanged.
static void seed_stream(gmp_randstate_t random, const char *name,
                        uint32_t atomic) {
  mpz_t key;
  mpz_init(key);
  mpz_import(key, strlen(name), 1, 1, 0, 0, name);
  mpz_mul_2exp(key, key, 1);
  mpz_add_ui(key, key, atomic);
  mpz_mul_2exp(key, key, sizeof(seed) * CHAR_BIT);
  mpz_add_ui(key, key, seed);
  gmp_randseed(random, key);
  mpz_clear(key);
}

static void *call_from_int(FromIntOperation *operation, const uint64_t *words) {
  switch ((operation->width + 63) / 64) {
  case 1:
    return operation->apply.one(words[0]);
  case 2:
    return operation->apply.two(words[0], words[1]);
  default:
    return operation->apply.three(words[0], words[1], words[2]);
  }
}

static void fail_from_int(FromIntOperation *operation, const uint64_t *words,
                          mpz_srcptr expected, const char *problem) {
  fprintf(stderr, "%s (%s): %s\n  words =", operation->name,
          operation->atomic ? "atomic" : "local", problem);
  for (unsigned i = 0; i < (operation->width + 63) / 64; ++i)
    fprintf(stderr, " %#" PRIx64, words[i]);
  gmp_fprintf(stderr, "\n  expected = %#Zx\n", expected);
  print_position();
  abort();
}

// Passes the low words of input, which the entry point truncates further to the
// operand width.
static void check_from_int(FromIntOperation *operation, mpz_srcptr input) {
  unsigned width = operation->width, count = (width + 63) / 64;
  uint64_t words[3] = {0};
  mpz_t expected;
  mpz_init(expected);
  mpz_fdiv_r_2exp(expected, input, 64 * count);
  mpz_export(words, NULL, -1, sizeof(*words), 0, 0, expected);
  mpz_fdiv_r_2exp(expected, expected, width);
  // A set sign bit makes the ceiling remainder the value minus 2^width.
  if (!operation->isUnsigned && mpz_tstbit(expected, width - 1))
    mpz_cdiv_r_2exp(expected, expected, width);
  void *result = call_from_int(operation, words);
  if (!represents(result, expected))
    fail_from_int(operation, words, expected,
                  "result differs from GMP or is not canonical");
  release[operation->atomic](result);
  if (allocations != deallocations)
    fail_from_int(operation, words, expected,
                  "allocations and deallocations do not balance");
  mpz_clear(expected);
}

// Values near the immediate bounds, word boundaries, and the operand width,
// with long runs of ones and zeros, and bits above the operand width.
static void random_from_int_input(mpz_t value, gmp_randstate_t random,
                                  unsigned width) {
  unsigned word = GMP_NUMB_BITS;
  unsigned edges[] = {word - 3, word - 2, word - 1, word, width - 1, width};
  switch (gmp_urandomm_ui(random, 3)) {
  case 0:
    power_of_two(
        value, edges[gmp_urandomm_ui(random, sizeof(edges) / sizeof(*edges))]);
    add_small(value, random_delta(random, 4));
    break;
  case 1:
    mpz_rrandomb(value, random, 1 + gmp_urandomm_ui(random, width + 64));
    break;
  default:
    mpz_urandomb(value, random, gmp_urandomm_ui(random, width + 65));
    break;
  }
  if (gmp_urandomb_ui(random, 1))
    mpz_neg(value, value);
}

static void check_from_int_stream(FromIntOperation *operation,
                                  gmp_randstate_t random, mpz_t input) {
  unsigned word = GMP_NUMB_BITS, width = operation->width;
  // Zero and one with either sign; both sides of the immediate bounds and of
  // 2^(word-2), 2^(word-1), and 2^word; powers of two at two-word boundaries;
  // and the operand width's signed minimum and maximum and unsigned maximum.
  // Inputs wider than the operand are truncated.
  struct {
    unsigned exponent;
    long delta;
    int negate;
  } grid[] = {
      {0, -1, 0},           {0, 0, 0},          {0, 0, 1},
      {word - 3, -1, 0},    {word - 3, 0, 0},   {word - 3, 0, 1},
      {word - 3, 1, 1},     {word - 2, -1, 0},  {word - 2, 0, 0},
      {word - 1, -1, 0},    {word - 1, 0, 0},   {word, -1, 0},
      {word, 0, 0},         {word, 0, 1},       {2 * word - 1, 0, 0},
      {2 * word - 1, 0, 1}, {2 * word, 0, 0},   {2 * word, 0, 1},
      {width - 1, 0, 1},    {width - 1, -1, 0}, {width, -1, 0},
  };
  phase = "boundary sweep";
  positionCount = 0;
  for (size_t i = 0; i < sizeof(grid) / sizeof(*grid); ++i) {
    power_of_two(input, grid[i].exponent);
    add_small(input, grid[i].delta);
    if (grid[i].negate)
      mpz_neg(input, input);
    check_from_int(operation, input);
  }
  seed_stream(random, operation->name, operation->atomic);
  phase = "random iteration";
  positionCount = iterations;
  for (unsigned long i = 0; i < iterations; ++i) {
    position = i + 1;
    random_from_int_input(input, random, width);
    check_from_int(operation, input);
  }
}

// Each chain feeds every result into the next operation as its uniquely owned
// lhs, so operands also carry the growth headroom and the stale limbs above the
// magnitude that the runtime leaves behind, and come from earlier reuse. A
// constant rhs may also come back as the result and become the next lhs.
static void check_chains(uint32_t atomic, gmp_randstate_t random) {
  size_t operationCount = sizeof(binaryOperations) / sizeof(*binaryOperations);
  mpz_t accumulator, operand, expected;
  mpz_inits(accumulator, operand, expected, NULL);
  seed_stream(random, "chain", atomic);
  phase = "chain";
  positionCount = iterations / 8;
  for (unsigned long chain = 0; chain < positionCount; ++chain) {
    position = chain + 1;
    random_operand(accumulator, random);
    void *result = make_operand(
        (Operand){accumulator, random_spare(random), NULL}, atomic, 0);
    unsigned steps = 8 + gmp_urandomm_ui(random, 9);
    for (unsigned step = 1; step <= steps; ++step) {
      BinaryOperation *operation;
      // Keep repeated products from growing without bound.
      do
        operation = &binaryOperations[gmp_urandomm_ui(random, operationCount)];
      while (operation->oracle == mpz_mul &&
             mpz_sizeinbase(accumulator, 2) > 1000);
      void *rhs = NULL;
      switch (gmp_urandomm_ui(random, 4)) {
      case 0:
        random_operand(operand, random);
        break;
      case 1:
        related_operand(operand, accumulator, random);
        break;
      case 2: {
        size_t index = gmp_urandomm_ui(random, CONSTANT_COUNT);
        mpz_set(operand, constants[index]);
        rhs = constant[atomic](index);
        break;
      }
      default:
        mpz_set_si(operand, random_delta(random, 4));
        break;
      }
      if (!rhs)
        rhs = make_operand((Operand){operand, random_spare(random), NULL},
                           atomic, 0);
      operation->oracle(expected, accumulator, operand);
      result = operation->apply[atomic](result, rhs);
      if (!represents(result, expected)) {
        fprintf(stderr,
                "%s (%s, chained): result differs from GMP or is not "
                "canonical\n",
                operation->name, atomic ? "atomic" : "local");
        gmp_fprintf(stderr, "  lhs = %#Zx\n  rhs = %#Zx\n", accumulator,
                    operand);
        fprintf(stderr, "  step %u of %u\n", step, steps);
        print_position();
        abort();
      }
      mpz_swap(accumulator, expected);
    }
    release[atomic](result);
    if (allocations != deallocations) {
      fprintf(stderr,
              "chained operations (%s): allocations and deallocations do not "
              "balance\n",
              atomic ? "atomic" : "local");
      print_position();
      abort();
    }
  }
  mpz_clears(accumulator, operand, expected, NULL);
}

// Appends 2^exponent + delta and its negation.
static void add_boundary_pair(mpz_t *boundaries, size_t *count,
                              unsigned exponent, long delta) {
  mpz_ptr positive = boundaries[(*count)++];
  mpz_ptr negative = boundaries[(*count)++];
  mpz_init(positive);
  power_of_two(positive, exponent);
  add_small(positive, delta);
  mpz_init(negative);
  mpz_neg(negative, positive);
}

// Accepts a whole unsigned number in C notation. strtoul alone would accept a
// prefix of the text or wrap a negative number around.
static int parse_count(const char *text, unsigned long *value) {
  if (!isdigit((unsigned char)*text))
    return 0;
  char *end;
  errno = 0;
  *value = strtoul(text, &end, 0);
  return !*end && errno != ERANGE;
}

int main(int argc, char **argv) {
  if (argc > 3 || (argc > 1 && !parse_count(argv[1], &seed)) ||
      (argc > 2 && !parse_count(argv[2], &iterations))) {
    fputs("usage: rcgmp-differential [seed [iterations]]\n", stderr);
    return 2;
  }
  gmp_randstate_t random;
  gmp_randinit_default(random);

  // Each compiled constant holds its value, and one outside the immediate range
  // is the same static object with a zero count on every call.
  for (size_t i = 0; i < CONSTANT_COUNT; ++i) {
    mpz_init_set_str(constants[i], constantValues[i], 10);
    for (uint32_t atomic = 0; atomic < 2; ++atomic) {
      void *value = constant[atomic](i);
      if (!represents(value, constants[i]) ||
          (!rcgmp_is_immediate(value) &&
           (references(value, atomic) != 0 || constant[atomic](i) != value))) {
        fprintf(stderr,
                "constant %s (%s): value differs from GMP, is not canonical, "
                "or is counted\n",
                constantValues[i], atomic ? "atomic" : "local");
        abort();
      }
    }
  }

  // Zero, one, both sides of the immediate bounds, and one or two full limbs;
  // the square roots of the immediate bound, whose products land exactly on
  // it; and the signed word bound.
  mpz_t boundaries[21];
  size_t boundaryCount = 0;
  mpz_init_set_si(boundaries[boundaryCount++], 0);
  mpz_init_set_si(boundaries[boundaryCount++], 1);
  mpz_init_set_si(boundaries[boundaryCount++], -1);
  unsigned edges[] = {GMP_NUMB_BITS - 3, GMP_NUMB_BITS};
  for (size_t edge = 0; edge < sizeof(edges) / sizeof(*edges); ++edge)
    for (long delta = -1; delta <= 1; ++delta)
      add_boundary_pair(boundaries, &boundaryCount, edges[edge], delta);
  unsigned powers[] = {(GMP_NUMB_BITS - 3) / 2, (GMP_NUMB_BITS - 2) / 2,
                       GMP_NUMB_BITS - 1};
  for (size_t power = 0; power < sizeof(powers) / sizeof(*powers); ++power)
    add_boundary_pair(boundaries, &boundaryCount, powers[power], 0);
  // No spare limbs, so that every growing result needs a new allocation, and
  // enough for any result of two boundary operands to fit in either one.
  size_t boundarySpares[] = {0, 4};

  mpz_t lhs, rhs;
  mpz_inits(lhs, rhs, NULL);
  for (size_t index = 0;
       index < sizeof(binaryOperations) / sizeof(*binaryOperations); ++index) {
    BinaryOperation *operation = &binaryOperations[index];
    for (uint32_t atomic = 0; atomic < 2; ++atomic) {
      phase = "boundary sweep";
      positionCount = 0;
      for (size_t spare = 0; spare < 2; ++spare) {
        for (size_t i = 0; i < boundaryCount; ++i) {
          Operand x = {boundaries[i], boundarySpares[spare], NULL};
          for (size_t j = 0; j < boundaryCount; ++j) {
            Operand y = {boundaries[j], boundarySpares[spare], NULL};
            check_pair(operation, atomic, x, y, boundarySpares[spare] > 0);
          }
          check_aliased(operation, atomic, x);
        }
      }
      // Constants against boundary operands on either side and each other.
      for (size_t i = 0; i < CONSTANT_COUNT; ++i) {
        Operand c = {constants[i], 0, constant[atomic](i)};
        for (size_t spare = 0; spare < 2; ++spare) {
          for (size_t j = 0; j < boundaryCount; ++j) {
            Operand y = {boundaries[j], boundarySpares[spare], NULL};
            check_pair(operation, atomic, c, y, boundarySpares[spare] > 0);
            check_pair(operation, atomic, y, c, boundarySpares[spare] > 0);
          }
        }
        for (size_t j = 0; j < CONSTANT_COUNT; ++j) {
          Operand d = {constants[j], 0, constant[atomic](j)};
          check_pair(operation, atomic, c, d, 0);
        }
        check_aliased(operation, atomic, c);
      }
      for (size_t i = 0; i < boundaryCount; ++i)
        for (size_t j = 0; j < boundaryCount; ++j)
          if (fits_immediate(boundaries[i]) && fits_immediate(boundaries[j]))
            check_runtime(operation, atomic, boundaries[i], boundaries[j]);
      seed_stream(random, operation->name, atomic);
      phase = "random iteration";
      positionCount = iterations;
      for (unsigned long i = 0; i < iterations; ++i) {
        position = i + 1;
        random_operand(lhs, random);
        if (gmp_urandomm_ui(random, 4) == 0)
          related_operand(rhs, lhs, random);
        else
          random_operand(rhs, random);
        Operand x = {lhs, random_spare(random), NULL};
        Operand y = {rhs, random_spare(random), NULL};
        check_pair(operation, atomic, x, y, 0);
        check_aliased(operation, atomic, x);
        // A random constant replaces either operand.
        size_t index = gmp_urandomm_ui(random, CONSTANT_COUNT);
        Operand c = {constants[index], 0, constant[atomic](index)};
        if (gmp_urandomb_ui(random, 1))
          check_pair(operation, atomic, c, y, 0);
        else
          check_pair(operation, atomic, x, c, 0);
      }
    }
  }

  for (uint32_t atomic = 0; atomic < 2; ++atomic)
    check_chains(atomic, random);
  for (size_t index = 0;
       index < sizeof(fromIntOperations) / sizeof(*fromIntOperations); ++index)
    check_from_int_stream(&fromIntOperations[index], random, lhs);

  for (size_t i = 0; i < boundaryCount; ++i)
    mpz_clear(boundaries[i]);
  for (size_t i = 0; i < CONSTANT_COUNT; ++i)
    mpz_clear(constants[i]);
  mpz_clears(lhs, rhs, NULL);
  gmp_randclear(random);
  return 0;
}
