#include "Test.h"

void *imm_max(void);
void *imm_min(void);
void *above_max(void);
void *above_max_atomic(void);
void *below_min(void);
void *wide(void);
void *one_limb_max(void);
void *int64_min(void);
void *limb_order(void);
typedef struct {
  void *first, *second;
} Pair;
Pair negatives(void);
void *unbalanced(void);
void *square(void);
void *add_constant(void *);
void *add_constant_atomic(void *);
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
void release_many(size_t);
void *accumulate_local(size_t);
void *accumulate_atomic(size_t);

static const char *AboveMax = "2305843009213693952";
static const char *Wide = "-340282366920938463463374607431768211456";

static int fits_immediate(mpz_t value) {
  return mpz_cmp_si(value, RCGMP_IMM_MIN) >= 0 &&
         mpz_cmp_si(value, RCGMP_IMM_MAX) <= 0;
}

static void expect(void *value, const char *decimal, uint32_t atomic) {
  mpz_t expected;
  mpz_init_set_str(expected, decimal, 10);
  int fits = fits_immediate(expected);
  // Values outside the immediate range are immortal objects whose header
  // matches a runtime allocation of exactly their magnitude.
  CHECK(rcgmp_is_immediate(value) == fits);
  if (!fits) {
    RCGMPMpz *heap = value;
    CHECK((uintptr_t)value % sizeof(void *) == 0);
    CHECK(rcgmp_is_immortal(heap, atomic));
    CHECK(heap->capacity ==
          (uintptr_t)(heap->size < 0 ? -heap->size : heap->size));
  }
  check_value(value, expected);
  mpz_clear(expected);
}

typedef struct {
  void *(*apply[2])(void *, void *);
  void (*oracle)(mpz_ptr, mpz_srcptr, mpz_srcptr);
} Binary;

static Binary operations[] = {
    {{addi_local, addi_atomic}, mpz_add}, {{subi_local, subi_atomic}, mpz_sub},
    {{muli_local, muli_atomic}, mpz_mul}, {{andi_local, andi_atomic}, mpz_and},
    {{xori_local, xori_atomic}, mpz_xor},
};
#define OPERATION_COUNT (sizeof(operations) / sizeof(*operations))

typedef struct {
  void *(*get)(void);
  const char *decimal;
} Constant;

// A constant object has a zero count in either mode, so each one serves as an
// operand of both modes.
static Constant constants[] = {
    {above_max, "2305843009213693952"},
    {below_min, "-2305843009213693953"},
    {one_limb_max, "18446744073709551615"},
    {wide, "-340282366920938463463374607431768211456"},
};
#define CONSTANT_COUNT (sizeof(constants) / sizeof(*constants))

static uintptr_t ref_count(RCGMPMpz *value, uint32_t atomic) {
  return atomic
             ? atomic_load_explicit(&value->refs.atomic, memory_order_relaxed)
             : value->refs.local;
}

static void expect_constants(uint32_t atomic) {
  for (size_t i = 0; i < CONSTANT_COUNT; ++i)
    expect(constants[i].get(), constants[i].decimal, atomic);
}

// Apply op to constant c and operand x, with c on the left or the right.
static void check_with_constant(Binary *op, Constant *c, void *x, mpz_t value,
                                int left, uint32_t atomic) {
  void *constant = c->get();
  mpz_t expected, other;
  mpz_init(expected);
  mpz_init_set_str(other, c->decimal, 10);
  void *result;
  if (left) {
    op->oracle(expected, other, value);
    result = op->apply[atomic](constant, x);
  } else {
    op->oracle(expected, value, other);
    result = op->apply[atomic](x, constant);
  }
  check_value(result, expected);
  CHECK(rcgmp_is_immediate(result) == fits_immediate(expected));
  // Adding zero, or subtracting it from the constant, hands the constant back.
  // A heap result is either an operand that is a constant or a fresh object
  // owned only by the caller.
  if (mpz_sgn(value) == 0 &&
      (op->oracle == mpz_add || (op->oracle == mpz_sub && left)))
    CHECK(result == constant);
  if (!rcgmp_is_immediate(result) && rcgmp_is_immortal(result, atomic))
    CHECK(result == constant || result == x);
  else if (!rcgmp_is_immediate(result))
    CHECK(ref_count(result, atomic) == 1);
  rcgmp_mpz_release(result, atomic);
  expect_constants(atomic);
  mpz_clears(expected, other, NULL);
}

static void check_constant_operands(void) {
  enum { Immediate, Unique, Shared };
  struct {
    int kind;
    intptr_t small;
    const char *decimal;
  } operands[] = {
      {Immediate, 0, NULL},
      {Immediate, 1, NULL},
      {Immediate, -1, NULL},
      {Immediate, RCGMP_IMM_MAX, NULL},
      {Immediate, RCGMP_IMM_MIN, NULL},
      {Unique, 0, "4611686018427387904"},
      {Unique, 0, "-18446744073709551621"},
      {Shared, 0, "4611686018427387904"},
  };
  mpz_t value;
  mpz_init(value);
  for (uint32_t atomic = 0; atomic < 2; ++atomic) {
    for (size_t op = 0; op < OPERATION_COUNT; ++op) {
      for (int left = 0; left < 2; ++left) {
        for (size_t c = 0; c < CONSTANT_COUNT; ++c) {
          Constant *constant = &constants[c];
          for (size_t i = 0; i < sizeof(operands) / sizeof(*operands); ++i) {
            if (operands[i].decimal)
              mpz_set_str(value, operands[i].decimal, 10);
            else
              mpz_set_si(value, operands[i].small);
            void *x = from_mpz(value, atomic);
            CHECK(rcgmp_is_immediate(x) == (operands[i].kind == Immediate));
            if (operands[i].kind == Shared)
              rcgmp_mpz_retain(x, atomic);
            check_with_constant(&operations[op], constant, x, value, left,
                                atomic);
            if (operands[i].kind == Shared) {
              check_value(x, value);
              rcgmp_mpz_release(x, atomic);
            }
          }
          // Two constants, including the same one twice.
          for (size_t i = 0; i < CONSTANT_COUNT; ++i) {
            mpz_set_str(value, constants[i].decimal, 10);
            check_with_constant(&operations[op], constant, constants[i].get(),
                                value, left, atomic);
          }
        }
      }
    }
  }
  mpz_clear(value);
}

static void check_repeated_counting(void) {
  size_t freed = deallocations;
  release_many(0);
  release_many(1);
  release_many(1000);
  CHECK(deallocations == freed);
  for (uint32_t atomic = 0; atomic < 2; ++atomic) {
    for (size_t i = 0; i < CONSTANT_COUNT; ++i) {
      void *c = constants[i].get();
      for (int n = 0; n < 1000; ++n)
        CHECK(rcgmp_mpz_retain(c, atomic) == c);
      for (int n = 0; n < 1000; ++n)
        rcgmp_mpz_release(c, atomic);
      CHECK(constants[i].get() == c);
    }
    expect_constants(atomic);
  }
  CHECK(deallocations == freed);

  size_t counts[] = {0, 1, 1000};
  for (uint32_t atomic = 0; atomic < 2; ++atomic) {
    for (size_t i = 0; i < sizeof(counts) / sizeof(*counts); ++i) {
      mpz_t expected;
      mpz_init_set_str(expected, atomic ? Wide : "18446744073709551615", 10);
      mpz_mul(expected, expected, expected);
      mpz_mul_ui(expected, expected, counts[i]);
      void *result =
          atomic ? accumulate_atomic(counts[i]) : accumulate_local(counts[i]);
      check_value(result, expected);
      CHECK(rcgmp_is_immediate(result) == fits_immediate(expected));
      rcgmp_mpz_release(result, atomic);
      mpz_clear(expected);
    }
    expect_constants(atomic);
  }
}

int main(void) {
  expect(imm_max(), "2305843009213693951", 0);
  expect(imm_min(), "-2305843009213693952", 0);
  expect(above_max(), AboveMax, 0);
  expect(above_max_atomic(), AboveMax, 1);
  CHECK(above_max() == above_max_atomic());
  expect(below_min(), "-2305843009213693953", 0);
  expect(wide(), Wide, 1);
  expect(one_limb_max(), "18446744073709551615", 1);
  expect(int64_min(), "-9223372036854775808", 0);
  expect(limb_order(), "55340232221128654850", 0);
  Pair small = negatives();
  expect(small.first, "-1", 0);
  expect(small.second, "-5", 0);

  void *constant = unbalanced();
  CHECK(constant == above_max());
  expect(constant, AboveMax, 0);

  mpz_t expected, operand;
  mpz_inits(expected, operand, NULL);
  mpz_set_str(expected, Wide, 10);
  mpz_mul(expected, expected, expected);
  void *result = square();
  check_value(result, expected);
  rcgmp_mpz_release(result, 0);

  // A constant is never unique, so the runtime does not write into it, even
  // when the other operand is a small immediate or a unique heap value.
  const char *operands[] = {"-3", "3", "-18446744073709551621"};
  for (uint32_t atomic = 0; atomic < 2; ++atomic) {
    const char *decimal = atomic ? Wide : AboveMax;
    for (size_t i = 0; i < sizeof(operands) / sizeof(*operands); ++i) {
      mpz_set_str(operand, operands[i], 10);
      mpz_set_str(expected, decimal, 10);
      mpz_add(expected, expected, operand);
      void *x = from_mpz(operand, atomic);
      result = atomic ? add_constant_atomic(x) : add_constant(x);
      check_value(result, expected);
      rcgmp_mpz_release(result, atomic);
      expect(atomic ? wide() : above_max(), decimal, atomic);
    }
  }
  check_constant_operands();
  check_repeated_counting();
  CHECK(allocations == deallocations);
  mpz_clears(expected, operand, NULL);
  return 0;
}
