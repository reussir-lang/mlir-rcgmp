#include "Test.h"

void *imm_max(void);
void *imm_min(void);
void *above_max(void);
void *above_max_atomic(void);
void *below_min(void);
void *wide(void);
void *unbalanced(void);
void *square(void);
void *add_constant(void *);
void *add_constant_atomic(void *);

static const char *AboveMax = "2305843009213693952";
static const char *Wide = "-340282366920938463463374607431768211456";

static void expect(void *value, const char *decimal, uint32_t atomic) {
  mpz_t expected;
  mpz_init_set_str(expected, decimal, 10);
  int fits = mpz_cmp_si(expected, RCGMP_IMM_MIN) >= 0 &&
             mpz_cmp_si(expected, RCGMP_IMM_MAX) <= 0;
  // Values outside the immediate range are immortal objects.
  CHECK(rcgmp_is_immediate(value) == fits);
  CHECK(fits || rcgmp_is_immortal(value, atomic));
  check_value(value, expected);
  mpz_clear(expected);
}

int main(void) {
  expect(imm_max(), "2305843009213693951", 0);
  expect(imm_min(), "-2305843009213693952", 0);
  expect(above_max(), AboveMax, 0);
  expect(above_max_atomic(), AboveMax, 1);
  CHECK(above_max() == above_max_atomic());
  expect(below_min(), "-2305843009213693953", 0);
  expect(wide(), Wide, 1);

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
  CHECK(allocations == deallocations);
  mpz_clears(expected, operand, NULL);
  return 0;
}
