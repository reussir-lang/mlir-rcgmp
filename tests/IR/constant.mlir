// RUN: %rcgmp-opt %s | %rcgmp-opt | %FileCheck %s
// RUN: %rcgmp-opt %s -mlir-print-op-generic | %FileCheck %s --check-prefix=GENERIC
// RUN: %rcgmp-opt %s -cse | %FileCheck %s --check-prefix=CSE
// RUN: %rcgmp-opt %s -canonicalize | %FileCheck %s --check-prefix=CANON

// The attribute uses the narrowest signless type holding the value.
// CHECK-LABEL: func.func @values
// CHECK: rcgmp.constant 0 : !rcgmp.mpz
// CHECK: rcgmp.constant -1 : !rcgmp.mpz
// CHECK: rcgmp.constant 1 : !rcgmp.mpz
// CHECK: rcgmp.constant 18446744073709551615 : !rcgmp.mpz
// CHECK: rcgmp.constant -340282366920938463463374607431768211456 : <atomic>
// GENERIC: <{value = false}>
// GENERIC: <{value = true}>
// GENERIC: <{value = 1 : i2}>
// GENERIC: <{value = 18446744073709551615 : i65}>
// GENERIC: <{value = -340282366920938463463374607431768211456 : i129}>
func.func @values() -> (!rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz<atomic>) {
  %zero = rcgmp.constant 0 : !rcgmp.mpz
  %minus = rcgmp.constant -1 : !rcgmp.mpz
  %one = rcgmp.constant 1 : !rcgmp.mpz
  %hex = rcgmp.constant 0xffffffffffffffff : !rcgmp.mpz
  %min = rcgmp.constant -340282366920938463463374607431768211456 : !rcgmp.mpz<atomic>
  return %zero, %minus, %one, %hex, %min : !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz<atomic>
}

// Constants need no reference counting, so equal constants may merge
// into one value consumed twice, and unused constants may be erased.
// CSE-LABEL: func.func @merge
// CSE-NEXT: %[[C:.*]] = rcgmp.constant 36893488147419103232 : !rcgmp.mpz
// CSE-NEXT: %[[R:.*]] = rcgmp.addi %[[C]], %[[C]] : !rcgmp.mpz
// CSE-NEXT: return %[[R]]
func.func @merge() -> !rcgmp.mpz {
  %unused = rcgmp.constant 7 : !rcgmp.mpz
  %a = rcgmp.constant 36893488147419103232 : !rcgmp.mpz
  %b = rcgmp.constant 36893488147419103232 : !rcgmp.mpz
  %r = rcgmp.addi %a, %b : !rcgmp.mpz
  return %r : !rcgmp.mpz
}

// Upstream folds for i1 constants do not apply to integer constants.
// CANON-LABEL: func.func @select
// CANON: scf.if %{{.*}} -> (!rcgmp.mpz)
// CANON: rcgmp.constant 0 : !rcgmp.mpz
// CANON: rcgmp.constant -1 : !rcgmp.mpz
func.func @select(%p: i1) -> !rcgmp.mpz {
  %r = scf.if %p -> !rcgmp.mpz {
    %a = rcgmp.constant 0 : !rcgmp.mpz
    scf.yield %a : !rcgmp.mpz
  } else {
    %b = rcgmp.constant -1 : !rcgmp.mpz
    scf.yield %b : !rcgmp.mpz
  }
  return %r : !rcgmp.mpz
}
