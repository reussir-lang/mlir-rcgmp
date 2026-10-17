// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf | %FileCheck %s --check-prefix=SCF
// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf -convert-scf-to-cf -convert-to-llvm -reconcile-unrealized-casts | %FileCheck %s --check-prefix=LLVM

// Equal values share one read-only object for both reference-counting modes.
// LLVM-DAG: llvm.mlir.global private unnamed_addr constant @__rcgmp_mpz_2000000000000000(dense<[0, 1, 1, 2305843009213693952]> : tensor<4xi64>) {addr_space = 0 : i32, alignment = 8 : i64} : !llvm.array<4 x i64>
// LLVM-DAG: llvm.mlir.global private unnamed_addr constant @__rcgmp_mpz_neg_2000000000000001(dense<[0, -1, 1, 2305843009213693953]> : tensor<4xi64>)
// LLVM-DAG: llvm.mlir.global private unnamed_addr constant @__rcgmp_mpz_neg_100000000000000000000000000000000(dense<[0, -3, 3, 0, 0, 1]> : tensor<6xi64>)

// SCF-LABEL: func.func @imm_max
// SCF: %[[BITS:.*]] = arith.constant 9223372036854775805 : i64
// SCF: rcgmp.bridge.mpz.from_bits %[[BITS]] : i64 to !rcgmp.mpz
func.func @imm_max() -> !rcgmp.mpz {
  %c = rcgmp.constant 2305843009213693951 : !rcgmp.mpz
  return %c : !rcgmp.mpz
}
// SCF-LABEL: func.func @imm_min
// SCF: arith.constant -9223372036854775807 : i64
func.func @imm_min() -> !rcgmp.mpz {
  %c = rcgmp.constant -2305843009213693952 : !rcgmp.mpz
  return %c : !rcgmp.mpz
}
// SCF-LABEL: func.func @above_max
// SCF: rcgmp.bridge.mpz.immortal 2305843009213693952 : i63 : !rcgmp.mpz
// LLVM-LABEL: llvm.func @above_max
// LLVM: llvm.mlir.addressof @__rcgmp_mpz_2000000000000000 : !llvm.ptr
func.func @above_max() -> !rcgmp.mpz {
  %c = rcgmp.constant 2305843009213693952 : !rcgmp.mpz
  return %c : !rcgmp.mpz
}
// LLVM-LABEL: llvm.func @above_max_atomic
// LLVM: llvm.mlir.addressof @__rcgmp_mpz_2000000000000000 : !llvm.ptr
func.func @above_max_atomic() -> !rcgmp.mpz<atomic> {
  %c = rcgmp.constant 2305843009213693952 : !rcgmp.mpz<atomic>
  return %c : !rcgmp.mpz<atomic>
}
func.func @below_min() -> !rcgmp.mpz {
  %c = rcgmp.constant -2305843009213693953 : !rcgmp.mpz
  return %c : !rcgmp.mpz
}
func.func @wide() -> !rcgmp.mpz<atomic> {
  %c = rcgmp.constant -340282366920938463463374607431768211456 : !rcgmp.mpz<atomic>
  return %c : !rcgmp.mpz<atomic>
}

// Constants may be retained, released, and consumed without balancing.
func.func @unbalanced() -> !rcgmp.mpz {
  %c = rcgmp.constant 2305843009213693952 : !rcgmp.mpz
  rcgmp.retain %c : !rcgmp.mpz
  rcgmp.release %c : !rcgmp.mpz
  rcgmp.release %c : !rcgmp.mpz
  return %c : !rcgmp.mpz
}
func.func @square() -> !rcgmp.mpz {
  %c = rcgmp.constant -340282366920938463463374607431768211456 : !rcgmp.mpz
  %r = rcgmp.muli %c, %c : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @add_constant(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %c = rcgmp.constant 2305843009213693952 : !rcgmp.mpz
  %r = rcgmp.addi %c, %x : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @add_constant_atomic(%x: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %c = rcgmp.constant -340282366920938463463374607431768211456 : !rcgmp.mpz<atomic>
  %r = rcgmp.addi %x, %c : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
