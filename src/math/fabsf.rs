#[inline]
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn fabsf(x: f32) -> f32 {
    // On wasm32 we know that LLVM's intrinsic will compile to an optimized
    // `f32.abs` native instruction, so we can leverage this for both code size
    // and speed.
    llvm_intrinsically_optimized! {
        #[cfg(target_arch = "wasm32", target_arch = "arm")] {
            return unsafe { ::core::intrinsics::fabsf32(x) }
        }
    }
    f32::from_bits(x.to_bits() & 0x7fffffff)
}
