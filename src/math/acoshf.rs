use super::{log1pf, logf, sqrtf};

const LN2: f32 = 0.693147180559945309417232121458176568;

/* acosh(x) = log(x + sqrt(x*x-1)) */
pub fn acoshf(x: f32) -> f32 {
    let u = x.to_bits();
    let a = u & 0x7fffffff;

    if a < 0x3f800000 + (1 << 23) {
        /* |x| < 2, invalid if x < 1 or nan */
        /* up to 2ulp error in [1,1.125] */
        return log1pf(x - 1.0 + sqrtf((x - 1.0) * (x - 1.0) + 2.0 * (x - 1.0)));
    }
    if a < 0x3f800000 + (12 << 23) {
        /* |x| < 0x1p12 */
        return logf(2.0 * x - 1.0 / (x + sqrtf(x * x - 1.0)));
    }
    /* x >= 0x1p12 */
    return logf(x) + LN2;
}

#[cfg(test)]
mod tests {

    // ---- acoshf_matches_musl stdout ----
    // thread 'main' panicked at 'INPUT: [1026245936] EXPECTED: [4290772992] ACTUAL 88.72196', /target/aarch64-unknown-linux-gnu/release/build/libm-11858a34fff673b0/out/musl-tests.rs:134:17
    // note: Run with `RUST_BACKTRACE=1` environment variable to display a backtrace.

    #[test]
    fn test_from_ci() {
        let ret = super::acoshf(f32::from_bits(1026245936));
        assert!(ret.is_nan() == f32::from_bits(4290772992).is_nan());
    }

    // is acosh(0.04181403) really 88.72196 or nan?
    // musl says it's nan
    // keisan says its 0 +1.52897010249848802665389305677i
    // #[test]
    // fn test_from_ci() {
    //     let ret = super::acoshf(f32::from_bits(1026245936));
    //     assert!(ret == 88.72196);
    // }
}
