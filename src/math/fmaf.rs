// Copyright (c) 2019 Benjamin Schultzer

// Permission is hereby granted, free of charge, to any
// person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the
// Software without restriction, including without
// limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software
// is furnished to do so, subject to the following
// conditions:

// The above copyright notice and this permission notice
// shall be included in all copies or substantial portions
// of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
// ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
// TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
// PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
// SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
// IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

// This algorithm is based on Dekker's TwoProduct
// S. M. Rump, T. Ogita, and S. Oishi, Accurate floating-point summation part I: faithful rounding, SIAM J. Sci. Comput., 31 (2008), pp. 189–224.
// and this FMSF https://stackoverflow.com/a/30121217

/// Fused multiply-add Compute x * y + z
#[inline]
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn fmaf(x: f32, y: f32, z: f32) -> f32 {
    // TODO is a summation algorithm necessary?
    // If so we could use either FastAccSum or FastPrecSum http://www.ti3.tu-harburg.de/paper/rump/Ru08b.pdf,
    // or https://en.wikipedia.org/wiki/Pairwise_summation which is usally used in FFT.
    let (hx, lx) = split(x);
    let (hy, ly) = split(y);
    ((hx * hy + z) + hx * ly + lx) + lx * ly
}

#[inline]
fn split(x: f32) -> (f32, f32) {
    let t = 1e0 * x; // FMSF uses (1 << 12) + 1) * x == 4.097e3 * x; we use (0 << 12) + 1) * x == 1e0 * x
    let hi = t - (t - x);
    let lo = x - hi;
    (hi, lo)
}

#[cfg(test)]
mod tests {
    use core::f32::{INFINITY, MAX, MIN_POSITIVE, NAN, NEG_INFINITY};
    use rand::Rng;
    extern "C" {
        pub fn fmaf(x: f32, y: f32, z: f32) -> f32;
    }

    pub const F32_MIN_SUBNORM: f32 = 1.401298464324817070923730e-45;

    pub fn equal(x: f32, y: f32) -> bool {
        if __equal__(x, y, 1) {
            return true;
        }
        panic!("X: {} Y: {}", x, y);
    }

    pub fn __equal__(x: f32, y: f32, ulp: i32) -> bool {
        if x.is_nan() != y.is_nan() {
            // one is nan but the other is not
            return false;
        }
        if x.is_nan() && y.is_nan() {
            return true;
        }
        if x.is_infinite() != y.is_infinite() {
            // one is inf but the other is not
            return false;
        }
        let xi: i32 = unsafe { core::intrinsics::transmute(x) };
        let yi: i32 = unsafe { core::intrinsics::transmute(y) };
        if (xi < 0) != (yi < 0) {
            // different sign
            return false;
        }
        let ulps = (xi - yi).abs();
        ulps <= ulp
    }

    #[test]
    fn validation() {
        let mut t = 0.;
        let mut ief = 0.;
        let mut ies = 0.;
        let mut ef = 0.;
        let mut es = 0.;
        let mut r = rand::thread_rng();
        for _i in 0..10000 {
            t += 1.;
            let x = r.gen::<f32>();
            let y = r.gen::<f32>();
            let z = r.gen::<f32>();
            let expected = unsafe { fmaf(x, y, z) };
            let result = super::super::fmaf(x, y, z);
            if !__equal__(expected, result, 1) {
                ief += 1.;
            } else {
                ies += 1.;
            }
            if !__equal__(expected, result, 0) {
                ef += 1.;
            } else {
                es += 1.;
            }
        }
        let exact: f64 = (es / t) * 100.;
        let exact_failure: f64 = (ef / t) * 100.;
        let inexact: f64 = (ies / t) * 100.;
        let inexact_failure: f64 = (ief / t) * 100.;
        panic!("OUT OF {} TESTS | {}% EXACT MACTHES | {}% EXACT FAILURES | {}% INEXACT MATCHES | {}% INEXACT FAILURES", t, exact, exact_failure, inexact, inexact_failure);
    }
    #[test]
    pub fn test_const() {
        assert!(equal(
            unsafe { fmaf(NAN, 2., 3.) },
            super::super::fmaf(NAN, 2., 3.)
        ));
        assert!(equal(
            unsafe { fmaf(-NAN, 2., 3.) },
            super::super::fmaf(-NAN, 2., 3.)
        ));
        assert!(equal(
            unsafe { fmaf(NAN, 2., 3.) },
            super::super::fmaf(NAN, 2., 3.)
        ));
        assert!(equal(
            unsafe { fmaf(-NAN, 2., 3.) },
            super::super::fmaf(-NAN, 2., 3.)
        ));
        assert!(equal(
            unsafe { fmaf(1., NAN, 3.) },
            super::super::fmaf(1., NAN, 3.)
        ));
        assert!(equal(
            unsafe { fmaf(1., -NAN, 3.) },
            super::super::fmaf(1., -NAN, 3.)
        ));
        assert!(equal(
            unsafe { fmaf(1., NAN, 3.0) },
            super::super::fmaf(1., NAN, 3.0)
        ));
        assert!(equal(
            unsafe { fmaf(1., -NAN, 3.0) },
            super::super::fmaf(1., -NAN, 3.0)
        ));
        assert!(equal(
            unsafe { fmaf(1., 2., NAN) },
            super::super::fmaf(1., 2., NAN)
        ));
        assert!(equal(
            unsafe { fmaf(1., 2., -NAN) },
            super::super::fmaf(1., 2., -NAN)
        ));
        assert!(equal(
            unsafe { fmaf(1., 2., NAN) },
            super::super::fmaf(1., 2., NAN)
        ));
        assert!(equal(
            unsafe { fmaf(1., 2., -NAN) },
            super::super::fmaf(1., 2., -NAN)
        ));
        assert!(equal(
            unsafe { fmaf(MAX, MAX, NAN) },
            super::super::fmaf(MAX, MAX, NAN)
        ));
        assert!(equal(
            unsafe { fmaf(MAX, MAX, -NAN) },
            super::super::fmaf(MAX, MAX, -NAN)
        ));
        assert!(equal(
            unsafe { fmaf(F32_MIN_SUBNORM, F32_MIN_SUBNORM, NAN) },
            super::super::fmaf(F32_MIN_SUBNORM, F32_MIN_SUBNORM, NAN)
        ));
        assert!(equal(
            unsafe { fmaf(F32_MIN_SUBNORM, F32_MIN_SUBNORM, -NAN) },
            super::super::fmaf(F32_MIN_SUBNORM, F32_MIN_SUBNORM, -NAN)
        ));
        assert!(equal(
            unsafe { fmaf(MIN_POSITIVE, MIN_POSITIVE, NAN) },
            super::super::fmaf(MIN_POSITIVE, MIN_POSITIVE, NAN)
        ));
        assert!(equal(
            unsafe { fmaf(MIN_POSITIVE, MIN_POSITIVE, -NAN) },
            super::super::fmaf(MIN_POSITIVE, MIN_POSITIVE, -NAN)
        ));
        assert!(equal(
            unsafe { fmaf(NAN, NAN, NAN) },
            super::super::fmaf(NAN, NAN, NAN)
        ));
        assert!(equal(
            unsafe { fmaf(NAN, NAN, -NAN) },
            super::super::fmaf(NAN, NAN, -NAN)
        ));
        assert!(equal(
            unsafe { fmaf(NAN, -NAN, NAN) },
            super::super::fmaf(NAN, -NAN, NAN)
        ));
        assert!(equal(
            unsafe { fmaf(NAN, -NAN, -NAN) },
            super::super::fmaf(NAN, -NAN, -NAN)
        ));
        assert!(equal(
            unsafe { fmaf(-NAN, NAN, NAN) },
            super::super::fmaf(-NAN, NAN, NAN)
        ));
        assert!(equal(
            unsafe { fmaf(-NAN, NAN, -NAN) },
            super::super::fmaf(-NAN, NAN, -NAN)
        ));
        assert!(equal(
            unsafe { fmaf(-NAN, -NAN, NAN) },
            super::super::fmaf(-NAN, -NAN, NAN)
        ));
        assert!(equal(
            unsafe { fmaf(-NAN, -NAN, -NAN) },
            super::super::fmaf(-NAN, -NAN, -NAN)
        ));
        assert!(equal(
            unsafe { fmaf(1., NAN, NAN) },
            super::super::fmaf(1., NAN, NAN)
        ));
        assert!(equal(
            unsafe { fmaf(1., NAN, -NAN) },
            super::super::fmaf(1., NAN, -NAN)
        ));
        assert!(equal(
            unsafe { fmaf(1., -NAN, NAN) },
            super::super::fmaf(1., -NAN, NAN)
        ));
        assert!(equal(
            unsafe { fmaf(1., -NAN, -NAN) },
            super::super::fmaf(1., -NAN, -NAN)
        ));
        assert!(equal(
            unsafe { fmaf(NAN, 2., NAN) },
            super::super::fmaf(NAN, 2., NAN)
        ));
        assert!(equal(
            unsafe { fmaf(NAN, 2., -NAN) },
            super::super::fmaf(NAN, 2., -NAN)
        ));
        assert!(equal(
            unsafe { fmaf(-NAN, 2., NAN) },
            super::super::fmaf(-NAN, 2., NAN)
        ));
        assert!(equal(
            unsafe { fmaf(-NAN, 2., -NAN) },
            super::super::fmaf(-NAN, 2., -NAN)
        ));
        assert!(equal(
            unsafe { fmaf(NAN, NAN, 3.) },
            super::super::fmaf(NAN, NAN, 3.)
        ));
        assert!(equal(
            unsafe { fmaf(NAN, -NAN, 3.) },
            super::super::fmaf(NAN, -NAN, 3.)
        ));
        assert!(equal(
            unsafe { fmaf(-NAN, NAN, 3.) },
            super::super::fmaf(-NAN, NAN, 3.)
        ));
        assert!(equal(
            unsafe { fmaf(-NAN, -NAN, 3.) },
            super::super::fmaf(-NAN, -NAN, 3.)
        ));
        assert!(equal(
            unsafe { fmaf(INFINITY, 0., NAN) },
            super::super::fmaf(INFINITY, 0., NAN)
        ));
        assert!(equal(
            unsafe { fmaf(INFINITY, 0., -NAN) },
            super::super::fmaf(INFINITY, 0., -NAN)
        ));
        assert!(equal(
            unsafe { fmaf(NEG_INFINITY, 0., NAN) },
            super::super::fmaf(NEG_INFINITY, 0., NAN)
        ));
        assert!(equal(
            unsafe { fmaf(NEG_INFINITY, 0., -NAN) },
            super::super::fmaf(NEG_INFINITY, 0., -NAN)
        ));
        assert!(equal(
            unsafe { fmaf(0., INFINITY, NAN) },
            super::super::fmaf(0., INFINITY, NAN)
        ));
        assert!(equal(
            unsafe { fmaf(0., INFINITY, -NAN) },
            super::super::fmaf(0., INFINITY, -NAN)
        ));
        assert!(equal(
            unsafe { fmaf(0., NEG_INFINITY, NAN) },
            super::super::fmaf(0., NEG_INFINITY, NAN)
        ));
        assert!(equal(
            unsafe { fmaf(0., NEG_INFINITY, -NAN) },
            super::super::fmaf(0., NEG_INFINITY, -NAN)
        ));

        /* Bug 6801: errno setting may be missing.  */
        assert!(equal(
            unsafe { fmaf(INFINITY, 0., 1.) },
            super::super::fmaf(INFINITY, 0., 1.)
        ));
        assert!(equal(
            unsafe { fmaf(NEG_INFINITY, 0., 1.) },
            super::super::fmaf(NEG_INFINITY, 0., 1.)
        ));
        assert!(equal(
            unsafe { fmaf(0., INFINITY, 1.) },
            super::super::fmaf(0., INFINITY, 1.)
        ));
        assert!(equal(
            unsafe { fmaf(0., NEG_INFINITY, 1.) },
            super::super::fmaf(0., NEG_INFINITY, 1.)
        ));

        assert!(equal(
            unsafe { fmaf(INFINITY, INFINITY, NEG_INFINITY) },
            super::super::fmaf(INFINITY, INFINITY, NEG_INFINITY)
        ));
        assert!(equal(
            unsafe { fmaf(NEG_INFINITY, INFINITY, INFINITY) },
            super::super::fmaf(NEG_INFINITY, INFINITY, INFINITY)
        ));
        assert!(equal(
            unsafe { fmaf(INFINITY, NEG_INFINITY, INFINITY) },
            super::super::fmaf(INFINITY, NEG_INFINITY, INFINITY)
        ));
        assert!(equal(
            unsafe { fmaf(NEG_INFINITY, NEG_INFINITY, NEG_INFINITY) },
            super::super::fmaf(NEG_INFINITY, NEG_INFINITY, NEG_INFINITY)
        ));
        assert!(equal(
            unsafe { fmaf(INFINITY, 3.5, NEG_INFINITY) },
            super::super::fmaf(INFINITY, 3.5, NEG_INFINITY)
        ));
        assert!(equal(
            unsafe { fmaf(NEG_INFINITY, -7.5, NEG_INFINITY) },
            super::super::fmaf(NEG_INFINITY, -7.5, NEG_INFINITY)
        ));
        assert!(equal(
            unsafe { fmaf(-13.5, INFINITY, INFINITY) },
            super::super::fmaf(-13.5, INFINITY, INFINITY)
        ));
        assert!(equal(
            unsafe { fmaf(NEG_INFINITY, 7.5, INFINITY) },
            super::super::fmaf(NEG_INFINITY, 7.5, INFINITY)
        ));

        // assert!(equal(unsafe { fmaf(-MAX, -MAX, NEG_INFINITY) }, super::super::fmaf(-MAX, -MAX, NEG_INFINITY)));
        // assert!(equal(unsafe { fmaf(MAX / 2., MAX / 2., NEG_INFINITY) }, super::super::fmaf(MAX / 2., MAX / 2., NEG_INFINITY)));
        // assert!(equal(unsafe { fmaf(-MAX, MAX, INFINITY) }, super::super::fmaf(-MAX, MAX, INFINITY)));
        // assert!(equal(unsafe { fmaf(MAX / 2., -MAX / 4., INFINITY) }, super::super::fmaf(MAX / 2., -MAX / 4., INFINITY)));
        // assert!(equal(unsafe { fmaf(INFINITY, 4., INFINITY) }, super::super::fmaf(INFINITY, 4., INFINITY)));
        // assert!(equal(unsafe { fmaf(2., NEG_INFINITY, NEG_INFINITY) }, super::super::fmaf(2., NEG_INFINITY, NEG_INFINITY)));
        // assert!(equal(unsafe { fmaf(INFINITY, INFINITY, INFINITY) }, super::super::fmaf(INFINITY, INFINITY, INFINITY)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, NEG_INFINITY, INFINITY) }, super::super::fmaf(NEG_INFINITY, NEG_INFINITY, INFINITY)));
        // assert!(equal(unsafe { fmaf(INFINITY, NEG_INFINITY, NEG_INFINITY) }, super::super::fmaf(INFINITY, NEG_INFINITY, NEG_INFINITY)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, INFINITY, NEG_INFINITY) }, super::super::fmaf(NEG_INFINITY, INFINITY, NEG_INFINITY)));

        // assert!(equal(unsafe { fmaf(INFINITY, INFINITY, 0.) }, super::super::fmaf(INFINITY, INFINITY, 0.)));
        // assert!(equal(unsafe { fmaf(INFINITY, INFINITY, -0.) }, super::super::fmaf(INFINITY, INFINITY, -0.)));
        // assert!(equal(unsafe { fmaf(INFINITY, INFINITY, F32_MIN_SUBNORM) }, super::super::fmaf(INFINITY, INFINITY, F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(INFINITY, INFINITY, -F32_MIN_SUBNORM) }, super::super::fmaf(INFINITY, INFINITY, -F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(INFINITY, INFINITY, MIN_POSITIVE) }, super::super::fmaf(INFINITY, INFINITY, MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(INFINITY, INFINITY, -MIN_POSITIVE) }, super::super::fmaf(INFINITY, INFINITY, -MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(INFINITY, INFINITY, MAX) }, super::super::fmaf(INFINITY, INFINITY, MAX)));
        // assert!(equal(unsafe { fmaf(INFINITY, INFINITY, -MAX) }, super::super::fmaf(INFINITY, INFINITY, -MAX)));
        // assert!(equal(unsafe { fmaf(INFINITY, NEG_INFINITY, 0.) }, super::super::fmaf(INFINITY, NEG_INFINITY, 0.)));
        // assert!(equal(unsafe { fmaf(INFINITY, NEG_INFINITY, -0.) }, super::super::fmaf(INFINITY, NEG_INFINITY, -0.)));
        // assert!(equal(unsafe { fmaf(INFINITY, NEG_INFINITY, F32_MIN_SUBNORM) }, super::super::fmaf(INFINITY, NEG_INFINITY, F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(INFINITY, NEG_INFINITY, -F32_MIN_SUBNORM) }, super::super::fmaf(INFINITY, NEG_INFINITY, -F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(INFINITY, NEG_INFINITY, MIN_POSITIVE) }, super::super::fmaf(INFINITY, NEG_INFINITY, MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(INFINITY, NEG_INFINITY, -MIN_POSITIVE) }, super::super::fmaf(INFINITY, NEG_INFINITY, -MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(INFINITY, NEG_INFINITY, MAX) }, super::super::fmaf(INFINITY, NEG_INFINITY, MAX)));
        // assert!(equal(unsafe { fmaf(INFINITY, NEG_INFINITY, -MAX) }, super::super::fmaf(INFINITY, NEG_INFINITY, -MAX)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, INFINITY, 0.) }, super::super::fmaf(NEG_INFINITY, INFINITY, 0.)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, INFINITY, -0.) }, super::super::fmaf(NEG_INFINITY, INFINITY, -0.)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, INFINITY, F32_MIN_SUBNORM) }, super::super::fmaf(NEG_INFINITY, INFINITY, F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, INFINITY, -F32_MIN_SUBNORM) }, super::super::fmaf(NEG_INFINITY, INFINITY, -F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, INFINITY, MIN_POSITIVE) }, super::super::fmaf(NEG_INFINITY, INFINITY, MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, INFINITY, -MIN_POSITIVE) }, super::super::fmaf(NEG_INFINITY, INFINITY, -MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, INFINITY, MAX) }, super::super::fmaf(NEG_INFINITY, INFINITY, MAX)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, INFINITY, -MAX) }, super::super::fmaf(NEG_INFINITY, INFINITY, -MAX)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, NEG_INFINITY, 0.) }, super::super::fmaf(NEG_INFINITY, NEG_INFINITY, 0.)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, NEG_INFINITY, -0.) }, super::super::fmaf(NEG_INFINITY, NEG_INFINITY, -0.)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, NEG_INFINITY, F32_MIN_SUBNORM) }, super::super::fmaf(NEG_INFINITY, NEG_INFINITY, F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, NEG_INFINITY, -F32_MIN_SUBNORM) }, super::super::fmaf(NEG_INFINITY, NEG_INFINITY, -F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, NEG_INFINITY, MIN_POSITIVE) }, super::super::fmaf(NEG_INFINITY, NEG_INFINITY, MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, NEG_INFINITY, -MIN_POSITIVE) }, super::super::fmaf(NEG_INFINITY, NEG_INFINITY, -MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, NEG_INFINITY, MAX) }, super::super::fmaf(NEG_INFINITY, NEG_INFINITY, MAX)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, NEG_INFINITY, -MAX) }, super::super::fmaf(NEG_INFINITY, NEG_INFINITY, -MAX)));

        // assert!(equal(unsafe { fmaf(INFINITY, MAX, 0.) }, super::super::fmaf(INFINITY, MAX, 0.)));
        // assert!(equal(unsafe { fmaf(INFINITY, MAX, -0.) }, super::super::fmaf(INFINITY, MAX, -0.)));
        // assert!(equal(unsafe { fmaf(INFINITY, MAX, F32_MIN_SUBNORM) }, super::super::fmaf(INFINITY, MAX, F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(INFINITY, MAX, -F32_MIN_SUBNORM) }, super::super::fmaf(INFINITY, MAX, -F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(INFINITY, MAX, MIN_POSITIVE) }, super::super::fmaf(INFINITY, MAX, MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(INFINITY, MAX, -MIN_POSITIVE) }, super::super::fmaf(INFINITY, MAX, -MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(INFINITY, MAX, MAX) }, super::super::fmaf(INFINITY, MAX, MAX)));
        // assert!(equal(unsafe { fmaf(INFINITY, MAX, -MAX) }, super::super::fmaf(INFINITY, MAX, -MAX)));
        // assert!(equal(unsafe { fmaf(INFINITY, MIN_POSITIVE, 0.) }, super::super::fmaf(INFINITY, MIN_POSITIVE, 0.)));
        // assert!(equal(unsafe { fmaf(INFINITY, MIN_POSITIVE, -0.) }, super::super::fmaf(INFINITY, MIN_POSITIVE, -0.)));
        // assert!(equal(unsafe { fmaf(INFINITY, MIN_POSITIVE, F32_MIN_SUBNORM) }, super::super::fmaf(INFINITY, MIN_POSITIVE, F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(INFINITY, MIN_POSITIVE, -F32_MIN_SUBNORM) }, super::super::fmaf(INFINITY, MIN_POSITIVE, -F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(INFINITY, MIN_POSITIVE, MIN_POSITIVE) }, super::super::fmaf(INFINITY, MIN_POSITIVE, MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(INFINITY, MIN_POSITIVE, -MIN_POSITIVE) }, super::super::fmaf(INFINITY, MIN_POSITIVE, -MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(INFINITY, MIN_POSITIVE, MAX) }, super::super::fmaf(INFINITY, MIN_POSITIVE, MAX)));
        // assert!(equal(unsafe { fmaf(INFINITY, MIN_POSITIVE, -MAX) }, super::super::fmaf(INFINITY, MIN_POSITIVE, -MAX)));
        // assert!(equal(unsafe { fmaf(INFINITY, F32_MIN_SUBNORM, 0.) }, super::super::fmaf(INFINITY, F32_MIN_SUBNORM, 0.)));
        // assert!(equal(unsafe { fmaf(INFINITY, F32_MIN_SUBNORM, -0.) }, super::super::fmaf(INFINITY, F32_MIN_SUBNORM, -0.)));
        // assert!(equal(unsafe { fmaf(INFINITY, F32_MIN_SUBNORM, F32_MIN_SUBNORM) }, super::super::fmaf(INFINITY, F32_MIN_SUBNORM, F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(INFINITY, F32_MIN_SUBNORM, -F32_MIN_SUBNORM) }, super::super::fmaf(INFINITY, F32_MIN_SUBNORM, -F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(INFINITY, F32_MIN_SUBNORM, MIN_POSITIVE) }, super::super::fmaf(INFINITY, F32_MIN_SUBNORM, MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(INFINITY, F32_MIN_SUBNORM, -MIN_POSITIVE) }, super::super::fmaf(INFINITY, F32_MIN_SUBNORM, -MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(INFINITY, F32_MIN_SUBNORM, MAX) }, super::super::fmaf(INFINITY, F32_MIN_SUBNORM, MAX)));
        // assert!(equal(unsafe { fmaf(INFINITY, F32_MIN_SUBNORM, -MAX) }, super::super::fmaf(INFINITY, F32_MIN_SUBNORM, -MAX)));
        // assert!(equal(unsafe { fmaf(INFINITY, -MAX, 0.) }, super::super::fmaf(INFINITY, -MAX, 0.)));
        // assert!(equal(unsafe { fmaf(INFINITY, -MAX, -0.) }, super::super::fmaf(INFINITY, -MAX, -0.)));
        // assert!(equal(unsafe { fmaf(INFINITY, -MAX, F32_MIN_SUBNORM) }, super::super::fmaf(INFINITY, -MAX, F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(INFINITY, -MAX, -F32_MIN_SUBNORM) }, super::super::fmaf(INFINITY, -MAX, -F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(INFINITY, -MAX, MIN_POSITIVE) }, super::super::fmaf(INFINITY, -MAX, MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(INFINITY, -MAX, -MIN_POSITIVE) }, super::super::fmaf(INFINITY, -MAX, -MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(INFINITY, -MAX, MAX) }, super::super::fmaf(INFINITY, -MAX, MAX)));
        // assert!(equal(unsafe { fmaf(INFINITY, -MAX, -MAX) }, super::super::fmaf(INFINITY, -MAX, -MAX)));
        // assert!(equal(unsafe { fmaf(INFINITY, -F32_MIN_SUBNORM, 0.) }, super::super::fmaf(INFINITY, -F32_MIN_SUBNORM, 0.)));
        // assert!(equal(unsafe { fmaf(INFINITY, -F32_MIN_SUBNORM, -0.) }, super::super::fmaf(INFINITY, -F32_MIN_SUBNORM, -0.)));
        // assert!(equal(unsafe { fmaf(INFINITY, -F32_MIN_SUBNORM, F32_MIN_SUBNORM) }, super::super::fmaf(INFINITY, -F32_MIN_SUBNORM, F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(INFINITY, -F32_MIN_SUBNORM, -F32_MIN_SUBNORM) }, super::super::fmaf(INFINITY, -F32_MIN_SUBNORM, -F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(INFINITY, -F32_MIN_SUBNORM, MIN_POSITIVE) }, super::super::fmaf(INFINITY, -F32_MIN_SUBNORM, MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(INFINITY, -F32_MIN_SUBNORM, -MIN_POSITIVE) }, super::super::fmaf(INFINITY, -F32_MIN_SUBNORM, -MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(INFINITY, -F32_MIN_SUBNORM, MAX) }, super::super::fmaf(INFINITY, -F32_MIN_SUBNORM, MAX)));
        // assert!(equal(unsafe { fmaf(INFINITY, -F32_MIN_SUBNORM, -MAX) }, super::super::fmaf(INFINITY, -F32_MIN_SUBNORM, -MAX)));
        // assert!(equal(unsafe { fmaf(INFINITY, -MIN_POSITIVE, 0.) }, super::super::fmaf(INFINITY, -MIN_POSITIVE, 0.)));
        // assert!(equal(unsafe { fmaf(INFINITY, -MIN_POSITIVE, -0.) }, super::super::fmaf(INFINITY, -MIN_POSITIVE, -0.)));
        // assert!(equal(unsafe { fmaf(INFINITY, -MIN_POSITIVE, F32_MIN_SUBNORM) }, super::super::fmaf(INFINITY, -MIN_POSITIVE, F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(INFINITY, -MIN_POSITIVE, -F32_MIN_SUBNORM) }, super::super::fmaf(INFINITY, -MIN_POSITIVE, -F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(INFINITY, -MIN_POSITIVE, MIN_POSITIVE) }, super::super::fmaf(INFINITY, -MIN_POSITIVE, MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(INFINITY, -MIN_POSITIVE, -MIN_POSITIVE) }, super::super::fmaf(INFINITY, -MIN_POSITIVE, -MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(INFINITY, -MIN_POSITIVE, MAX) }, super::super::fmaf(INFINITY, -MIN_POSITIVE, MAX)));
        // assert!(equal(unsafe { fmaf(INFINITY, -MIN_POSITIVE, -MAX) }, super::super::fmaf(INFINITY, -MIN_POSITIVE, -MAX)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, MAX, 0.) }, super::super::fmaf(NEG_INFINITY, MAX, 0.)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, MAX, -0.) }, super::super::fmaf(NEG_INFINITY, MAX, -0.)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, MAX, F32_MIN_SUBNORM) }, super::super::fmaf(NEG_INFINITY, MAX, F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, MAX, -F32_MIN_SUBNORM) }, super::super::fmaf(NEG_INFINITY, MAX, -F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, MAX, MIN_POSITIVE) }, super::super::fmaf(NEG_INFINITY, MAX, MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, MAX, -MIN_POSITIVE) }, super::super::fmaf(NEG_INFINITY, MAX, -MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, MAX, MAX) }, super::super::fmaf(NEG_INFINITY, MAX, MAX)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, MAX, -MAX) }, super::super::fmaf(NEG_INFINITY, MAX, -MAX)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, F32_MIN_SUBNORM, 0.) }, super::super::fmaf(NEG_INFINITY, F32_MIN_SUBNORM, 0.)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, F32_MIN_SUBNORM, -0.) }, super::super::fmaf(NEG_INFINITY, F32_MIN_SUBNORM, -0.)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, F32_MIN_SUBNORM, F32_MIN_SUBNORM) }, super::super::fmaf(NEG_INFINITY, F32_MIN_SUBNORM, F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, F32_MIN_SUBNORM, -F32_MIN_SUBNORM) }, super::super::fmaf(NEG_INFINITY, F32_MIN_SUBNORM, -F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, F32_MIN_SUBNORM, MIN_POSITIVE) }, super::super::fmaf(NEG_INFINITY, F32_MIN_SUBNORM, MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, F32_MIN_SUBNORM, -MIN_POSITIVE) }, super::super::fmaf(NEG_INFINITY, F32_MIN_SUBNORM, -MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, F32_MIN_SUBNORM, MAX) }, super::super::fmaf(NEG_INFINITY, F32_MIN_SUBNORM, MAX)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, F32_MIN_SUBNORM, -MAX) }, super::super::fmaf(NEG_INFINITY, F32_MIN_SUBNORM, -MAX)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, MIN_POSITIVE, 0.) }, super::super::fmaf(NEG_INFINITY, MIN_POSITIVE, 0.)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, MIN_POSITIVE, -0.) }, super::super::fmaf(NEG_INFINITY, MIN_POSITIVE, -0.)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, MIN_POSITIVE, F32_MIN_SUBNORM) }, super::super::fmaf(NEG_INFINITY, MIN_POSITIVE, F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, MIN_POSITIVE, -F32_MIN_SUBNORM) }, super::super::fmaf(NEG_INFINITY, MIN_POSITIVE, -F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, MIN_POSITIVE, MIN_POSITIVE) }, super::super::fmaf(NEG_INFINITY, MIN_POSITIVE, MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, MIN_POSITIVE, -MIN_POSITIVE) }, super::super::fmaf(NEG_INFINITY, MIN_POSITIVE, -MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, MIN_POSITIVE, MAX) }, super::super::fmaf(NEG_INFINITY, MIN_POSITIVE, MAX)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, MIN_POSITIVE, -MAX) }, super::super::fmaf(NEG_INFINITY, MIN_POSITIVE, -MAX)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, -MAX, 0.) }, super::super::fmaf(NEG_INFINITY, -MAX, 0.)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, -MAX, -0.) }, super::super::fmaf(NEG_INFINITY, -MAX, -0.)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, -MAX, F32_MIN_SUBNORM) }, super::super::fmaf(NEG_INFINITY, -MAX, F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, -MAX, -F32_MIN_SUBNORM) }, super::super::fmaf(NEG_INFINITY, -MAX, -F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, -MAX, MIN_POSITIVE) }, super::super::fmaf(NEG_INFINITY, -MAX, MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, -MAX, -MIN_POSITIVE) }, super::super::fmaf(NEG_INFINITY, -MAX, -MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, -MAX, MAX) }, super::super::fmaf(NEG_INFINITY, -MAX, MAX)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, -MAX, -MAX) }, super::super::fmaf(NEG_INFINITY, -MAX, -MAX)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, -F32_MIN_SUBNORM, 0.) }, super::super::fmaf(NEG_INFINITY, -F32_MIN_SUBNORM, 0.)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, -F32_MIN_SUBNORM, -0.) }, super::super::fmaf(NEG_INFINITY, -F32_MIN_SUBNORM, -0.)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, -F32_MIN_SUBNORM, F32_MIN_SUBNORM) }, super::super::fmaf(NEG_INFINITY, -F32_MIN_SUBNORM, F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, -F32_MIN_SUBNORM, -F32_MIN_SUBNORM) }, super::super::fmaf(NEG_INFINITY, -F32_MIN_SUBNORM, -F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, -F32_MIN_SUBNORM, MIN_POSITIVE) }, super::super::fmaf(NEG_INFINITY, -F32_MIN_SUBNORM, MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, -F32_MIN_SUBNORM, -MIN_POSITIVE) }, super::super::fmaf(NEG_INFINITY, -F32_MIN_SUBNORM, -MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, -F32_MIN_SUBNORM, MAX) }, super::super::fmaf(NEG_INFINITY, -F32_MIN_SUBNORM, MAX)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, -F32_MIN_SUBNORM, -MAX) }, super::super::fmaf(NEG_INFINITY, -F32_MIN_SUBNORM, -MAX)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, -MIN_POSITIVE, 0.) }, super::super::fmaf(NEG_INFINITY, -MIN_POSITIVE, 0.)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, -MIN_POSITIVE, -0.) }, super::super::fmaf(NEG_INFINITY, -MIN_POSITIVE, -0.)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, -MIN_POSITIVE, F32_MIN_SUBNORM) }, super::super::fmaf(NEG_INFINITY, -MIN_POSITIVE, F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, -MIN_POSITIVE, -F32_MIN_SUBNORM) }, super::super::fmaf(NEG_INFINITY, -MIN_POSITIVE, -F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, -MIN_POSITIVE, MIN_POSITIVE) }, super::super::fmaf(NEG_INFINITY, -MIN_POSITIVE, MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, -MIN_POSITIVE, -MIN_POSITIVE) }, super::super::fmaf(NEG_INFINITY, -MIN_POSITIVE, -MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, -MIN_POSITIVE, MAX) }, super::super::fmaf(NEG_INFINITY, -MIN_POSITIVE, MAX)));
        // assert!(equal(unsafe { fmaf(NEG_INFINITY, -MIN_POSITIVE, -MAX) }, super::super::fmaf(NEG_INFINITY, -MIN_POSITIVE, -MAX)));
        // assert!(equal(unsafe { fmaf(MAX, INFINITY, 0.) }, super::super::fmaf(MAX, INFINITY, 0.)));
        // assert!(equal(unsafe { fmaf(MAX, INFINITY, -0.) }, super::super::fmaf(MAX, INFINITY, -0.)));
        // assert!(equal(unsafe { fmaf(MAX, INFINITY, F32_MIN_SUBNORM) }, super::super::fmaf(MAX, INFINITY, F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(MAX, INFINITY, -F32_MIN_SUBNORM) }, super::super::fmaf(MAX, INFINITY, -F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(MAX, INFINITY, MIN_POSITIVE) }, super::super::fmaf(MAX, INFINITY, MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(MAX, INFINITY, -MIN_POSITIVE) }, super::super::fmaf(MAX, INFINITY, -MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(MAX, INFINITY, MAX) }, super::super::fmaf(MAX, INFINITY, MAX)));
        // assert!(equal(unsafe { fmaf(MAX, INFINITY, -MAX) }, super::super::fmaf(MAX, INFINITY, -MAX)));
        // assert!(equal(unsafe { fmaf(F32_MIN_SUBNORM, INFINITY, 0.) }, super::super::fmaf(F32_MIN_SUBNORM, INFINITY, 0.)));
        // assert!(equal(unsafe { fmaf(F32_MIN_SUBNORM, INFINITY, -0.) }, super::super::fmaf(F32_MIN_SUBNORM, INFINITY, -0.)));
        // assert!(equal(unsafe { fmaf(F32_MIN_SUBNORM, INFINITY, F32_MIN_SUBNORM) }, super::super::fmaf(F32_MIN_SUBNORM, INFINITY, F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(F32_MIN_SUBNORM, INFINITY, -F32_MIN_SUBNORM) }, super::super::fmaf(F32_MIN_SUBNORM, INFINITY, -F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(F32_MIN_SUBNORM, INFINITY, MIN_POSITIVE) }, super::super::fmaf(F32_MIN_SUBNORM, INFINITY, MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(F32_MIN_SUBNORM, INFINITY, -MIN_POSITIVE) }, super::super::fmaf(F32_MIN_SUBNORM, INFINITY, -MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(F32_MIN_SUBNORM, INFINITY, MAX) }, super::super::fmaf(F32_MIN_SUBNORM, INFINITY, MAX)));
        // assert!(equal(unsafe { fmaf(F32_MIN_SUBNORM, INFINITY, -MAX) }, super::super::fmaf(F32_MIN_SUBNORM, INFINITY, -MAX)));
        // assert!(equal(unsafe { fmaf(MIN_POSITIVE, INFINITY, 0.) }, super::super::fmaf(MIN_POSITIVE, INFINITY, 0.)));
        // assert!(equal(unsafe { fmaf(MIN_POSITIVE, INFINITY, -0.) }, super::super::fmaf(MIN_POSITIVE, INFINITY, -0.)));
        // assert!(equal(unsafe { fmaf(MIN_POSITIVE, INFINITY, F32_MIN_SUBNORM) }, super::super::fmaf(MIN_POSITIVE, INFINITY, F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(MIN_POSITIVE, INFINITY, -F32_MIN_SUBNORM) }, super::super::fmaf(MIN_POSITIVE, INFINITY, -F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(MIN_POSITIVE, INFINITY, MIN_POSITIVE) }, super::super::fmaf(MIN_POSITIVE, INFINITY, MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(MIN_POSITIVE, INFINITY, -MIN_POSITIVE) }, super::super::fmaf(MIN_POSITIVE, INFINITY, -MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(MIN_POSITIVE, INFINITY, MAX) }, super::super::fmaf(MIN_POSITIVE, INFINITY, MAX)));
        // assert!(equal(unsafe { fmaf(MIN_POSITIVE, INFINITY, -MAX) }, super::super::fmaf(MIN_POSITIVE, INFINITY, -MAX)));
        // assert!(equal(unsafe { fmaf(-MAX, INFINITY, 0.) }, super::super::fmaf(-MAX, INFINITY, 0.)));
        // assert!(equal(unsafe { fmaf(-MAX, INFINITY, -0.) }, super::super::fmaf(-MAX, INFINITY, -0.)));
        // assert!(equal(unsafe { fmaf(-MAX, INFINITY, F32_MIN_SUBNORM) }, super::super::fmaf(-MAX, INFINITY, F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(-MAX, INFINITY, -F32_MIN_SUBNORM) }, super::super::fmaf(-MAX, INFINITY, -F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(-MAX, INFINITY, MIN_POSITIVE) }, super::super::fmaf(-MAX, INFINITY, MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(-MAX, INFINITY, -MIN_POSITIVE) }, super::super::fmaf(-MAX, INFINITY, -MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(-MAX, INFINITY, MAX) }, super::super::fmaf(-MAX, INFINITY, MAX)));
        // assert!(equal(unsafe { fmaf(-MAX, INFINITY, -MAX) }, super::super::fmaf(-MAX, INFINITY, -MAX)));
        // assert!(equal(unsafe { fmaf(-F32_MIN_SUBNORM, INFINITY, 0.) }, super::super::fmaf(-F32_MIN_SUBNORM, INFINITY, 0.)));
        // assert!(equal(unsafe { fmaf(-F32_MIN_SUBNORM, INFINITY, -0.) }, super::super::fmaf(-F32_MIN_SUBNORM, INFINITY, -0.)));
        // assert!(equal(unsafe { fmaf(-F32_MIN_SUBNORM, INFINITY, F32_MIN_SUBNORM) }, super::super::fmaf(-F32_MIN_SUBNORM, INFINITY, F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(-F32_MIN_SUBNORM, INFINITY, -F32_MIN_SUBNORM) }, super::super::fmaf(-F32_MIN_SUBNORM, INFINITY, -F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(-F32_MIN_SUBNORM, INFINITY, MIN_POSITIVE) }, super::super::fmaf(-F32_MIN_SUBNORM, INFINITY, MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(-F32_MIN_SUBNORM, INFINITY, -MIN_POSITIVE) }, super::super::fmaf(-F32_MIN_SUBNORM, INFINITY, -MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(-F32_MIN_SUBNORM, INFINITY, MAX) }, super::super::fmaf(-F32_MIN_SUBNORM, INFINITY, MAX)));
        // assert!(equal(unsafe { fmaf(-F32_MIN_SUBNORM, INFINITY, -MAX) }, super::super::fmaf(-F32_MIN_SUBNORM, INFINITY, -MAX)));
        // assert!(equal(unsafe { fmaf(-MIN_POSITIVE, INFINITY, 0.) }, super::super::fmaf(-MIN_POSITIVE, INFINITY, 0.)));
        // assert!(equal(unsafe { fmaf(-MIN_POSITIVE, INFINITY, -0.) }, super::super::fmaf(-MIN_POSITIVE, INFINITY, -0.)));
        // assert!(equal(unsafe { fmaf(-MIN_POSITIVE, INFINITY, F32_MIN_SUBNORM) }, super::super::fmaf(-MIN_POSITIVE, INFINITY, F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(-MIN_POSITIVE, INFINITY, -F32_MIN_SUBNORM) }, super::super::fmaf(-MIN_POSITIVE, INFINITY, -F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(-MIN_POSITIVE, INFINITY, MIN_POSITIVE) }, super::super::fmaf(-MIN_POSITIVE, INFINITY, MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(-MIN_POSITIVE, INFINITY, -MIN_POSITIVE) }, super::super::fmaf(-MIN_POSITIVE, INFINITY, -MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(-MIN_POSITIVE, INFINITY, MAX) }, super::super::fmaf(-MIN_POSITIVE, INFINITY, MAX)));
        // assert!(equal(unsafe { fmaf(-MIN_POSITIVE, INFINITY, -MAX) }, super::super::fmaf(-MIN_POSITIVE, INFINITY, -MAX)));
        // assert!(equal(unsafe { fmaf(MAX, NEG_INFINITY, 0.) }, super::super::fmaf(MAX, NEG_INFINITY, 0.)));
        // assert!(equal(unsafe { fmaf(MAX, NEG_INFINITY, -0.) }, super::super::fmaf(MAX, NEG_INFINITY, -0.)));
        // assert!(equal(unsafe { fmaf(MAX, NEG_INFINITY, F32_MIN_SUBNORM) }, super::super::fmaf(MAX, NEG_INFINITY, F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(MAX, NEG_INFINITY, -F32_MIN_SUBNORM) }, super::super::fmaf(MAX, NEG_INFINITY, -F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(MAX, NEG_INFINITY, MIN_POSITIVE) }, super::super::fmaf(MAX, NEG_INFINITY, MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(MAX, NEG_INFINITY, -MIN_POSITIVE) }, super::super::fmaf(MAX, NEG_INFINITY, -MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(MAX, NEG_INFINITY, MAX) }, super::super::fmaf(MAX, NEG_INFINITY, MAX)));
        // assert!(equal(unsafe { fmaf(MAX, NEG_INFINITY, -MAX) }, super::super::fmaf(MAX, NEG_INFINITY, -MAX)));
        // assert!(equal(unsafe { fmaf(F32_MIN_SUBNORM, NEG_INFINITY, 0.) }, super::super::fmaf(F32_MIN_SUBNORM, NEG_INFINITY, 0.)));
        // assert!(equal(unsafe { fmaf(F32_MIN_SUBNORM, NEG_INFINITY, -0.) }, super::super::fmaf(F32_MIN_SUBNORM, NEG_INFINITY, -0.)));
        // assert!(equal(unsafe { fmaf(F32_MIN_SUBNORM, NEG_INFINITY, F32_MIN_SUBNORM) }, super::super::fmaf(F32_MIN_SUBNORM, NEG_INFINITY, F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(F32_MIN_SUBNORM, NEG_INFINITY, -F32_MIN_SUBNORM) }, super::super::fmaf(F32_MIN_SUBNORM, NEG_INFINITY, -F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(F32_MIN_SUBNORM, NEG_INFINITY, MIN_POSITIVE) }, super::super::fmaf(F32_MIN_SUBNORM, NEG_INFINITY, MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(F32_MIN_SUBNORM, NEG_INFINITY, -MIN_POSITIVE) }, super::super::fmaf(F32_MIN_SUBNORM, NEG_INFINITY, -MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(F32_MIN_SUBNORM, NEG_INFINITY, MAX) }, super::super::fmaf(F32_MIN_SUBNORM, NEG_INFINITY, MAX)));
        // assert!(equal(unsafe { fmaf(F32_MIN_SUBNORM, NEG_INFINITY, -MAX) }, super::super::fmaf(F32_MIN_SUBNORM, NEG_INFINITY, -MAX)));
        // assert!(equal(unsafe { fmaf(MIN_POSITIVE, NEG_INFINITY, 0.) }, super::super::fmaf(MIN_POSITIVE, NEG_INFINITY, 0.)));
        // assert!(equal(unsafe { fmaf(MIN_POSITIVE, NEG_INFINITY, -0.) }, super::super::fmaf(MIN_POSITIVE, NEG_INFINITY, -0.)));
        // assert!(equal(unsafe { fmaf(MIN_POSITIVE, NEG_INFINITY, F32_MIN_SUBNORM) }, super::super::fmaf(MIN_POSITIVE, NEG_INFINITY, F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(MIN_POSITIVE, NEG_INFINITY, -F32_MIN_SUBNORM) }, super::super::fmaf(MIN_POSITIVE, NEG_INFINITY, -F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(MIN_POSITIVE, NEG_INFINITY, MIN_POSITIVE) }, super::super::fmaf(MIN_POSITIVE, NEG_INFINITY, MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(MIN_POSITIVE, NEG_INFINITY, -MIN_POSITIVE) }, super::super::fmaf(MIN_POSITIVE, NEG_INFINITY, -MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(MIN_POSITIVE, NEG_INFINITY, MAX) }, super::super::fmaf(MIN_POSITIVE, NEG_INFINITY, MAX)));
        // assert!(equal(unsafe { fmaf(MIN_POSITIVE, NEG_INFINITY, -MAX) }, super::super::fmaf(MIN_POSITIVE, NEG_INFINITY, -MAX)));
        // assert!(equal(unsafe { fmaf(-MAX, NEG_INFINITY, 0.) }, super::super::fmaf(-MAX, NEG_INFINITY, 0.)));
        // assert!(equal(unsafe { fmaf(-MAX, NEG_INFINITY, -0.) }, super::super::fmaf(-MAX, NEG_INFINITY, -0.)));
        // assert!(equal(unsafe { fmaf(-MAX, NEG_INFINITY, F32_MIN_SUBNORM) }, super::super::fmaf(-MAX, NEG_INFINITY, F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(-MAX, NEG_INFINITY, -F32_MIN_SUBNORM) }, super::super::fmaf(-MAX, NEG_INFINITY, -F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(-MAX, NEG_INFINITY, MIN_POSITIVE) }, super::super::fmaf(-MAX, NEG_INFINITY, MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(-MAX, NEG_INFINITY, -MIN_POSITIVE) }, super::super::fmaf(-MAX, NEG_INFINITY, -MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(-MAX, NEG_INFINITY, MAX) }, super::super::fmaf(-MAX, NEG_INFINITY, MAX)));
        // assert!(equal(unsafe { fmaf(-MAX, NEG_INFINITY, -MAX) }, super::super::fmaf(-MAX, NEG_INFINITY, -MAX)));
        // assert!(equal(unsafe { fmaf(-F32_MIN_SUBNORM, NEG_INFINITY, 0.) }, super::super::fmaf(-F32_MIN_SUBNORM, NEG_INFINITY, 0.)));
        // assert!(equal(unsafe { fmaf(-F32_MIN_SUBNORM, NEG_INFINITY, -0.) }, super::super::fmaf(-F32_MIN_SUBNORM, NEG_INFINITY, -0.)));
        // assert!(equal(unsafe { fmaf(-F32_MIN_SUBNORM, NEG_INFINITY, F32_MIN_SUBNORM) }, super::super::fmaf(-F32_MIN_SUBNORM, NEG_INFINITY, F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(-F32_MIN_SUBNORM, NEG_INFINITY, -F32_MIN_SUBNORM) }, super::super::fmaf(-F32_MIN_SUBNORM, NEG_INFINITY, -F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(-F32_MIN_SUBNORM, NEG_INFINITY, MIN_POSITIVE) }, super::super::fmaf(-F32_MIN_SUBNORM, NEG_INFINITY, MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(-F32_MIN_SUBNORM, NEG_INFINITY, -MIN_POSITIVE) }, super::super::fmaf(-F32_MIN_SUBNORM, NEG_INFINITY, -MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(-F32_MIN_SUBNORM, NEG_INFINITY, MAX) }, super::super::fmaf(-F32_MIN_SUBNORM, NEG_INFINITY, MAX)));
        // assert!(equal(unsafe { fmaf(-F32_MIN_SUBNORM, NEG_INFINITY, -MAX) }, super::super::fmaf(-F32_MIN_SUBNORM, NEG_INFINITY, -MAX)));
        // assert!(equal(unsafe { fmaf(-MIN_POSITIVE, NEG_INFINITY, 0.) }, super::super::fmaf(-MIN_POSITIVE, NEG_INFINITY, 0.)));
        // assert!(equal(unsafe { fmaf(-MIN_POSITIVE, NEG_INFINITY, -0.) }, super::super::fmaf(-MIN_POSITIVE, NEG_INFINITY, -0.)));
        // assert!(equal(unsafe { fmaf(-MIN_POSITIVE, NEG_INFINITY, F32_MIN_SUBNORM) }, super::super::fmaf(-MIN_POSITIVE, NEG_INFINITY, F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(-MIN_POSITIVE, NEG_INFINITY, -F32_MIN_SUBNORM) }, super::super::fmaf(-MIN_POSITIVE, NEG_INFINITY, -F32_MIN_SUBNORM)));
        // assert!(equal(unsafe { fmaf(-MIN_POSITIVE, NEG_INFINITY, MIN_POSITIVE) }, super::super::fmaf(-MIN_POSITIVE, NEG_INFINITY, MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(-MIN_POSITIVE, NEG_INFINITY, -MIN_POSITIVE) }, super::super::fmaf(-MIN_POSITIVE, NEG_INFINITY, -MIN_POSITIVE)));
        // assert!(equal(unsafe { fmaf(-MIN_POSITIVE, NEG_INFINITY, MAX) }, super::super::fmaf(-MIN_POSITIVE, NEG_INFINITY, MAX)));
        // assert!(equal(unsafe { fmaf(-MIN_POSITIVE, NEG_INFINITY, -MAX) }, super::super::fmaf(-MIN_POSITIVE, NEG_INFINITY, -MAX)));
    }
}
