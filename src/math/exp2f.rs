/* wf_exp2.c -- float version of w_exp2.c.
 * Conversion to float by Ian Lance Taylor, Cygnus Support, ian@cygnus.com.
 */

/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

use super::powf;

#[inline]
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn exp2f(x: f32) -> f32 {
    powf(2f32, x)
}
