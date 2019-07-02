use std::vec;

#[inline]
// #[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn powi(x: f64, exp: usize) -> f64 {
  let mut powers: Vec<f64>;
  powers.push(1.);
  powers.push(x);

  if exp == 0 { return 1. }
  let mut i = 1;
  while i < exp / 2 {
    if powers[2 * i] <= 0. {
      powers[2 * i] = powers[i] * powers[i];
    }
    i += 1;
  }
  if exp <= i {
    return powers[i]
  } else {
    0.
  }
}

#[cfg(test)]
mod tests {

  #[test]
  pub fn test_powi() {
    assert_eq!(super::powi(2.0, 20), (1 << 20) as f64);
    assert_eq!(super::powi(-1.0, 9), -1.0);
    assert!(super::powi(-1.0, 2).is_nan());
    assert!(super::powi(-1.0, 1).is_nan());
  }
}
