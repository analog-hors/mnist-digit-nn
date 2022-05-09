use std::ops::*;
use std::fmt::Debug;

use serde::{Serialize, Deserialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct FloatVec(Vec<f32>);

impl Debug for FloatVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl From<Vec<f32>> for FloatVec {
    fn from(vec: Vec<f32>) -> Self {
        Self(vec)
    }
}

impl Index<usize> for FloatVec {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for FloatVec {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl FloatVec {
    pub fn all(n: f32, len: usize) -> Self {
        Self(vec![n; len])
    }

    pub fn map(&self, f: impl FnMut(f32) -> f32) -> Self {
        Self(self.iter().map(f).collect())
    }

    pub fn exp(&self) -> Self {
        self.map(|n| n.exp())
    }

    pub fn dot(&self, other: &Self) -> f32 {
        (self * other).iter().sum()
    }

    pub fn iter(&self) -> impl Iterator<Item=f32> + '_ {
        self.0.iter().copied()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item=&mut f32> + '_ {
        self.0.iter_mut()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }
}

impl Neg for &FloatVec {
    type Output = FloatVec;

    fn neg(self) -> Self::Output {
        self.map(|n| -n)
    }
}

impl Neg for FloatVec {
    type Output = FloatVec;

    fn neg(self) -> Self::Output {
        -(&self)
    }
}

fn assert_compatible(a: &FloatVec, b: &FloatVec) {
    assert_eq!(a.len(), b.len(), "Cannot operate on vectors of size {} and {}", a.len(), b.len());
}

macro_rules! impl_ops {
    ($($trait:ident::$fn:ident,$assign_trait:ident::$assign_fn:ident;)*) => {$(
        // FloatVec -= &FloatVec
        impl $assign_trait<&FloatVec> for FloatVec {
            fn $assign_fn(&mut self, other: &FloatVec) {
                assert_compatible(self, other);
                for (dest, src) in self.iter_mut().zip(other.iter()) {
                    $assign_trait::$assign_fn(dest, src);
                }
            }
        }

        // FloatVec -= FloatVec
        impl $assign_trait for FloatVec {
            fn $assign_fn(&mut self, other: FloatVec) {
                $assign_trait::$assign_fn(self, &other);
            }
        }

        // FloatVec - &FloatVec
        impl $trait<&FloatVec> for FloatVec {
            type Output = FloatVec;

            fn $fn(mut self, other: &Self) -> Self::Output {
                $assign_trait::$assign_fn(&mut self, other);
                self
            }
        }

        // &FloatVec - FloatVec
        impl $trait<FloatVec> for &FloatVec {
            type Output = FloatVec;

            fn $fn(self, mut other: FloatVec) -> Self::Output {
                assert_compatible(self, &other);
                for (l, r) in self.iter().zip(other.iter_mut()) {
                    *r = $trait::$fn(l, *r);
                }
                other
            }
        }

        // &FloatVec - &FloatVec
        impl $trait<&FloatVec> for &FloatVec {
            type Output = FloatVec;

            fn $fn(self, other: &FloatVec) -> Self::Output {
                $trait::$fn(self.clone(), other)
            }
        }

        // FloatVec - FloatVec
        impl $trait for FloatVec {
            type Output = Self;

            fn $fn(self, other: Self) -> Self::Output {
                $trait::$fn(self, &other)
            }
        }

        // FloatVec -= f32
        impl $assign_trait<f32> for FloatVec {
            fn $assign_fn(&mut self, other: f32) {
                for dest in self.iter_mut() {
                    $assign_trait::$assign_fn(dest, other);
                }
            }
        }

        // FloatVec - f32
        impl $trait<f32> for FloatVec {
            type Output = FloatVec;

            fn $fn(mut self, other: f32) -> Self::Output {
                $assign_trait::$assign_fn(&mut self, other);
                self
            }
        }

        // f32 - FloatVec
        impl $trait<FloatVec> for f32 {
            type Output = FloatVec;

            fn $fn(self, mut other: FloatVec) -> Self::Output {
                for other in other.iter_mut() {
                    *other = $trait::$fn(self, *other);
                }
                other
            }
        }

        // &FloatVec - f32
        impl $trait<f32> for &FloatVec {
            type Output = FloatVec;

            fn $fn(self, other: f32) -> Self::Output {
                $trait::$fn(self.clone(), other)
            }
        }

        // f32 - &FloatVec
        impl $trait<&FloatVec> for f32 {
            type Output = FloatVec;

            fn $fn(self, other: &FloatVec) -> Self::Output {
                $trait::$fn(self, other.clone())
            }
        }
    )*}
}

impl_ops! {
    Add::add, AddAssign::add_assign;
    Sub::sub, SubAssign::sub_assign;
    Mul::mul, MulAssign::mul_assign;
    Div::div, DivAssign::div_assign;
}
