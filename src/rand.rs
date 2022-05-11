pub struct Rng(u128);

impl Default for Rng {
    fn default() -> Self {
        Self(0xB57D6A35CFD25BDBD774231501440A94 | 1)
    }
}

impl Rng {
    pub fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(0x2360ED051FC65DA44385DF649FCCF645);
        let rot = (self.0 >> 122) as u32;
        let xsl = ((self.0 >> 64) as u64) ^ (self.0 as u64);
        let next = xsl.rotate_right(rot);
        next as u32 as f64 / (u32::MAX as u64 + 1) as f64
    }
}
