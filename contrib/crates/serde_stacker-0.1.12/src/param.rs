#[derive(Copy, Clone)]
pub struct Param {
    pub red_zone: usize,
    pub stack_size: usize,
}

impl Default for Param {
    fn default() -> Self {
        Param {
            red_zone: 64 * 1024,
            stack_size: 2 * 1024 * 1024,
        }
    }
}

impl Param {
    pub fn new(red_zone: usize, stack_size: usize) -> Self {
        Param {
            red_zone,
            stack_size,
        }
    }
}
