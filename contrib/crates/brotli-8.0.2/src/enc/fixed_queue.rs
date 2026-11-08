pub const MAX_THREADS: usize = 16;

pub struct FixedQueue<T: Sized> {
    data: [Option<T>; MAX_THREADS],
    size: usize,
    start: usize,
}
impl<T: Sized> Default for FixedQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}
impl<T: Sized> FixedQueue<T> {
    pub fn new() -> Self {
        FixedQueue {
            data: [
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None,
            ],
            size: 0,
            start: 0,
        }
    }
    pub fn can_push(&self) -> bool {
        self.size < self.data.len()
    }
    pub fn size(&self) -> usize {
        self.size
    }
    pub fn push(&mut self, item: T) -> Result<(), ()> {
        if self.size == self.data.len() {
            return Err(());
        }
        let index = (self.start + self.size) % self.data.len();
        self.data[index] = Some(item);
        self.size += 1;
        Ok(())
    }
    pub fn pop(&mut self) -> Option<T> {
        if self.size == 0 {
            return None;
        }
        let index = self.start % self.data.len();
        let ret = self.data[index].take();
        self.start += 1;
        self.size -= 1;
        ret
    }
    pub fn how_much_free_space(&self) -> usize {
        self.data.len() - self.size
    }
    pub fn remove<F: Fn(&Option<T>) -> bool>(&mut self, f: F) -> Option<T> {
        if self.size == 0 {
            return None;
        }
        for index in 0..self.size {
            if f(&self.data[(self.start + index) % self.data.len()]) {
                let start_index = self.start % self.data.len();
                let target_index = (self.start + index) % self.data.len();
                let ret = self.data[target_index].take();
                let replace = self.data[start_index].take();
                let is_none = core::mem::replace(&mut self.data[target_index], replace);
                assert!(is_none.is_none());
                self.start += 1;
                self.size -= 1;
                return ret;
            }
        }
        None
    }
}
