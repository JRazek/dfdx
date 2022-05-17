pub trait CountElements {
    const NUM_ELEMENTS: usize;
    type Element: Clone + Default;
    const NUM_BYTES: usize = Self::NUM_ELEMENTS * std::mem::size_of::<Self::Element>();
}

impl CountElements for f32 {
    const NUM_ELEMENTS: usize = 1;
    type Element = Self;
}

impl<T: CountElements, const M: usize> CountElements for [T; M] {
    const NUM_ELEMENTS: usize = M * T::NUM_ELEMENTS;
    type Element = T::Element;
}

pub trait HasInner {
    type Inner;
}

impl<const M: usize> HasInner for [f32; M] {
    type Inner = Self;
}

impl<T: HasInner, const M: usize> HasInner for [T; M] {
    type Inner = T::Inner;
}

pub trait ZeroElements {
    const ZEROS: Self;
}

impl ZeroElements for f32 {
    const ZEROS: Self = 0.0;
}

impl<T: ZeroElements, const M: usize> ZeroElements for [T; M] {
    const ZEROS: Self = [T::ZEROS; M];
}

pub trait IsNdArray {
    type Array: 'static + Sized + Clone + CountElements + ZeroElements;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_0d_count() {
        assert_eq!(1, f32::NUM_ELEMENTS);
    }

    #[test]
    fn test_1d_count() {
        assert_eq!(5, <[f32; 5]>::NUM_ELEMENTS);
    }

    #[test]
    fn test_2d_count() {
        assert_eq!(15, <[[f32; 3]; 5]>::NUM_ELEMENTS);
    }

    #[test]
    fn test_3d_count() {
        assert_eq!(30, <[[[f32; 2]; 3]; 5]>::NUM_ELEMENTS);
    }
}
