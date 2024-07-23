use crate::{
    rope::{Rope, RopeMut}, Float
};

/// Represents a non-contiguous collection of data.
pub trait DynamicVector<S: Float> {
    /// Create `Self` by copying data from a slice of data.
    fn copy_from_slice(&mut self, v: &[S]);

    /// Get a non-contiguous rope.
    fn get_rope(&self) -> Rope<S>;
    /// Get a non-contiguous mutable rope.
    fn get_rope_mut(&mut self) -> RopeMut<S>;
}

impl<S: Float> DynamicVector<S> for Vec<S> {
    fn copy_from_slice(&mut self, v: &[S]) {
        <[S]>::copy_from_slice(self, v);
    }

    fn get_rope(&self) -> Rope<S> {
        Rope::new(&[&self[..]])
    }

    fn get_rope_mut(&mut self) -> RopeMut<S> {
        RopeMut::new([&mut self[..]])
    }
}

impl<S: Float, const DIMS: usize> DynamicVector<S> for [S; DIMS] {
    fn copy_from_slice(&mut self, v: &[S]) {
        <[S]>::copy_from_slice(&mut *self, v);
    }

    fn get_rope(&self) -> crate::rope::Rope<S> {
        Rope::new(&[self])
    }

    fn get_rope_mut(&mut self) -> crate::rope::RopeMut<S> {
        RopeMut::new([self])
    }
}

impl<S: Float> DynamicVector<S> for [S] {
    fn copy_from_slice(&mut self, v: &[S]) {
        <[S]>::copy_from_slice(&mut *self, v);
    }

    fn get_rope(&self) -> crate::rope::Rope<S> {
        Rope::new(&[self])
    }

    fn get_rope_mut(&mut self) -> crate::rope::RopeMut<S> {
        RopeMut::new([self])
    }
}

impl<S: Float> DynamicVector<S> for S {
    fn copy_from_slice(&mut self, v: &[S]) {
        std::slice::from_mut(self).copy_from_slice(v)
    }

    fn get_rope(&self) -> crate::rope::Rope<S> {
        Rope::new(&[std::slice::from_ref(self)])
    }

    fn get_rope_mut(&mut self) -> crate::rope::RopeMut<S> {
        RopeMut::new([std::slice::from_mut(self)])
    }
}

impl<S: Float> DynamicVector<S> for () {
    fn copy_from_slice(&mut self, v: &[S]) {
        [].copy_from_slice(v)
    }

    fn get_rope(&self) -> Rope<S> {
        Rope::new(&[&[]])
    }

    fn get_rope_mut(&mut self) -> RopeMut<S> {
        RopeMut::new([&mut []])
    }
}

/// The description of a physical system.
pub trait System<S: Float> {
    /// The number of parameters in the [`System::ControlSignal`] type.
    const CONTROL_SIGNAL_SIZE: usize;
    /// The number of parameters in the [`System::ControlParams`] type.
    const CONTROL_PARAMS_SIZE: usize;
    /// The number of parameters in the [`System::LatentState`] type.
    const LATENT_STATE_SIZE: usize;
    /// The number of parameters in the [`System::SystemState`] type.
    const SYSTEM_STATE_SIZE: usize;
    /// The number of parameters in the [`System::SystemObservation`] type.
    const OBSERVABLE_STATE_SIZE: usize;

    /// The configuration of the system.
    type SystemConfiguration;
    /// The configuration of the dynamics.
    type DynamicsConfiguration;

    /// The full state of the system
    type SystemState: DynamicVector<S>;
    /// The latent state of the system
    type LatentState: DynamicVector<S>;
    /// The control parameters of the system
    type ControlParams: DynamicVector<S>;
    /// The control signal of the system
    type ControlSignal: DynamicVector<S>;
    /// The observation of the system
    type SystemObservation: DynamicVector<S>;
}
