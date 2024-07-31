#![forbid(
    missing_docs,
    clippy::missing_assert_message,
    clippy::missing_docs_in_private_items,
    clippy::missing_asserts_for_indexing
)]
#![allow(unused)]
//! Defines the Simple Harmonic Oscillator model and the relevant default and dummy
//! implementations.

/// Implements the standard Python [`DriverInterface`] driver for the
/// [`SimpleHarmonicOscillator`](crate::system::SimpleHarmonicOscillator) alongside a dummy agent.
pub mod driver;

/// Implements the standard Python [`StatePredictionInterface`] driver for the
/// [`SimpleHarmonicOscillator`](crate::system::SimpleHarmonicOscillator) alongside a dummy agent.
pub mod state_estimator;

/// This module defines the [`SignalGenerator`] for the
/// [`SimpleHarmonicOscillator`](crate::system::SimpleHarmonicOscillator) as well as a
/// [`DummySignalGenerator`].
pub mod generator;

/// Defines the time evolution for our system.
pub mod simulator;

/// Contains the system definition and relevant types for a simple coupled harmonic oscillator
/// system. Defines the
/// [`SimpleHarmonicOscillator<T: Scalar>`](crate::system::SimpleHarmonicOscillator) implementation
/// of [`System<T: Scalar>`](common::system::System).
pub mod system;
