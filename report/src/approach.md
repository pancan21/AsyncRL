# Approach

This project has been split into very modular packages and is organized as a
cargo workspace. A rough description of the structure of the repository is as follows:

- `common/`: The shared utilities used by all the experiment implementations in the repo.
    - `src/lib.rs`: Defines the `Float` [trait][Traits] and the `FloatType`
    [enum][Enums] and the submodules within the crate.
    - `src/interface.rs`: The [trait][Traits] associated with each component of
    the experiment.
    - `src/system.rs`: The [trait][Traits] that describes the system itself.
    Contains [associated constants][AssociatedConstants] that describe the
    dimensionality of our system's parametrizations.
    - `src/coordinator.rs`: The asynchronous function that defines the
    evolution of an "experiment", i.e. one run of with a given implementation
    of [`DriverInterface`][DriverInterface],
    [`GeneratorInterface`][GeneratorInterface],
    [`SimulatorInterface`][SimulatorInterface], and
    [`StatePredictionInterface`][StatePredictionInterface] for some
    implementors of the `System` and `Float` [traits][Traits].
    - `src/rope.rs`: Defines a collection of multiple [slices][Slices] with the
    same [lifetime][Lifetimes] called `Rope` and `RopeMut`, where the former is
    for "immutable" or "shared" references and the latter is for "mutable" or
    "exclusive" references. These are "disjoint" from each other in some sense,
    for more info, see the [Rustnomicon's section on
    aliasing][MutableAliasing]. These are used to make it easier to avoid
    copying data.
    - `src/vector.rs`: Defines a simple constant-dimension vector type that
    just wraps the [array][Arrays] type. Has a lot of useful functions that
    make it easy to work and do arithmetic with.
    - `src/python.rs`: Provides a lot of useful Python utilities to make
    working with Rye virtual environments easier and working with JAX Arrays as
    [`Futures`][Futures] and JAX PRNG Key utilities.
    - `examples/jax.rs`: An example of using the Python utilities in this crate.
- `dummy_system/`:
    - `src/lib.rs`: Defines an implementation of each necessary component:
    driver, generator, simulator, and state predictor, for the trivial system
    of just incrementing the time, with delays added to each component to
    exaggerate the behavior of the asynchronicity.
    - `examples/dummy.rs`: The demonstration that runs the "experiment" with
    these dummy components.
- `sho/`:
    - `src/lib.rs`: Defines the submodules in the crate.
    - `src/system.rs`: Defines the Simple Harmonic Oscillator system in 2
    dimensions.
    - `src/driver.rs`: Implements the [`DriverInterface`][DriverInterface] by
    hooking into `sho/src/sho_agent.py`.
    - `src/generator.rs`: Implements the
    [`GeneratorInterface`][GeneratorInterface].
    - `src/simulator.rs`: Implements the
    [`SimulatorInterface`][SimulatorInterface]
    - `src/state_estimator.rs`: Implements the
    [`StatePredictionInterface`][StatePredictionInterface] by hooking into
    `sho/src/sho_state_predictor.py`.
    - `src/sho_agent.py`: Uses JAX to implement the learning for the
    reinforcement learning agent.
    - `src/sho_state_predictor.py`: Uses JAX to implement the learning for the
    state estimation agent.
    - `examples/sho.rs`: The demonstration that runs the "experiment" with the
    components defined in this crate.

[Arrays]: https://doc.rust-lang.org/reference/types/array.html
[AssociatedConstants]: https://doc.rust-lang.org/reference/items/associated-items.html#associated-constants
[Enums]: https://doc.rust-lang.org/reference/items/enumerations.html
[Futures]: https://rust-lang.github.io/async-book/02_execution/02_future.html
[Lifetimes]: https://doc.rust-lang.org/nomicon/lifetimes.html
[MutableAliasing]: https://doc.rust-lang.org/nomicon/aliasing.html
[Slices]: https://doc.rust-lang.org/reference/types/slice.html
[Traits]: https://doc.rust-lang.org/reference/items/traits.html

[DriverInterface]: introduction.md#DriverInterface
[GeneratorInterface]: introduction.md#GeneratorInterface
[SimulatorInterface]: introduction.md#SimulatorInterface
[StatePredictionInterface]: introduction.md#StatePredictionInterface
