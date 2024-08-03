# Approach

This project has been split into very modular packages and is organized as a
cargo workspace. A rough description of the structure of the repository is as follows:

- `common/`: The shared utilities used by all the experiment implementations in the repo.
    - `src/lib.rs`:
    - `src/interface.rs`: The [trait][Traits] associated with each component of
    the experiment.
    - `src/system.rs`: The [trait][Traits] that describes the system itself.
    Contains [associated constants][AssociatedConstants] that describe the
    dimensionality of our system's parametrizations.
    - `src/coordinator.rs`:
    - `src/rope.rs`:
    - `src/vector.rs`:
    - `src/python.rs`:
    - `examples/jax.rs`:
- `dummy_system/`:
    - `src/lib.rs`:
    - `examples/dummy.rs`:
- `sho/`:
    - `src/lib.rs`:
    - `src/driver.rs`:
    - `src/generator.rs`:
    - `src/simulator.rs`:
    - `src/state_estimator.rs`:
    - `src/system.rs`:
    - `src/sho_agent.py`:
    - `src/sho_state_predictor.py`:
    - `src/test_sho_agent.py`:


[Traits]: https://doc.rust-lang.org/reference/items/traits.html
[AssociatedConstants]: https://doc.rust-lang.org/reference/items/associated-items.html#associated-constants
