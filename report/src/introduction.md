# Introduction

The problem at hand is to model the behavior of a model-free reinforcement
learning algorithm that is aiming to replicate some (observable) target
dynamics, but with only a limited interface to the model's behavior, i.e. the
model is both partially-observable and partially-controllable.

The main constraint we add on top of this is incorporating the asynchronicity
of computation. We know that computing the output of our agent happens in
non-zero time, so we modify our computation loop to be asynchronous and have
our model compute parameters for a synchronous signal generator.

In this model, we have 3 main components and define interfaces between them:

- Reinforcement Learning Agent (Driver)
- Signal Generator
- System (Simulator)

Using Rust trait syntax (for more information on that, check the
[reference](https://doc.rust-lang.org/reference/items/traits.html)),
we can write out the behaviour of each component.

<a id="DriverInterface"></a>
```rust
{{#include ../../common/src/interfaces.rs:DriverInterface}}
```

<a id="GeneratorInterface"></a>
```rust
{{#include ../../common/src/interfaces.rs:GeneratorInterface}}
```

<a id="SimulatorInterface"></a>
```rust
{{#include ../../common/src/interfaces.rs:SimulatorInterface}}
```

Now, it's rather interesting that we have a `S::LatentState` as the input to
our [`DriverInterface`](#DriverInterface), why is that? That's because there is
another component that has not been mentioned yet: the State Predictor:

<a id="StatePredictionInterface"></a>
```rust
{{#include ../../common/src/interfaces.rs:StatePredictionInterface}}
```

> [!Note]
> For a series of partial observations, which can have a rather high
> dimensionality, we use the [State Predictor](#StatePredictionInterface) to
> compress the partial observations into a latent vector and the uncertainty in
> the aforementioned latent vector.
