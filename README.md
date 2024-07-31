# Build Guide

Build with the following line of `bash` code:
```bash
cargo build --release
```

# Inference Flow

```mermaid
graph LR
    RL([RL Agent])
    G([Generator])
    SE([State Estimator])
    S([Simulation])

    S -- Observations --> SE
    SE -- Current State Estimate --> RL
    RL -- Control Parameters --> G
    G -- Control Signal --> S

    classDef edgeLabel background:#1b1b2b;
    classDef edgeLabel color:#ababbb;
    classDef edgeLabel outline-width: 1px;
    classDef edgeLabel outline-color:#555555;
    classDef edgeLabel outline-style: solid;
    classDef edgeLabel padding: 5pt;
```

# Message Types

### Hyperparameters

```rust
/// A tuple of dimensions in the latent embedding of the system state.
const LATENT_SPACE_SHAPE: (usize,);

/// The number of previous observations and controls to use to Markovianize the
/// process with the state estimator.
const DELAY_DEPTH: usize;
```

### System Dependent

```rust
/// The number of observable components of the state.
const OBSERVABLE_SHAPE: (usize,);

/// The number of components of the control parameters.
const CONTROL_PARAMETER_SHAPE: (usize,);

/// The number of components of the control signal.
const CONTROL_SIGNAL_SHAPE: (usize,);
```

```rust
type ObservableState <== Tensor<T, OBSERVABLE_SHAPE>;
type ControlParameterState <== Tensor<T, CONTROL_PARAMETER_SHAPE>;
type ControlSignalState <== Tensor<T, CONTROL_SIGNAL_SHAPE>;
type StateTensor <== Tensor<T, LATENT_SPACE_SHAPE>;

struct Observation {
    state: ObservableState,
    controls: ControlSignalState,
    time_observed: f64,
}

struct Observations([Observation; DELAY_DEPTH]);
struct CurrentStateEstimate(StateTensor);
```

# Training Flow

### RL Agent

Note that here, $H_n$ refers to the observations from the state estimator and
$A_n$ refers to the actions produced by the reinforcement learning agent.

```mermaid
graph LR
    S{{System +
    State Estimator}}
    RL{RL Agent}
    SE(($$H_t$$))
    NSE(("$$H_{t + 1}$$"))
    A(($$A_t$$))
    NA(("$$A_{t + 1}$$"))
    L(("$$\mathcal{L}$$"))

    S .-> SE
    SE --> RL --> A
    A --> S --> NSE
    NSE --> RL --> NA
    SE & NSE & A & NA --> L
    NA .-> S
    L .->|"$$\partial \mathcal{L}$$"| RL

    classDef edgeLabel background:#1b1b2b;
```

### State Estimator

Note that here, $S_n$ refers to the observations from the simulations, $A_n$
refers to the actions produced by the generator, and $T_n$ refers to the
encoded time.

```mermaid
graph LR
    S{{System}}
    SE{State Estimator}
    ISE{Inverse \n State Estimator}
    NO(($$S_t$$))
    NT(($$T_t$$))
    PNO(("$$\tilde{S}_t$$"))
    PT(("$$T_{t-1,t-2,\dots}$$"))
    PO(("$$S_{t-1,t-2,\dots}$$"))
    PC(("$$A_{t-1,t-2,\dots}$$"))
    CSE(($$\mu_t$$))
    CSU(($$\sigma_t$$))
    L(("$$\mathcal{L}$$"))

    S --> PO & PC & PT & NO & NT
    PO & PC & PT & NT --> SE --> CSE & CSU --> ISE --> PNO
    NO & PNO & CSU --> L
    L .-> SE & ISE

    classDef edgeLabel background:#1b1b2b;
```

### Next Steps
---

- Documentation
- Implement (& Test) the driver, state-estimator
- Test the Execution Loop
- Test the Simulator
- Integration Tests
