# Build Guide

First, install [Rye](https://rye.astral.sh/). Then synchronize the dependencies with the `bash` command:

```bash
rye sync
```

Now, activate the virtual environment (variations for different shells are also
available, refer to Rye documentation):
```bash
source .venv/bin/activate
```

Build with the following line of `bash` code:
```bash
cargo build --release
```

If the Python libraries cannot be found, try setting the appropriate flags in
the `.cargo/config.toml` file. To do this, try adding the `build.rustflags`
field. The following is based on the default installation for Rye.
```toml
[build]
rustflags = ["-L", "$HOME/.rye/py/cpython@3.12.3/lib/"]
```
 
__Note, you will have to manually set `$HOME` to the correct directory__, i.e.
```toml
[build]
rustflags = ["-L", "/home/my_username/.rye/py/cpython@3.12.3/lib/"]
```
