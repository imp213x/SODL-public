
# Step 22 Pass Test

## Build
```powershell
cargo test
```

## What should pass
- `sodl-semantic-color` tests
- `sodl-semantic-cube` tests

## Acceptance
If all tests pass:
- color semantic code packing/unpacking works
- hierarchy affects semantic distance
- cube lattice nearest-neighbor ordering works

These modules are then ready to be consumed by Carla-side experiments.
