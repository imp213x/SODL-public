
# SODL Semantic Research Notes

This note connects the semantic modules to the earlier CPU/GPU optimization research.

## Findings carried forward
The earlier research suggested that practical system wins often come from:
- reducing data movement
- increasing locality
- replacing large float representations with compact codes
- precomputing reusable artifacts instead of rebuilding
- clustering semantically related data for better access patterns

## Step 22 interpretation
### Color-spectrum module
- functions like a compact semantic palette
- supports cheap distance checks
- makes hierarchical semantic routing explicit

### Ruby-cube lattice module
- functions like an interpretable semantic manifold
- supports low-cost neighborhood lookup
- preserves multiple semantic axes in a structured, human-readable way

## Why SODL is the right engine
SODL ensures:
- raw bytes stay immutable
- semantic experiments remain reproducible derivations
- different representations can coexist
- Carla can compare them without losing provenance
