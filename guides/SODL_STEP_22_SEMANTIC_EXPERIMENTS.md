
# SODL Step 22 — Semantic Experiments (Color Spectrum + Ruby Cube Lattice)

This step adds two experimental, model-agnostic semantic modules to SODL:

- `sodl-semantic-color`
- `sodl-semantic-cube`

These are **representation experiments**, not replacements for the source dataset. SODL remains the engine:
origins stay immutable, derivations stay lineage-tracked, and AI systems can choose whether to use these compact artifacts for retrieval/training pre-processing.

## Why these two modules
The earlier CPU/GPU research review highlighted that meaningful speedups often come from reducing:
- bytes moved
- redundant rebuilds
- memory bandwidth pressure
- full-float semantic representations

It also pointed toward:
- compact coding / dictionary-style encodings
- semantic clustering
- quantized representations
- locality-friendly layouts

These two modules are the first controlled experiments in that direction.

## Design choices

### Color module
We use **32 bits total**:
- 8-bit hierarchy
- 24-bit RGB

Why not 8-bit total?
- 8-bit total only gives 256 codes, which is too small for meaningful semantic separation.
- Full RGB gives 16.7M local positions.
- The extra hierarchy byte gives coarse semantic routing.

Why not store Hue too?
- Hue is derivable from RGB.
- Storing it would be redundant and increase ambiguity instead of helping hierarchy.

So the chosen representation is:

`H-R-G-B`

Where:
- `H` = semantic family / hierarchy bucket
- `RGB` = local semantic coordinate

### Ruby cube module
We use a **3D lattice** with an optional `layer` byte.

Why lattice instead of free vectors?
- Lattices are interpretable.
- Distances are cheap.
- They support semantic neighborhoods naturally.
- They are easy to pin into lineage as deterministic artifacts.

Default intended axes:
- X = topic
- Y = intent / action
- Z = emotion / context

These axes are not fixed by code; they are part of the artifact manifest.

## How to use
- Raw dataset remains an `Origin`
- Semantic representations become derived artifacts
- Carla or any future AI can consume:
  - raw dataset
  - embeddings
  - color-coded semantic artifact
  - cube-lattice semantic artifact

All remain traceable through SODL lineage.

## What this does *not* solve yet
- no chunked CAS
- no compression of the raw blob itself
- no replacement of embeddings
- no AGI claims

This is strictly a reproducible experiment layer on top of SODL.
