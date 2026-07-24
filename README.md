# pycopdem

**Cached [Copernicus GLO-30](https://registry.opendata.aws/copernicus-dem/)
elevation with on-read terrain derivatives — download once per chunk,
never twice.** Every elevation pixel this machine ever downloads lands
in one sparse, chunk-indexed store on the DEM's native 1-arc-second
grid; slope, aspect, flow accumulation, TWI and Heat Load Index are
computed on read and never stored. Part of the
[Borevitz Lab](https://borevitzlab.anu.edu.au/) ecosystem.

## How it works

```
{data_root}/copdem_store/
├── index.db      # SQLite ledger: which chunks are populated
└── dem.zarr      # global sparse array; only written 1200×1200-px chunks exist
```

- The GLO-30 DEM is served as one COG per 1° × 1° cell on a 1-arc-second
  EPSG:4326 lattice. The store uses the same lattice globally, chunked
  at 1200 px (1/3°) — so **3 × 3 chunks nest exactly inside every
  tile**, and fetching a chunk is a single integer-aligned windowed
  read from one COG. No resampling, ever.
- Any bbox maps deterministically to a set of chunk ids.
  `Store.get_ds(bbox)` diffs them against the ledger and downloads
  **only the missing chunks**. Elevation is time-invariant, so there's
  no time axis and no date bookkeeping.
- A 1° tile absent from S3 is genuinely all ocean: its chunks are
  stored as nodata and marked complete.
- Derivatives are a read-time transform (like everything derived in
  this ecosystem): request them per call, pay compute not disk.

## Usage

The core API is **query-agnostic** — just a bbox:

```python
from pycopdem.store import Store

store = Store()
bbox = [148.36265, -33.52606, 148.38265, -33.50606]  # [W, S, E, N]

ds = store.get_ds(bbox)                              # elevation (lat, lon)
ds = store.get_ds(bbox, derivatives=('slope', 'aspect', 'twi', 'hli'))

store.fill(bbox)                                     # → 0: already local
```

Derivatives: `slope` and `aspect` (degrees), `accumulation` (pysheds
fill-pits → fill-depressions → resolve-flats → flowdir → accumulation),
`twi` (ln(accumulation / tan slope)), `hli` (McCune & Keon 2002).
Dependencies resolve automatically — asking for `twi` computes slope
and accumulation internally.

Pipelines that speak the shared `borevitz_lab.query.Query` use the
adapters (dates on the query are ignored — elevation doesn't change):

```python
ds = store.get_ds_query(query, derivatives=('slope',))
```

`download_terrain(query)` remains as a thin wrapper.

## Performance

Live measurements against the Copernicus S3 bucket — a ~2 × 2 km AOI
(one *chunk* = 1200 × 1200 px ≈ 37 × 30 km):

| Scenario | Downloaded | Time |
|---|---|---|
| Cold fill | 1 chunk | 10.3 s |
| Same request again | nothing | **0.0 s** |
| AOI shifted ~28 km east | 1 new chunk | 4.2 s |
| Read cached window (1200²) | — | 0.4 s |
| Read + all 5 derivatives | — | 3.2 s (pysheds dominates) |

Store footprint: ~10 MB for two chunks (~2 200 km² of elevation).
Absolute times vary with network; the zero is the point — it's a
ledger lookup, no network involved.

## Install

### Conda (recommended)

```bash
conda install -c conda-forge -c thestochasticman pycopdem
```

### From source

All lab repos share one conda environment, **`borevitz_lab`** — each
repo's `environment.yml` creates it if missing and adds its own
packages if it exists (never use `--prune`):

```bash
conda env update -n borevitz_lab -f environment.yml
conda activate borevitz_lab
pip install -e ../borevitz_lab   # shared core (not yet on PyPI)
pip install -e .
```

Package design (shared across the lab's packages — no inheritance,
composition only):

- **`Query`** (from `borevitz-lab`) — identity: what region.
- **`CopernicusDEM`** (`pycopdem.copdem`) — config: tile source.
- **`Paths`** (`pycopdem.paths`) — derived locations of the store for a
  given `Config`.
- **`grid`** — the fixed 1-arc-second grid and tile/chunk nesting
  (pure, offline-testable math).
- **`derive`** — the terrain derivatives (pure array math + pysheds).
- **`Store`** (`pycopdem.store`) — ties them together.

## Test

```bash
# offline (pure math + synthetic store):
python pycopdem/grid.py     # True
python pycopdem/paths.py    # True
python pycopdem/derive.py   # True
python pycopdem/store.py    # True

# live (small real reads from the Copernicus S3 bucket):
python pycopdem/download_terrain.py  # True
```
