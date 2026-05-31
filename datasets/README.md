# datasets

Where the benchmark data lives. Large raw datasets are **not** fully committed
(GitHub size limits); only small sample files are included so the pipeline can
be smoke-tested. Download the full datasets from the sources below.

## Layout

| Folder | Committed sample | Full dataset source |
|--------|------------------|---------------------|
| `AirQuality/` | `pm25_ground.txt`, `pm25_missing.txt` (Beijing PM2.5) | [UCI: Beijing Multi-Site Air Quality](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data) |
| `PSM/` | `test.csv`, `test_label.csv` (Pooled Server Metrics) | eBay PSM (server metrics) |
| `PhysioNet/` | `link.txt` (download pointer only) | [PhysioNet Challenge 2012](https://physionet.org/content/challenge-2012/1.0.0/) |

The FMGAN study additionally uses Weather, Electricity, and ETT, fetched on the
fly via the PyPOTS / TSDB ecosystem (`tsdb`, `benchpots`); see
[`FMGAN/`](../FMGAN/).

## Expected format

Models read per-dataset arrays of shape `(timesteps, features)`. The FMGAN
loaders expect a prepared `data.npz` per dataset (generated locally, gitignored);
the thesis models read the raw `.txt` / `.csv` directly. Place any downloaded
files in the matching subfolder.

> Datasets are redistributed here under their original licenses/terms; consult
> each source before redistribution.
