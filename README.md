# Data-Driven Decision Support for Sustainable Urban Mobility Planning

This project compares bus service on routes 10, 12, and 46 in Astana using recorded GTFS trips and OpenStreetMap data. The analysis evaluates walking access to stops, operating speed, and travel-time variability across 18 route-direction profiles covering three weekday departure windows.

The Analytic Hierarchy Process (AHP) sets criterion weights, and TOPSIS ranks the profiles within each window. Sensitivity checks compare alternative weights, simple additive weighting, walking-distance thresholds, and resampled service dates. The specialist judgments and dashboard usability data are simulated.

## Repository contents

| Path | Contents |
| --- | --- |
| [`scripts/`](scripts/) | Trip reconstruction, spatial analysis, AHP–TOPSIS ranking, sensitivity analysis, usability simulation, and figure generation |
| [`raw_data/`](raw_data/) | GTFS inputs and saved OpenStreetMap extracts |
| [`results/`](results/) | Trip audit, indicators, rankings, sensitivity checks, and simulated usability results |
| [`figures/`](figures/) | Figures in PDF and PNG format |
| [`docs/Assignment_4_Overleaf.zip`](docs/Assignment_4_Overleaf.zip) | Assignment 4 LaTeX manuscript, bibliography, and figures |

## Reproducing the analysis

The scripts take the project directory as their argument and read or write working files in `out/`. To regenerate the figures from the archived results, run the following commands in PowerShell from the repository root.

```powershell
python -m pip install numpy pandas scipy matplotlib
Copy-Item -Path results -Destination out -Recurse
python scripts/s5_figures.py .
python scripts/s6_breakdown.py .
```

To rebuild the analysis from the inputs, place the contents of `raw_data/gtfs_data/gtfs_data/` in `data/zenodo/gtfs/gtfs_data/` and run scripts 1 through 6 in order. The spatial stage also requires GeoPandas, NetworkX, Shapely, and OSMnx. It uses the Overpass API, so a new extraction can produce different results from the saved October 2026 analysis. The optional map-export script expects the archived walking-network and building JSON files in `data/osm_cache/`, with filenames beginning with `3e0ea461` and `b4c2619f`, respectively.

## Data sources

- Astana bus operations data from Mansurova et al., recorded from July to September 2024. [Zenodo dataset](https://doi.org/10.5281/zenodo.15769359) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Spatial data extracted on 22 October 2026. © OpenStreetMap contributors. [Attribution and license information](https://www.openstreetmap.org/copyright) · [Open Database License](https://opendatacommons.org/licenses/odbl/1-0/)

## Authors

Danial Amankos, Ilyas Kapanov, and Sanzhar Saparbaev

Research Methods and Tools, Astana IT University
