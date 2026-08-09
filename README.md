# PyOctal-PostProcessing

Post-processing for photonic simulation and experimental measurement data —
parsing raw instrument/simulation output, spectrum analysis, curve fitting,
and plotting.

The repository is split into two parts:
- **`postprocessing/`** — the core library. Reusable parsers, analysis
  classes, fitting models, and formatting helpers with no hardcoded paths.
- **`tools/`** — standalone run scripts, one per instrument or experiment
  type, that import from `postprocessing/` and point it at real data. These
  are meant to be copied/edited per-experiment (folder paths, voltages,
  filenames are hardcoded at the top of each `main()`).

## `postprocessing/` — core library

| Module | Purpose |
| --- | --- |
| `analysis.py` | `PAnalysis` — resonance analysis of a single transmission spectrum: peak-finding, resonance wavelength, FSR, linewidth/FWHM, Q-factor. |
| `fitting.py` | Curve-fitting routines for ring resonator and ring-assisted MZI (Ramzi) transmission models. |
| `fom.py` | Figure-of-merit calculations between two biasing conditions of a modulator: extinction ratio (ER), optical modulation amplitude (OMA), transmitted power (TP). |
| `modulator.py` | Device-level modulator calculations: modulation efficiency, capacitance, and loss from measured phase/resonance shift. |
| `photonics.py` | Underlying ring resonator / MZI physics formulas (transmission, phase response) shared by `fitting.py` and simulation tools. |
| `zernike.py` | Zernike polynomial fitting/reconstruction of scattered (x, y, z) surface data — used for wafer bow/warp analysis. |
| `parser.py` | File parsers for instrument/simulation output: Keysight OMR/PAS (`omr_parse`), ADS csv (`ads_parse`), Cadence Virtuoso csv (`vcsv_parse`), Touchstone s-parameters (`snp_parse`), MATLAB `.mat`/`.h5` (`matlab_parse`), Wyko profilometer `.ASC` (`wyko_asc_parse`), KLA stylus profiles (`kla_stylus_parse`), and wafer point-map `.txt` files (`wafer_txt_parse`). |
| `utils/conversion.py` | dB ↔ linear unit conversions. |
| `utils/formatter.py` | `Publication` — matplotlib styling for publication-quality figures, plus shared plot helpers (`add_wafer_circles`, `twin_x`, `set_titles`). |
| `utils/op.py` | Signal processing helpers: rolling-window averaging, S-parameter normalisation, Savitzky-Golay/Butterworth filtering. |

## `tools/` — run scripts

### `tools/exp/` — experimental data processing

| Script | Purpose |
| --- | --- |
| `bandwidth.py` | Electro-optic bandwidth (3dB point) from LCA S21 spectra across bias voltages. |
| `bruker.py` | Wafer bow/warp: parses a Wyko `.ASC` height map and fits/plots its Zernike decomposition. |
| `cmp.py` | CMP polish-rate and wafer point-map contour plots (thickness removed, rate, over/underlayer selectivity). |
| `cmp_pos.py` | Converts a reticle shot-map layout into stage (x, y) coordinates for CMP sampling. |
| `completeease.py` | Exports ellipsometer measurement data to an Excel workbook. |
| `coupling_model.py` | Measures ring cross/self-coupling power ratio from a pair of OMR transmission files. |
| `fpp.py` | Four-point-probe wafer sheet-resistance contour plot (121-point layout). |
| `heater.py` | Thermal heater characterisation: optical power vs. applied voltage/electrical power. |
| `kla_stylus.py` | KLA stylus profilometer dishing/step-height profiles across a wafer. |
| `rta.py` | Plots RTA furnace historical process logs and recipe steps. |
| `spectrum/transmission.py` | Overlays raw transmission spectra across bias voltages. |
| `spectrum/ring_fitting.py` | Fits the ring resonator model to a measured spectrum to extract coupling coefficients. |
| `spectrum/fom.py` | Modulation efficiency and phase shift vs. voltage from resonance-shift tracking. |
| `spectrum/er_oma_tp.py` | ER/OMA/TP spectra across bias voltages for finding the optimal operating wavelength. |

### `tools/sim/` — simulation data processing

| Script | Purpose |
| --- | --- |
| `ads.py` | Effective capacitance/resistance vs. frequency from a Keysight ADS export. |
| `mzi.py` | MZI insertion loss vs. arm length/voltage from simulated `neff`/loss data. |
| `n_alpha.py` | Free-carrier dispersion (Δn, Δα) vs. doping concentration. |
| `photonics/coupling.py` | Ring cross/self-coupling coefficient vs. coupling radius. |
| `plot_fom.py` | Extinction ratio / insertion loss figure-of-merit plot with 3dB/6dB markers. |
| `plot_mzi.py` | MZI power/transmission spectra across ring and heater bias points. |
| `plot_spectrum.py` | Simple transmission spectrum power plot across bias voltages. |

## `tests/`

Unit tests for `postprocessing/`. Run with:
```
python -m unittest discover -s tests
```

## Usage

Run any tool script as a module (edit the hardcoded paths/parameters at the
top of its `main()` first):
```
python -m tools.exp.heater
```
