# PyOctal-PostProcessing

Post-processing simulation and measurement data of photonic data. It features functions such as parsers, fitting, spectrum analysis, and plotting data.

## Directory Structure
```
.
<folder>
├── postprocessing           # core library
├── tools                    # standalone processing files are under here.
    ├── exp                     # experimental data processing
    ├── sim                     # simulation data processing
<files>
├── requirements.txt         # contain all required python packages for this repository
```

To run the files, run them as a module:
```python
# example
python -m tools.exp.heater
```
