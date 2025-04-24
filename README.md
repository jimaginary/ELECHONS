# ELEC Honours Thesis Work
This is a repo for my electrical engineering honours thesis at UNSW.
The subject matter is graph signal processing.

## Reproduce Figs
To reproduce the figures in the Thesis A report:
1. Clone this repository
2. Navigate to ELECHONS/code
```
$ python
>>> import regress_temp as r
>>> r.init('mean')
>>> r.plot_all()
>>> exit()
```
3. Plots will be located in ELECHONS/plts.
4. The above can also be done for 'max' and 'min' temperature statistics.

## Todo
- Review IEEE Signal Processing Magazine articles
- Implement kronecker time-series/spatial graph product GFT and replicate compression results
- Note this ^ splits FT = FT1 kron FT2 w/ eigvals dependent on graph product
- w/ FT1 dim n, FT2 dim m, this can be decomposed into nT(FT2)+mT(FT1)
- For a GFT and an FFT respectively we get O(nmlogm+mn^2)
- Need to FFT each time series, then GFT each graph series.

### Long run investigation ideas
- Stochastic signals on graphs
- Methods for constructing sparse graphs that give good results (dimensionality reduction)

### Working on convolution of screened poisson problem green's functions
- see https://physics.stackexchange.com/questions/581052/yukawa-potential-in-higher-dimensions for general green's function.

### Data modifications
- Acorn data station 048027 min has some -999degC entries which are clearly invalid. These have been removed.
- Removed data files from stations without latitutde and longtitude data in the metadata set
- Backfilled, then forwardfilled data in filled_* folders in datasets for ease of use.
- Realised some entries weren't just empty but the dates themselves weren't in the dataset, so redid backfilling to fill missing dates

### Results
- As in Moura (2014), got some compression results selecting highest magnitude spectra components
- For the graph product spectra GFT kron DFT
- A kron B v can be decomposed as applying B to chunks of v, then A to differently selected chunks of the output
- In our case, we FFT each time series, then for each frequency GFT the graph data, giving an output 104x17838 size spectra
- Below we have coefficient fraction vs RMSE for each compressed stat, with worse but similar results to Moura
- Note that the results are identical due to Moura (2014) for graph Cartesian, Strong, and Kronecker products
