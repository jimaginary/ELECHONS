# ELEC Honours Thesis Work
This is a repo for my electrical engineering honours thesis at UNSW.
The subject matter is graph signal processing.

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

### Tips
- Work on two things at once

### Data modifications
- Acorn data station 048027 min has some -999degC entries which are clearly invalid. These have been removed.
- Removed data files from stations without latitutde and longtitude data in the metadata set
- Backfilled, then forwardfilled data in filled_* folders in datasets for ease of use.

### Results
- As in Moura (2014), got some compression results selecting highest magnitude spectra components
- For the graph product spectra GFT kron DFT
- A kron B v can be decomposed as applying B to chunks of v, then A to differently selected chunks of the output
- In our case, we FFT each time series, then for each frequency GFT the graph data, giving an output 104x17838 size spectra
- Below we have coefficient fraction vs RMSE for each compressed stat, with worse but similar results to Moura
- Note that the results are identical due to Moura (2014) for graph Cartesian, Strong, and Kronecker products
max data compression RMSE
1/50,	1/20,	1/15,	1/10,	1/7,	1/5,	1/3,	1/2,	1/1,	
8.86,	7.33,	6.85,	6.15,	5.49,	4.82,	3.64,	2.51,	0.00,	
min data compression RMSE
1/50,	1/20,	1/15,	1/10,	1/7,	1/5,	1/3,	1/2,	1/1,	
15.23,	13.02,	12.28,	11.16,	10.06,	8.90,	6.79,	4.72,	0.00,	
mean data compression RMSE
1/50,	1/20,	1/15,	1/10,	1/7,	1/5,	1/3,	1/2,	1/1,	
8.54,	6.95,	6.47,	5.80,	5.18,	4.54,	3.43,	2.36,	0.00,
