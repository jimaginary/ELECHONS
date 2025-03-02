# ELEC Honours Thesis Work
This is a repo for my electrical engineering honours thesis at UNSW.
The subject matter is graph signal processing.

## Todo
- Review IEEE Signal Processing Magazine articles
- Implement kronecker time-series/spatial graph product GFT and replicate compression results

### Long run investigation ideas
- Stochastic signals on graphs
- Methods for constructing sparse graphs that give good results (dimensionality reduction)

### Tips
- Work on two things at once

### Data modifications
- Acorn data station 048027 min has some -999degC entries which are clearly invalid. These have been removed.
- Removed data files from stations without latitutde and longtitude data in the metadata set
- Backfilled, then forwardfilled data in filled_* folders in datasets for ease of use.
