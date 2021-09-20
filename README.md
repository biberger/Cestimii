# Cestimii
## Multiscale Curvature Estimation using Integral Invariants.

This Python library was made as part of the Master's thesis "Integral Invariants for Multiscale Curvature Estimations on Cell Membranes" by Simon Biberger. It contains the implementation of the paper "Principal curvatures from the integral invariant viewpoint" by Helmut Pottmann, Johannes Wallner, Yong-Liang Yang, Yu-Kun Lai, and Shi-Min Huc from 2007 ([doi](https://doi.org/10.1016/j.cagd.2007.07.004)). Additionally, their idea of splitting occupancy grids was formalised in the thesis and two algorithms for occupancy grid splitting were proposed: 
* a simple algorithm, which produces a highly structured set of surface-covering cubes fast, 
* a sweep plane algorithm that constructs a set of cubes, which adapt organically to the local geometry.

The step to go from a set of surface-covering cubes to a set of occupancy grids can be non-trivial for complex shapes. Thus, we proposed a extension of the framework by Pottmann et al., which uses a more relaxed notion of occupancy grids. This allows the use of a dataset's probability or intensity values for multiscale curvature estimation. 
As choosing an appropriate scale can be difficult, Cestimii contains a method to efficiently average results over multiple scales at once by reprocessing results from lower scales.

## Installation
Currently, only the code itself is contained in here. In the upcoming days, a proper installation guide and a Python package will follow.

## Dependencies
Cestimii depends on the following libraries
* for calculations: [NumPy](https://github.com/numpy/numpy), [SciPy](https://github.com/scipy/scipy),
* for visualisations (optional): [matplotlib](https://github.com/matplotlib/matplotlib), [napari](https://github.com/napari/napari),
* for the processing of image stacks (optional): PIL ([Pillow](https://github.com/python-pillow/Pillow)), [h5py](https://github.com/h5py/h5py).

## FAQ
TBD.

## Images
The following two images show the mean curvature estimations using the regular and multiple scales at once curvature estimation framework with relaxed occupancy grids for scales r=15 and r=18,20,22,...,30. The images show two separate parts of a segmented cell membrane dataset provided by Diz-Muñoz, EMBL Heidelberg. 

![Mean Curvatures for r=15](https://user-images.githubusercontent.com/89973708/132211418-41dbc97c-1138-48b1-b18b-8991d5ff8c87.png)
![Mean Curvatures for r=18,20,22,...,30](https://user-images.githubusercontent.com/89973708/132211425-ad8a623a-36ee-4e30-b62a-f91297d96fdf.png)

## Acknowledgements
I would like to express my deepest appreciation to Brigitte Forster-Heinlein from the University of Passau and Virginie Uhlmann from EMBL-EBI for their continuous support, invaluable advice and excellent feedback. 
This project was supported by a fellowship within the IFI programme of the German Academic Exchange Service.
