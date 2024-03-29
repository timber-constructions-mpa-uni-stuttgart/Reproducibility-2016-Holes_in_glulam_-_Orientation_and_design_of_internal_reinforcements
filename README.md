#### Reproducibility repository for:

Holes in glulam - Orientation and design of internal reinforcements
===================================================================

#### Authors:
- Tapia Camú, Cristóbal (```cristobal.tapia-camu@mpa.uni-stuttgart.de```)
- Aicher, Simon (```simon.aicher@mpa.uni-stuttgart.de```)

Publication Date: August 2016

## Content of this repository:
This is the reproducibility repository for the paper "Holes in glulam - Orientation and design of internal reinforcements".
It contains some code and data used to produce the results shown in the paper, as well as minor corrections of the text.
The files contained in this repository are listed below:

- [Paper (PDF)](WCTE_2016_Tapia_Aicher_final.pdf)
- coefficients_fit
    - ```obtain_coefficients.py```
    - data
        - ```Ft_90_hd135_w_100_lA2_lad200_reinf18.dat```
        - ```...```
- examples
    - ```Design_Example_Paper.ipynb```

### Requirements:
To use the code listed above, the following tools must be installed:
- python >= 3.3
    - numpy
    - sympy
    - jupyter
    - seaborn
    - matplotlib
    - scipy

### Abstract:
The usage of holes in glulam and LVL beams is a common practice in timber constructions and requires in many cases the application of reinforcement.
At present, Eurocode 5 does not contain design rules for holes, nor for their reinforcement, which are, however, regulated in the German National Annex to EC5.
Although it has been proven that internal rod-like reinforcements improve the shear force capacity of a beam with holes, several problems still remain, particularly the inability to successfully reduce peak stresses at the periphery of the hole, especially shear stresses.
Inclined internal steel rod reinforcements were studied and compared with vertically oriented rods, which is currently the only regulated application.
The analysis revealed a reduction of both perpendicular to grain tensile stresses and shear stresses, which for the case of vertical rods are not reduced at all.
A first attempt at the design of such inclined reinforcements was made by deriving an equation based on the results from FEM simulations.
The design approach was then applied to an example case.
Experimental verification of the theoretical observations is still necessary and ongoing, though a very promising approach for an improved internal reinforcement and its respective design can already be observed.

### Corrections:
1. There were some small notation errors in the original publication, specifically in the equations (5) and (6), where the factor _**w**_ (width of the cross section) was forgotten.
Nevertheless, for the calculations this factor was indeed used.
The equation should look like this:

    ![equation](images/equation_5.png)  
    
    ![equation](images/equation_6.png)  

2. When calibrating the coefficients _**c1**_, _**c2**_ and _**c3**_ of Equation (4) the double of the shear force (and thus also the double of the moment) was erroneously considered.
The Equation (4) is still valid, only the coefficients needed to be recalculated:

    | c1 | c2    | c3   |
    |:--:|:-----:|:----:|
    | 1.2| 0.012 | 0.36 |

    The example presented in Section 5.1 can be calculated using the jupyter [notebook](examples/Design_Example_Paper.ipynb) present in the [example](examples/README.md) folder

3. The figure 9c and 3b show the arrows representing the shear forces in the wrong direction.

### Citation:
Bibtex citation:
```
@InProceedings{Tapia2016a,
  author    = {Tapia, C. and Aicher, S.},
  title     = {Holes in glulam -- orientation and design of internal reinforcements},
  booktitle = {CD-ROM Proceedings of the World Conference on Timber Engineering (WCTE 2016)},
  year      = {2016},
  editor    = {J. Eberhardsteiner and W. Winter and A. Fadai and M. Pöll},
  address   = {Vienna, Austria},
  month     = {August 22-25},
  publisher = {Vienna University of Technology, Austria},
  file      = {pdf file:Tagungen/2016_WCTE/Contribution1003_a_final.pdf:PDF},
  groups    = {WCTE_2016},
  isbn      = {978-3-903039-00-1},
}
```
