#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sy
import seaborn.apionly as sns
from scipy.optimize import minimize

# Create color palette
col_pal = sns.color_palette("Paired", 8)

def main():
    """main function"""
    # call the function to obtain the coefficients
    obtain_coefficients()

def obtain_coefficients():
    """
    Obtain the coefficients 'c1', 'c2' and 'c3' for the Equation (4).

    :returns: a dictionary with the named coefficients

    """
    ######################################################################
    # Read the data obtained from the FE computations:
    #
    # This data contains the axial and tangential forces in each
    # reinforcement obtained at the vertical position where the crack is
    # defined (for this study it was constant: 45°)
    ######################################################################
    # Define geometric properties
    height = 450. # mm
    # Define the different configurations that will be read:
    conf_1 = {'lad':200., 'diam':18., 'width':120., 'height':height, 'hd':height*0.3}
    conf_2 = {'lad':200., 'diam':12., 'width':120., 'height':height, 'hd':height*0.3}
    conf_3 = {'lad':200., 'diam':18., 'width':100., 'height':height, 'hd':height*0.3}
    # all in one list
    list_configs = [conf_1, conf_2, conf_3]
    # import files
    for config in list_configs:
        # Data for the reinforced case
        file_path = (
                'data/Ft_90_hd135_w_{w:1.0f}_lA2_'
                'lad{lad:1.0f}_reinf{d_reinf:1.0f}.dat'
                ).format(
                        w=config['width'], lad=config['lad'],
                        d_reinf=config['diam']
                        )
        # Data for the unreinforced case
        file_path_unreinforced = 'data/Ft_90_hd135_lA2_no_reinf.dat'
        # read the file with pandas
        data_aux = pd.read_csv(file_path, index_col=0)
        data_unr_aux = pd.read_csv(file_path_unreinforced, index_col=0)
        # add the data to the configuration's dictionary
        config['data'] = data_aux
        config['data_unreinforced'] = data_unr_aux
        # Also calculate inertia of the cross-section of each configuration
        config['I'] = config['height']**3 * config['width'] / 12.

    ######################################################################
    # Feed the data to an optimze function to get the requiered
    # coefficients c1, c2 and c3
    ######################################################################
    shear_force = 50e3 # N (given in the FE model)
    moment_1 = -shear_force * 900.
    moment_2 = -shear_force * (900. + 135)
    #
    res_coeffs = minimize(find_coefficients, x0=np.array([0.5,0.01,0.3]),
                          args=(list_configs, shear_force, moment_1, moment_2),
                          method='SLSQP')
    c1, c2, c3 = res_coeffs.x
    print('+----------------------------------')
    print('Coefficients found:')
    print('c1 = {c1:2.3f}; \nc2 = {c2:2.3f} \nc3 = {c3:2.3f}'.format(
            c1=c1, c2=c2, c3=c3) )
    print('+----------------------------------')

    ######################################################################
    # Generate plot to compare FE calculations and equation results
    ######################################################################
    # Create figure and axis instance
    fig = plt.figure(figsize=(9.8,3.7))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, sharey=ax1)
    # loop for every data set
    for ix, config in enumerate(list_configs):
        # get the angles of inclination (stored as indices of the
        # DataFrame)
        angles = config['data'].index.values
        ##################################################
        # retrieve glulam properties
        h = config['height']
        I = config['I']
        w = config['width']
        lad = config['lad']
        hd = config['hd']
        diam = config['diam']
        # Shear stresses
        F_rV = calc_F_rV(shear=shear_force, height=h, width=w, I=I, angle=angles,
                              lad=lad, hd=hd)
        # Bending stresses
        F_rM_1 = calc_F_rM(moment=moment_1, height=h, width=w, I=I, hd=hd,
                             positive_bending=True)
        F_rM_2 = calc_F_rM(moment=moment_2, height=h, width=w, I=I, hd=hd,
                             positive_bending=False)
        # Get the tensile vertical forces
        F_t90_1 = config['data_unreinforced']['Ft90_1'].values
        F_t90_2 = config['data_unreinforced']['Ft90_2'].values
        # Get the axial force in the rod
        F_rod_1 = config['data']['F_norm_1']
        F_rod_2 = config['data']['F_norm_2']
        ##################################################
        # Calculate the axial forces in the rod
        ##################################################
        R_n_1 = calc_Rn(F_V=F_rV, F_M=F_rM_1, Ft90=F_t90_1,
                        d=diam, w=w, beta=angles, c1=c1, c2=c2, c3=c3)
        R_n_2 = calc_Rn(F_V=F_rV, F_M=F_rM_2, Ft90=F_t90_2,
                        d=diam, w=w, beta=angles, c1=c1, c2=c2, c3=c3)
        # Plot the FE results 
        ax1.plot(F_rod_1*1e-3, color=col_pal[2*(ix)], ls='none', marker='o')
        ax2.plot(F_rod_2*1e-3, color=col_pal[2*(ix)], ls='none', marker='o')
        # Plot the results from the formula
        ax1.plot(angles, R_n_1*1e-3, color=col_pal[2*(ix)+1])
        ax2.plot(angles, R_n_2*1e-3, color=col_pal[2*(ix)+1])

    ax1.set_ylim(ymin=0)
    ax1.set_ylabel('Axial force $F_r$ in the rod [kN]')
    ax1.set_xlabel('rod inclination [degree]')
    ax2.set_xlabel('rod inclination [degree]')
    # Add some annotations
    ax1.annotate('Left rod', xy=(0.98,0.02), xycoords='axes fraction',
                 va='bottom', ha='right')
    ax2.annotate('Right rod', xy=(0.98,0.02), xycoords='axes fraction',
                 va='bottom', ha='right')
    ax1.grid(True)
    ax2.grid(True)
    fig.tight_layout()
    plt.show()

    return 1

def find_coefficients(coeffs, list_configurations, shear_force, moment1, moment2):
    """
    Calculate the axial force in the reinforcements using 'calc_Rn()' (Eq. (4)), and compare
    the results against the FE computations to obtain the difference between them.

    :coeffs: numpy array of size = 3 with the coefficients c1, c2 and c3
    :list_configurations: list with the studied configurations
    :moment1: moment at the left edge of the hole
    :moment2: moment at the right edge of the hole
    :shear_force: shear force at the requiered edge of the hole

    :returns: double. A metric to determine how good or bad the results are in comparisson with
        the FE solutions

    """
    c1, c2, c3 = coeffs
    #V = sy.symbols('V') # shear force
    #y = sy.symbols('y') # vertical position
    V = shear_force
    M_1 = moment1
    M_2 = moment2

    # Initialize sum of squares
    sum_sq = 0.

    for config in list_configurations:
        # The angles are retrieved from the data (which correspond to
        # the index of the DataFrame)
        angles = config['data'].index.values
        # retrieve glulam properties
        h = config['height']
        I = config['I']
        w = config['width']
        lad = config['lad']
        hd = config['hd']
        diam = config['diam']
        # Shear stresses
        F_rV_aux = calc_F_rV(shear=V, height=h, width=w, I=I, angle=angles,
                              lad=lad, hd=hd)
        # Bending stresses
        F_rM_aux = calc_F_rM(moment=M_1, height=h, width=w, I=I, hd=hd, positive_bending=True)
        F_rM_aux2 = calc_F_rM(moment=M_2, height=h, width=w, I=I, hd=hd, positive_bending=False)
        # Get the tensile vertical forces
        F_t90_aux = config['data_unreinforced']['Ft90_1'].values
        F_t90_aux2 = config['data_unreinforced']['Ft90_2'].values
        ##################################################
        # Calculate the axial forces in the rod
        ##################################################
        R_n_1 = calc_Rn(F_V=F_rV_aux, F_M=F_rM_aux, Ft90=F_t90_aux,
                        d=diam, w=w, beta=angles, c1=c1, c2=c2, c3=c3)
        R_n_2 = calc_Rn(F_V=F_rV_aux, F_M=F_rM_aux2, Ft90=F_t90_aux2,
                        d=diam, w=w, beta=angles, c1=c1, c2=c2, c3=c3)
        # compare against the FE results
        diff_aux = (R_n_1 - config['data']['F_norm_1'])**2
        diff_aux2 = (R_n_2 - config['data']['F_norm_2'])**2
        #
        sum_sq += np.sum(diff_aux) + np.sum(diff_aux2)

    return sum_sq

def calc_F_rV(shear, height, width, I, hd, angle, lad):
    """
    Calculate the contribution of the shear force F_rV, according to Eq. (5)

    :shear: shear force
    :height: height (depth) of the beam
    :width: width of the cross-section
    :I: moment of inertia
    :hd: diameter of the hole (if circular)
    :angle: angle of inlination of the rod
    :lad: anchorage length (distance from the crack plane to the end of the reinforcement,
        along the reinforcement)

    :returns: float

    """
    # - lower integration limit
    y1 = hd * 0.5 * 0.7 - lad * np.cos(np.radians(angle))
    # - upper integration limit
    y2 = hd * 0.5 * 0.7
    # Nummerical integration
    F_rV = (-0.125 * shear * height**2 * width * y1 / I
           + 0.125 * shear * height**2 * width * y2 / I
           + shear * width * y1**3 / (I * 6.)
           - shear * width * y2**3 / (I * 6.) )
    # Symbolic integration
    #y = sy.Symbol('y')
    #shear_stress = shear / (2. * I) * (height**2 / 4. - y**2)
    #F_rV = sy.integrate(shear_stress * width, (y, y1, 0.15*height*0.7))
    #V_r_aux_2 = sy.integrate(shear_stress * w, (y, y_0, 0.15*h*0.7))

    return F_rV

def calc_F_rM(moment, height, width, I, hd, positive_bending=True):
    """
    Calculate the contribution of the moment to the axial force of the reinforcement, acc. to
    Eq. (6)

    :moment: moment
    :height: height (depth) of the beam
    :width: width of the cross-section
    :I: moment of inertia
    :hd: diameter of the hole (if circular)
    :positive_bending: bool. Defines whether the rod is inserted in a bending-tensile zone
        (True) or in a bending-compressive zone (False). This will modify the integration
        limits accordingly.

    :returns: float

    """
    if positive_bending:
        # - lower integration limit
        y1 = - height /2.
        # - upper integration limit
        y2 = - hd * 0.5 * 0.7
    else:
        # - lower integration limit
        y1 = hd * 0.5 * 0.7
        # - upper integration limit
        y2 = height / 2.

    # Nummerical integration
    F_rM = (moment * width) / (2. * I) * (y2**2 - y1**2)
    # Symbolic integration
    #y = sy.Symbol('y')
    #bending_1 = -moment * y / I
    #bending_2 = -M_2 * y / I
    #F_rM = sy.integrate(bending_1 * width, (y, -height/2., -0.15*height*0.7))
    #M_r_aux_2 = sy.integrate(bending_2 * w, (y, 0.15*h*0.7, h/2.))

    return F_rM

def calc_Rn(F_V, F_M, Ft90, c1, c2, c3, beta, d, w):
    """
    Calculate the axial force in the reinforcement

    :F_V: portion of the force associated with the shear force
    :F_M: portion of the force associated to the moment
    :Ft90: vertical tensile force acc. to German National Annex to EN 1995-1-1
    :beta: inclination of the reinforcement (in degrees). beta = 0° --> vertical position
    :c1: coefficient c1
    :c2: coefficient c2
    :c3: coefficient c3
    :d: diameter of the reinforcement
    :w: width of the glulam beam

    :returns: maximal axial force of the reinforcement acc. to Eq. (4)

    """
    # Calculate the sinus and cosinus
    cos_B = np.cos( np.radians(beta) )
    sin_B = np.sin( np.radians(beta) )
    # Apply equation (4)
    R_n = ( (F_V * c1 + F_M * c2 ) * sin_B * np.sqrt(d/w) + Ft90 * c3 * d/np.sqrt(w) ) / cos_B

    return R_n

if __name__ == "__main__":
    main()
