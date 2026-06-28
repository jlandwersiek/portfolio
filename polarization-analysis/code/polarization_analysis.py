#multiple regression using least squares fit to polynomial 
#use reduced chi squared to compare goodness of fit of degree 3/degree 4 polynomials

#data are observables from electron->proton deep inelastic scattering experiments

#Rsum is plotted against cosine theta - the angle between the plane of the incident beam and the reaction plane
    #K meson center of mass

#the function Rsum is dependent on:
    #the polarization transferred from electron beam to recoil hyperon,
    #the polarization of virtual photons exchanged during collision,
    #the total crossection - probability of event(scattering) i.e. interaction strength

#limitations of detectors at theta=0 require that polarization/Rsum values at cos(theta)=1 be extrapolated through data analysis 
#the value of Rsum when theta = 0 is an important parameter in determining the structure functions of the proton

#structure functions provide insight on quark-gluon dynamics:
    #quark - antiquark momentum distributions
    #quark - antiquark number density
    #transverse spin distributions tdue to scattering from transverse virtual photons
    #gluon dynamics due to scattering from longitudinal virtual photons
    #test quantum chromodynamics (QCD) predictions


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sympy import solve, symbols, diff, lambdify
from numpy.linalg import inv


#j=3, degree 3; j=4, degree 4
for j in range(3,5):
    print(f"Using degree {j}")   
    
    #pull files
    #data separated into bins of invariant energy W
    filenames = sorted(Path('.').glob('*.txt'))
    
    #column names for datafiles
    columns = ['Q2','W','cos','eps','o','do','pz1','dpz1','pz','dpz','c','R','dR']
    
    df = [pd.read_csv(f, names=columns, usecols=lambda col: col not in columns[-2:]) for f in filenames]
    
    #chunk dataframe for iterations through datasets
    data_set = [df[i] for i in range(8)]
    
    #datafile for energies, sigma,dsigma at 1, c0 at 1 for each dataset
    s = pd.read_csv('all_sigma_odoc1.csv', header=None, names=['dataset','o1','do1','c1','wp'])
     
        
    #iterate through datasets
    for i in range(0,8):
        #extract the dataset energy from s['dataset'] for the current dataset
        dataset_energy = s['dataset'][i]
        #data - polarization and crosssection data, dependent on angle between incident and reaction plane
        cos, pz, pz1, dpz, dpz1, o, do, c0 = [data_set[i][col] for col in ['cos', 'pz', 'pz1', 'dpz', 'dpz1', 'o', 'do','c']]
        Rs = (pz+pz1)*o/c0  #Rsum to be plotted against cos 
        dRpz = o/c0 #partial derivatives of Rsum to be used in error propagation
        dRo = (pz+pz1)/c0
        dRs= np.sqrt((dRpz*dpz)**2+(dRpz*dpz1)**2+(dRo*do)**2) #uncertainty of Rsum 
        
        
        #define symbolic parameters for algebraic manipulation
        if j == 3:
            a = symbols('a:3')
        else:
            a = symbols('a:4')
        x = symbols('x')
        
        #define symbolic function f with parameters a to be optimized
        f_sym = a[0] * (1 + x) + a[1] * (x + x**2) + a[2] * (x**3 - x)
        if j != 3:
            f_sym += a[3] * (x**4 + x)        
        
        #add cos values to symbolic f so only params a are unknown
        f_values = [f_sym.subs(x, cos) for cos in cos]

        #begin fitting routine 
        #define chi-squared function to be minimized 
        chi_squared = sum(((Rs - f) / dRs)**2 for Rs, f, dRs in zip(Rs, f_values, dRs))
      
        #compute derivatives of chi-squared with respect to each parameter
        dX0 = diff(chi_squared, a[0])
        dX1 = diff(chi_squared, a[1])
        dX2 = diff(chi_squared, a[2])
        if j != 3:
            dX3 = diff(chi_squared, a[3])
        
        #solve the system of coupled equations for a[0], a[1], a[2]        
        sol = solve([diff(chi_squared, param) for param in a], a)
        print('parameters: ', sol)
        
        #uncertainty in parameters determined by covariance matrix
        #to find covariance matrix, take the inverse of curvature matrix, alpha_lk = sum((f_l(x_i)*f_k(x_i))/var_i),
        #where x_i are plotted x values, f_l/f_k are the terms of the fitting function with params a divided out,
        #and var_i is the error in data squared
        
        #define polynomial terms as functions of cosine
        def polynomial_terms(cos, j):
            f0 = 1 + cos
            f1 = cos + cos**2
            f2 = cos**3 - cos
            terms = [f0, f1, f2]
            if j != 3:
                f3 = cos + cos**4
                terms.append(f3)
            return terms
        
        #define the alpha matrix elements
        def compute_alpha_matrix(terms, dRs):
            def alph(l, k):
                return np.sum((l * k) / (dRs**2))
            
            size = len(terms)
            alpha = np.zeros((size, size))
            
            for i in range(size):
                for j in range(size):
                    alpha[i, j] = alph(terms[i], terms[j])
            return alpha
        
        #compute the covariance matrix
        def compute_covariance_matrix(cos, dRs, j):
            terms = polynomial_terms(cos, j)  # Generate polynomial terms
            alpha = compute_alpha_matrix(terms, dRs)  # Create the alpha matrix
            covariance_matrix = inv(alpha)  # Compute the inverse (covariance matrix)
            return covariance_matrix
        
        c = compute_covariance_matrix(cos, dRs, j)
        
        def dy(p):
            #compute the polynomial derivatives with respect to parameters
            derivatives = [1. + p, p + p**2., p**3. - p]
            if j != 3:
                derivatives.append(p + p**4.)

            #calculate dy using the covariance matrix and derivatives
            terms = 0
            num_derivatives = len(derivatives)
            for i in range(num_derivatives):
                for k in range(i, num_derivatives):  # Only consider the upper triangle of the covariance matrix
                    factor = 2 if i != k else 1  # Double for off-diagonal elements
                    terms += factor * c[i, k] * derivatives[i] * derivatives[k]
            return np.sqrt(terms)
        
            
        #convert symbolic function f_sym into a numerical function
        f_num = lambdify(x, f_sym.subs(sol), 'numpy')
        
        #generate cos values for plotting
        cos_values = np.linspace(-1, 1, 100)
        
        #evaluate f(cos) with the optimized parameters
        f_plot = f_num(cos_values)
        
        #compute error bands using dy
        error_bands = np.array([dy(p) for p in cos_values])
        
        #upper and lower bounds of the error bands
        f_upper = f_plot + error_bands
        f_lower = f_plot - error_bands
        
        #plot fitted function and error bands
        plt.figure(figsize=(10, 6))
        plt.plot(cos_values, f_plot, label='Fitted Function $f(x)$', color='blue', linewidth=2)
        plt.fill_between(cos_values, f_lower, f_upper, color='lightblue', alpha=0.5, label='Error Band')
        plt.scatter(cos, Rs, color='red', label='Data Points')  #plot the data points
        plt.errorbar(cos, Rs, yerr=dRs, fmt='o', color='red', label='Data Uncertainty', capsize=3)
        
        #add plot labels and legend
        plt.title(f"Fitted Function with Error Bands (Degree {j}) for {dataset_energy}", fontsize=14)
        plt.xlabel('cos', fontsize=12)
        plt.ylabel('$f(cos)$', fontsize=12)
        plt.grid(True)
        plt.legend(fontsize=10)
        plt.show()
        
        
        #compute chi-squared value
        chi_squared_results = {}
        residuals = Rs - np.array([f_num(cos_val) for cos_val in cos])
        chi_squared_value = np.sum((residuals / dRs)**2)

        #degree of freedom: number of data points - number of parameters
        dof = len(cos) - len(a)

        #compute reduced chi-squared
        chi_squared_red = chi_squared_value / dof
        print(f"Dataset {i + 1}, Degree {j}, Reduced Chi-Squared: {chi_squared_red:.4f}")

        #store results
        chi_squared_results[(i, j)] = chi_squared_red