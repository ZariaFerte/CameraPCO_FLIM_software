# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 09:56:04 2024

@author: Zaria Ferté
"""
from lmfit import*
import math
import numpy as np
import sympy as sp


models_names = []
models_formula = []
models_infos=[]


def boltzmann(pH, Top=6, Bottom=0.5, V50=7.5, slope=1):
    return Bottom + (Top - Bottom) / (1 + 10 ** ((pH - V50) / slope))

cotphi, pH, Top, Bottom, V50, slope = sp.symbols('cotphi pH Top Bottom V50 slope ')
expr = boltzmann(pH, Top, Bottom, V50, slope)-cotphi
relation=sp.lambdify((cotphi, Top, Bottom, V50, slope),sp.solve(expr, pH)[0])
            
def fromphi(phi):
    if math.tan(math.radians(phi)) == 0:
        return 0
    else:
        return 1/(math.tan(math.radians(phi)))   # cotan = 1/tan1
        
model = Model(boltzmann, independent_vars=['pH'], nan_policy='omit', name='pH Boltzmann')
info=['cotdPhi (deg)','cotdPhi (deg)', relation, fromphi]
            
models_names.append(model.name.split('(')[1].rstrip(')'))
models_formula.append(model)
models_infos.append(info)




def linear (x, k=1, d=0):
    return k*x+d

x, k, d, y = sp.symbols('x k d y')
expr = linear(x,k,d)-y
relation=sp.lambdify((y,k,d),sp.solve(expr, x)[0])

def fromphi(phi):
    return phi

model = Model(linear,independent_vars=['x'], nan_policy='omit', name='Linear phase')
info=['phi (deg)','phi (deg)', relation, fromphi]

models_names.append(model.name.split('(')[1].rstrip(')'))
models_formula.append(model)
models_infos.append(info)
model = Model(linear,independent_vars=['x'], nan_policy='omit', name='Linear tau')
info=['tau (µs)','tau (µs)', relation, fromphi]

models_names.append(model.name.split('(')[1].rstrip(')'))
models_formula.append(model)
models_infos.append(info)




def ExpDecay (x, A=100, C=1, k=0.1):
    if any (isinstance(arg, (sp.Basic, sp.Expr)) for arg in [x, A, C, k]):
        return (C + A*sp.exp(-k*x))
        
    else:
        return (C + A*np.exp(-k*x))

y, x, A, C, k = sp.symbols('y x A C k')
expr = ExpDecay(x, A, C, k)-y
relation=sp.lambdify((y, A, C, k),sp.solve(expr, x)[0])

def fromphi(phi):
    return (phi)

model = Model(ExpDecay, independent_vars=['x'], nan_policy='omit', name='Exponential decay phase')
info=['phi (deg)','phi (deg)', relation, fromphi]      
models_names.append(model.name.split('(')[1].rstrip(')'))
models_formula.append(model)
models_infos.append(info)

model = Model(ExpDecay, independent_vars=['x'], nan_policy='omit', name='Exponential decay tau')
info=['tau (µs)','tau (µs)', relation, fromphi]       
models_names.append(model.name.split('(')[1].rstrip(')'))
models_formula.append(model)
models_infos.append(info)




def SternVolmerTwoSites (pO2, tau0=0, Ksv1=0.1, Ksv2=0.01, f=0.9):
    return (tau0*(f/(1 + Ksv1*pO2) + (1-f)/(1 + Ksv2*pO2)))

tau, pO2, tau0, Ksv1, Ksv2, f= sp.symbols('tau pO2 tau0 Ksv1 Ksv2 f')
expr = SternVolmerTwoSites(pO2, tau0, Ksv1, Ksv2, f)-tau
relation=sp.lambdify((tau, tau0, Ksv1, Ksv2, f),sp.solve(expr, pO2)[0])

def fromphi(phi): #(Tau will be calculated in the main program)
    return phi

model = Model(SternVolmerTwoSites, independent_vars=['pO2'], nan_policy='omit', name='Stern Volmer Two Sites')
info=['tau (µs)','tau (µs)', relation, fromphi]
          
models_names.append(model.name.split('(')[1].rstrip(')'))
models_formula.append(model)
models_infos.append(info)





kB=1.380649
def Arrhenius (T, k0=1e-3, k1=1e-3, dE=1e-20):
    if any (isinstance(arg, (sp.Basic, sp.Expr)) for arg in [T, k0, k1, dE]):
        return (1/(k0+k1*sp.exp(-dE/(kB *(T+273.15)))))
    else:
        return (1/(k0+k1*np.exp(-dE/(kB*(T+273.15)))))

tau, T, k0, k1, dE = sp.symbols('tau T k0 k1 dE')
expr = Arrhenius(T, k0, k1, dE)-tau
sol=sp.solve(expr, T)[0]
sol=sol*10**23
if sol:
    relation=sp.lambdify((tau, k0, k1, dE),sol)
    
def fromphi(phi): #(Tau will be calculated in the main program)
    return phi

model = Model(Arrhenius, independent_vars=['T'], nan_policy='omit', name='Arrhenius (T° in °C)')
info=['tau (µs)','tau (µs)', relation, fromphi]
          
models_names.append(model.name.split('(')[1].rstrip(')'))
models_formula.append(model)
models_infos.append(info)

