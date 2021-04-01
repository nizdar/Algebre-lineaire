# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 09:58:15 2020

@author: NIZDAR
"""

import numpy as np
from copy import deepcopy
import numpy.linalg as lg

def pivomax(m, c): 
    max=abs(m[c][c]) #initialiser avec l'element de la diagonale
    indicemax=c
    for i in range(c+1,len(m)): 
        if abs(m[i][c])>max:
            max=abs(m[i][c])
            indicemax=i
    return(indicemax)
    
def echange(m, l1, l2):
    x=m[l2,:].copy()
    m[l2,:]=m[l1,:]
    m[l1,:]=x
    
def dilatation(m,i,a):
    m[i,:]=a*m[i,:]
    
def transvection(m,i,j,a) :
    m[i,:]=m[i,:]+a*m[j,:] 
    
def augmente_identite(m):
    n=len(m)
    md=np.concatenate((m,np.eye(n,n)), axis=1)
    return md 

def augmente_vecteur(m,b):
    a=deepcopy(m)
    b1=deepcopy(b)
    return(np.concatenate((a,b1.T), axis=1))

#...............Forme echlonnée d'une matrice m................
def echlonne(m):
    n=len(m)
    for j in range(0,n-1): 
        pivo=pivomax(m, j) 
        if m[pivo][j]!=0:
            echange(m, pivo, j) 
            for i in range(j+1, n): 
                if m[i][j]!=0: 
                    a=-m[i][j]/m[j][j]          
                    transvection(m,i,j,a)    
    return m 

#Forme echlonnée d'une matrice m carrée concatenée à droite par l'identité
def echlonne_augmente(m):
    m=augmente_identite(m)
    echlonne(m)
    return m 

#Forme echlonnée reduite d'une matrice m carrée concatenée à droite par l'identité
def echlonne_reduit_augmente(m):
    n=len(m)
    m=echlonne_augmente(m) 
    for j in range(n-1,-1,-1): 
        if m[j][j]!=0:
            dilatation(m,j,1/m[j][j])
            for i in range(j-1,-1,-1):
                transvection(m,i,j,-m[i][j])
    return m

#forme echlonnée reduite d'une matrice m concatenée à droite par le transposé du vecteur b
def echlonne_reduit(m,b):
    n=len(m)
    m=augmente_vecteur(m,b) 
    m=echlonne(m)  
    for j in range(n-1,-1,-1): 
        if m[j][j]!=0:
            dilatation(m,j,1/m[j][j])
            for i in range(j-1,-1,-1):
                transvection(m,i,j,-m[i][j])
    return m 

#.........calculer determinant de m après echlonnement......... 
def det_echlonne(m):
    n=len(m)
    m=echlonne(m)
    k=1
    for j in range(n) :
        k*=m[j][j]
        if k==0 :
            break
    return k

#.......verifier si une matrice carrée m est inversible.........    
def test_inverse(m):
    a=deepcopy(m)
    return det_echlonne(a)!=0

#calculer l'invese d'une matrice en augmentant avec la matrice identité....
def inverse(m):
    n=len(m)
    if test_inverse(m):
        md=echlonne_reduit_augmente(m)
        inv=md[0 : n , n : 2*n]
        return inv
    else:
        return "m n'est pas inversible"

#......calculer la matrice m-val*Id avec val une valeur propre de m.....
def matrice_esp_propre(m, val):
    n=len(m)
    return m-val*np.eye(n,n)  

#.......vérifier si un vecteur est nul........   
def vecteur_nul(b):
    n=len(b[0])
    e=0 
    for j in range(0,n):
        if b[0][j]==0 :
            e+=1
    return e==n

#........determiner une base pour le ker d'une matrice m......
def base_ker(m):
    n=len(m) 
    if test_inverse(m):
        x="Le noyau est reduit à 0"  
    else:
        m=echlonne_augmente(np.transpose(m)) 
        for i in range(n-1,-1,-1):
            if vecteur_nul(m[i:i+1, 0:n]):
                x=m[i:i+1, n:2*n]
    return x

#........determiner une base pour le sous espace caractéristique associé à val               
def esp_propre(m,val):
    return base_ker(matrice_esp_propre(m, val)) 

#Calcul de la solution unique d'un système linéaire avec la methode de la remonté
def gauss_unique(m,b): 
    mb=echlonne(augmente_vecteur(m,b)) 
    n=len(mb)
    x=[0]*n 
    x[n-1]=mb[n-1][n]/mb[n-1][n-1] 
    for i in range(n-2,-1,-1):  
        for j in range(i+1,n): 
            mb[i][n]=mb[i][n]-mb[i][j]*x[j]
        x[i]=mb[i][n]/mb[i][i]
    return(x)

#......calcul d'une solution particulière d'un système lineaire non homogène........
def gauss_sol_part(m,b):
    n=len(m)
    a=deepcopy(m)
    a=echlonne_reduit(a,b)
    return np.transpose(a[0:n, n:n+1])   

#........resolution d'un système lineaire non homogène.........
def resolve_non_homog(m,b):
    if test_inverse(m):
        print("le système admet l'unique solution")
        print(gauss_unique(m,b))
    else:
        a=deepcopy(m)
        a=augmente_vecteur(a,b)
        if lg.matrix_rank(m)==lg.matrix_rank(a):
            print("le système admet une infinité de solution, une solution particulière est:")
            print(gauss_sol_part(m,b)) 
        else:
            print("le système n'admet pas de solutions")

#resolution d'un système lineaire homogène, ça revient à trouver des elements du ker de m         .
def resolve_homog(m,b):
    base_ker(m)

#............resoudre un système lineaire donné...........
def resolve(m,b):
    if vecteur_nul(b):
        resolve_homog(m,b)
    else:
        resolve_non_homog(m,b)

#----------------------------------------------------------------------------------------
#determiner la base associée à un block de jordan d'un vecteur b, elemt de base de l'espace caracteristique
def block(m,b):
    base=np.transpose(b)
    b1=deepcopy(b) 
    a=deepcopy(m)
    a1=augmente_vecteur(a,b)
    while lg.matrix_rank(m)==lg.matrix_rank(a1):
        b1=gauss_sol_part(a,b1)
        base=np.concatenate((base,np.transpose(b1)), axis=1)
        a1=augmente_vecteur(a,b1)  
    return base

#determiner les bases des blocks de Jordan associés à un espace caracteristique 
def base_e(m,val):
    a=deepcopy(m)
    a=matrice_esp_propre(m, val)
    espace=base_ker(a)
    n=len(espace)

    #extraire le premier vecteur de la base qui se trouve dans la première ligne de espace 
    b=espace[0:1,0:len(espace[0])]  
    base=block(a,b)    
    for i in range(1,n):
        b=espace[i:i+1,0:len(espace[0])]
        base=np.concatenate((base, block(a,b)), axis=1) 
    return base 

#----------------------------------------------------------------------------------------
