# -*- coding: utf-8 -*-

from math import*
import numpy as np
import matplotlib.pyplot as plt

########################################################################
#Parâmetros iniciais
n_layers=3
angulos=[90,0.001,90]
rk=[23.00*10**(-3),23.25*10**(-3),23.75*10**(-3),24.00*10**(-3)]
Fe=-10000
Te=0
p0=0

########################################################################
#Propriedades do compósito
E1 = 125*10**9
E2 = 8*10**9;
E3 = E2;
mu12 = 0.35;
mu13 = mu12;
mu23 = 0.02;
G12 = 4*10**9;
G13 = G12;
G23 = E2/(2*(1 + mu23))
Xt=2090*10**6
Yt=64*10**6
Zt=Yt
Xc=-1717*10**6
Yc=-210*10**6
Zc=Yc
S1=186*10**6
########################################################################
#Matriz de flexibilidade
def S():
    S = np.array([[1/E1, -(mu12/E1), -(mu13/E1), 0, 0, 0],
        [-(mu12/E1), 1/E2, -(mu23/E2), 0, 0, 0], 
        [-(mu13/E1), -(mu23/E2), 1/E3, 0, 0, 0],
        [0, 0, 0, 1/G23, 0, 0],
        [0, 0, 0, 0, 1/G13, 0],
        [0, 0, 0, 0, 0, 1/G12]])
    return S
########################################################################
#Gera a matriz de rigidez
C = np.linalg.inv(S())
########################################################################
# Matriz de transformação T1 em função de theta
def T1(theta):
  m = np.cos(theta*pi/180)
  n = np.sin(theta*pi/180)
  T1 = np.array([[m**2, n**2, 0, 0, 0, 2*m*n],
               [n**2, m**2, 0, 0, 0, -2*m*n],
               [0, 0, 1, 0, 0, 0], 
               [0, 0, 0, m, -n, 0],
               [0, 0, 0, n, m, 0],
               [-m*n, m*n, 0, 0, 0, m**2 - n**2]])
  return T1
########################################################################
# Matriz de transformação T2 em função de theta
def T2(theta):
  m = np.cos(theta*pi/180)
  n = np.sin(theta*pi/180)
  T2 = np.array([[m**2, n**2, 0, 0, 0, m*n],
                [n**2, m**2, 0, 0, 0, -m*n],
                [0, 0, 1, 0, 0, 0], 
                [0, 0, 0, m, -n, 0],
                [0, 0, 0, n, m, 0],
                [-2*m*n, 2*m*n, 0, 0, 0,m**2 - n**2]])
  return T2
########################################################################
#Gera a matriz de rigidez em coordenadas cilíndricas
def CB(theta):
    T1inv = np.linalg.inv(T1(theta))
    cb = np.dot(np.inner(T1inv,C), T2(theta))
    return cb
########################################################################
#Calcula o coeficiente alfa_1 com base na matriz de rigidez dada
def alfa_1 (cb):
  alfa_1=(cb[0][1]-cb[0][2])/(cb[2][2]-cb[1][1])
  return alfa_1
########################################################################
#Calcula o coeficiente alfa_1 com base na matriz de rigidez dada
def alfa_2(cb):
  alfa_2=(cb[1][5]-2*cb[2][5])/(4*cb[2][2]-cb[1][1])
  return alfa_2
########################################################################
#Calcula o coeficiente alfa_1 com base na matriz de rigidez dada
def beta(cb):
  beta=(cb[1][1]/cb[2][2])**(0.5)
  return beta
########################################################################
#Criando uma matriz de rigidez transformada (CB) para cada camada do
#laminado
lista_cb=[]
for k in range(n_layers):
    theta=angulos[k]
    cb_k=CB(theta)
    lista_cb.append(cb_k)
########################################################################
#Define os coeficiente para a condição de equilíbrio de deslocamento
#entre uma camada k (mais interna) e k1(mais externa)
def cond_eq_desloc(r,k):
    c_k=lista_cb[k]
    c_k1=lista_cb[k+1]
    b_k=beta(c_k)
    b_k1=beta(c_k1)
    a1_k=alfa_1(c_k)
    a1_k1=alfa_1(c_k1)
    a2_k=alfa_2(c_k)
    a2_k1=alfa_2(c_k1)
    coef_Dk=r**b_k
    coef_Dk1=-r**b_k1
    coef_Ek=r**(-b_k)
    coef_Ek1=-r**(-b_k1)
    coef_e0=r*(a1_k-a1_k1)
    coef_g0=(r**2)*(a2_k-a2_k1)
    coefs=np.zeros(2*n_layers+2)
    coefs[k]=coef_Dk
    coefs[k+1]=coef_Dk1
    coefs[k+3]=coef_Ek
    coefs[k+4]=coef_Ek1
    coefs[6]=coef_e0
    coefs[7]=coef_g0
    return coefs
########################################################################
#Define os coeficientes para a tensão radial de uma camada k
def m_sigma_r(r,k):
    cb=lista_cb[k]
    b=beta(cb)
    a1=alfa_1(cb)
    a2=alfa_2(cb)
    coef_D=r**(b-1)*cb[1][2]+cb[2][2]*(r**(-1+b))*b
    coef_E=r**(-1-b)*cb[1][2]-cb[2][2]*(r**(-1-b))*b
    coef_e0=cb[0][2]+a1*(cb[1][2]+cb[2][2])
    coef_g0=r*(cb[1][2]*a2+2*cb[2][2]*a2+cb[1][5])
    coefs=np.zeros(2*n_layers+2)
    coefs[k]=coef_D
    coefs[k+3]=coef_E
    coefs[6]=coef_e0
    coefs[7]=coef_g0
    return coefs
########################################################################   
#Define os coeficientes para a tensão axial de uma camada k
def m_sigma_z(r,k):
    cb=lista_cb[k]
    b=beta(cb)
    a1=alfa_1(cb)
    a2=alfa_2(cb)
    coef_D=r**(b-1)*cb[0][1]+cb[0][2]*(r**(-1-b))*b
    coef_E=r**(-1-b)*cb[0][1]-cb[0][2]*(r**(-1-b))*b
    coef_e0=cb[0][0]+a1*(cb[0][1]+cb[0][2])
    coef_g0=cb[0][1]+a2*r+cb[0][2]*2*r*a2+cb[0][5]*r
    coefs=np.zeros(2*n_layers+2)
    coefs[k]=coef_D
    coefs[k+3]=coef_E
    coefs[6]=coef_e0
    coefs[7]=coef_g0
    return coefs
########################################################################
#Define os coeficientes para a integral de força axial em uma camada k
def int1(k):
    cb=lista_cb[k]
    b=beta(cb)
    a1=alfa_1(cb)
    a2=alfa_2(cb)
    cDa=((-rk[k]**(1+b)+rk[k+1]**(1+b))*(cb[0][1]+cb[0][2]*b))/(1+b)
    cEa=((rk[k]**(-b))*(rk[k+1]**(-b))*((-rk[k]**(b))*rk[k+1]
    +rk[k]*rk[k+1]**(b))*(cb[0][1]-cb[0][2]*b))/(-1+b)
    ce=-0.5*(rk[k]**(2)-rk[k+1]**(2))*(cb[0][0]+a1*(cb[0][1]+cb[0][2]))
    cg0=-(1/3)*(rk[k]**(3)-rk[k+1]**(3))*(cb[0][5]+
    a2*(cb[0][1]+2*cb[0][2]))
    coefs=np.zeros(2*n_layers+2)
    coefs[k]=cDa
    coefs[k+3]=cEa
    coefs[6]=ce
    coefs[7]=cg0
    return coefs
########################################################################
#Define os coeficientes para a integral de força axial em uma camada k
def int2(k):
    cb=lista_cb[k]
    b=beta(cb)
    a1=alfa_1(cb)
    a2=alfa_2(cb)
    cDa=((-rk[k]**(2+b)+rk[k+1]**(2+b))*(cb[1][5]+cb[2][5]*b))/(2+b)
    cEa=((rk[k]**(-b))*(rk[k+1]**(-b))*((-rk[k]**(b))*(rk[k+1]**(2))+
    (rk[k]**(2))*(rk[k+1]**(b)))*(cb[1][5]-cb[2][5]*b))/(-2+b)
    ce=-(1/3)*(rk[k]**(3)-rk[k+1]**(3))*(cb[0][5]+a1*(cb[1][5]+cb[2][5]))
    cg0=-0.25*(rk[k]**(4)-rk[k+1]**(4))*(cb[5][5]+a2*(cb[1][5]+2*cb[2][5]))
    coefs=np.zeros(2*n_layers+2)
    coefs[k]=cDa
    coefs[k+3]=cEa
    coefs[6]=ce
    coefs[7]=cg0
    return coefs
########################################################################
#Definição das linhas na matriz do sistema linear
eq_1=cond_eq_desloc(rk[1],0)
eq_2=cond_eq_desloc(rk[2],1)
eq_3=m_sigma_r(rk[1],0)-m_sigma_r(rk[1],1)
eq_4=m_sigma_r(rk[2],1)-m_sigma_r(rk[2],2)
eq_5=m_sigma_r(rk[0],0)
eq_6=m_sigma_r(rk[3],2)
eq_7=2*pi*(int1(0)+int1(1)+int1(2))
eq_8=2*pi*(int2(0)+int2(1)+int2(2))
########################################################################
#Resoloção do sistema linear
cond_contorno=np.array([[0],[0],[0],[0],[-p0],[0],[p0*pi*(rk[0]**2)+Fe],[Te]])
m=np.array([eq_1,eq_2,eq_3,eq_4,eq_5,eq_6,eq_7,eq_8])
m_c=np.dot(np.linalg.inv(m),cond_contorno)
########################################################################
#Função para calcular a deformação radial em r, numa camada k
def er(r,k):
    e0=float(m_c[6])
    g0=float(m_c[7])
    d=float(m_c[k])
    e=float(m_c[k+3])
    cb=lista_cb[k]
    b=beta(cb)
    a1=alfa_1(cb)
    a2=alfa_2(cb)
    e_r=(e0*a1)+(2*g0*r*a2)+((r**(-1+b))*d*b)-((r**(-1-b))*e*b)
    return e_r
########################################################################
#Função para calcular o deslocamento radial em r, numa camada k
def ur(r,k):
    e0=float(m_c[6])
    g0=float(m_c[7])
    d=float(m_c[k])
    e=float(m_c[k+3])
    cb=lista_cb[k]
    b=beta(cb)
    a1=alfa_1(cb)
    a2=alfa_2(cb)
    u_r=(d*r**b)+(e*r**(-b))+(a1*e0*r)+(a2*(r**(2))*g0)
    return u_r
########################################################################
#Função para calcular a deformação angular em r, numa camada k
def e_theta(r,k):
    u=ur(r,k)
    return u/r
########################################################################
#Função para calcular a deformação em cisalhamento em r, numa camada k
def gama_z_theta(r):
    g0=float(m_c[7])
    return g0*r
########################################################################
#Função para calcular a tensão axial em r, numa camada k
def sigma_z(r,k):
    cb=lista_cb[k]
    cb11=cb[0][0]
    cb12=cb[0][1]
    cb13=cb[0][2]
    cb16=cb[0][5]
    e0=float(m_c[6])
    sig_z=cb11*e0+cb12*e_theta(r,k)+cb13*er(r,k)+cb16*gama_z_theta(r)
    return sig_z
########################################################################
#Função para calcular a tensão circunferencial em r, numa camada k
def sigma_theta(r,k):
    cb=lista_cb[k]
    cb12=cb[0][1]
    cb22=cb[1][1]
    cb23=cb[1][2]
    cb26=cb[1][5]
    e0=float(m_c[6])
    sig_theta=cb12*e0+cb22*e_theta(r,k)+cb23*er(r,k)+cb26*gama_z_theta(r)
    return sig_theta
########################################################################
#Função para calcular a tensão radial em r, numa camada k
def sigma_r(r,k):
    cb=lista_cb[k]
    cb13=cb[0][2]
    cb33=cb[2][2]
    cb23=cb[1][2]
    cb36=cb[2][5]
    e0=float(m_c[6])
    sig_r=cb13*e0+cb23*e_theta(r,k)+cb33*er(r,k)+cb36*gama_z_theta(r)
    return sig_r
########################################################################
#Função para calcular a tensão de cisalhamento em r, numa camada k
def tau_ztheta(r,k):
    cb=lista_cb[k]
    cb16=cb[0][5]
    cb26=cb[1][5]
    cb36=cb[2][5]
    cb66=cb[5][5]
    e0=float(m_c[6])
    tau_zt=cb16*e0+cb26*e_theta(r,k)+cb36*er(r,k)+cb66*gama_z_theta(r)
    return tau_zt
########################################################################
#Função que gera o vetor de tensões em coordenadas cilíndricas
def sigma_cyl(r,k):
    return np.array([[sigma_z(r,k)],
                     [sigma_theta(r,k)],
                     [sigma_r(r,k)],
                     [0],
                     [0],
                     [tau_ztheta(r,k)]])
########################################################################
#Função que gera o vetor de tensões em coordenadas do material
def sigma_mat(r,k):
    return np.dot(T1(angulos[k]),sigma_cyl(r,k))
########################################################################
#Função que retorna o índice de falha sob o critério de Tsai-Hill para
#cada r, numa camada k 
def IFTsaiHill(r,k):
    sig_1=float(sigma_mat(r,k)[0])
    sig_2=float(sigma_mat(r,k)[1])
    sig_12=float(sigma_mat(r,k)[2])
    if sig_1>0:
        X=Xt
    else:
        X=Xc
    if sig_2>0:
        Y=Yt
    else:
        Y=Yc
    IF=(sig_1**2)/(X**2)-((sig_1*sig_2)/X**2)+(sig_2**2)/(Y**2)+(sig_12**2)/(S1**2)
    return IF**0.5
########################################################################
#Função que retorna o índice de falha sob o critério de Tensão Máxima para
#cada em r, numa camada k 
def IFMaxStress(r,k):
    sig_1=float(sigma_mat(r,k)[0])
    sig_2=float(sigma_mat(r,k)[1])
    sig_12=float(sigma_mat(r,k)[2])
    if sig_1>0:
        X=Xt
    else:
        X=Xc
    if sig_2>0:
        Y=Yt
    else:
        Y=Yc
    IF=max(sig_1/X,sig_2/Y,abs(sig_12/S1))
    return IF
########################################################################
#Função que gera o gráfico de tensão axial ao longo das camadas
def plot_sig_Z():
    lista_x=[]
    lista_y=[]
    for k in range(n_layers):
        sig_z_k=float(sigma_cyl(rk[k],k)[0])/10**6
        lista_x.append(rk[k]*10**3)
        lista_y.append(sig_z_k)
        sig_z_k1=float(sigma_cyl(rk[k+1],k)[0])/10**6
        lista_x.append(rk[k+1]*10**3)
        lista_y.append(sig_z_k1)
    plt.plot(lista_x,lista_y)
    plt.grid(color='gray',linestyle='--',linewidth=0.5)
    plt.title('Tensão Axial [MPa] x Raio [mm]')
    plt.xlabel('Raio [mm]')
    plt.ylabel('Tensão Axial [MPa]')
    plt.show()
########################################################################
#Função que gera o gráfico de tensão circunferencial ao longo das camadas
def plot_sig_hoop():
    lista_x=[]
    lista_y=[]
    for k in range(n_layers):
        sig_hoop_k=float(sigma_cyl(rk[k],k)[1])/10**6
        lista_x.append(rk[k]*10**3)
        lista_y.append(sig_hoop_k)
        sig_hoop_k1=float(sigma_cyl(rk[k+1],k)[1])/10**6
        lista_x.append(rk[k+1]*10**3)
        lista_y.append(sig_hoop_k1)
    plt.plot(lista_x,lista_y)
    plt.grid(color='gray',linestyle='--',linewidth=0.5)
    plt.title('Tensão Circunferencial [MPa] x Raio [mm]')
    plt.xlabel('Raio [mm]')
    plt.ylabel('Tensão Circunferencial [MPa]')
    plt.show()
########################################################################
#Função que gera o gráfico de tensão radial ao longo das camadas
def plot_sig_r():
    lista_x=[]
    lista_y=[]
    for k in range(n_layers):
        sig_r_k=float(sigma_cyl(rk[k],k)[2])/10**6
        lista_x.append(rk[k]*10**3)
        lista_y.append(sig_r_k)
        sig_r_k1=float(sigma_cyl(rk[k+1],k)[2])/10**6
        lista_x.append(rk[k+1]*10**3)
        lista_y.append(sig_r_k1)
    plt.plot(lista_x,lista_y)
    plt.grid(color='gray',linestyle='--',linewidth=0.5)
    plt.title('Tensão Radial [MPa] x Raio [mm]')
    plt.xlabel('Raio [mm]')
    plt.ylabel('Tensão Radial [MPa]')
    plt.show()
########################################################################
#Função que gera o gráfico de IF para Tsail-Hill
def plot_IF_TH():
    lista_x=[]
    lista_y=[]
    for k in range(n_layers):
        if_k=float(IFTsaiHill(rk[k],k))
        lista_x.append(rk[k]*10**3)
        lista_y.append(Fe/if_k)
        if_k1=float(IFTsaiHill(rk[k+1],k))
        lista_x.append(rk[k+1]*10**3)
        lista_y.append(Fe/if_k1)
    plt.plot(lista_x,lista_y)
    plt.grid(color='gray',linestyle='--',linewidth=0.5)
    plt.title('Índice de Falha x Raio [mm]')
    plt.xlabel('Raio [mm]')
    plt.ylabel('IF')
    plt.show()
########################################################################
#Função que gera o gráfico de IF para o critério de Tensão Máxima
def plot_IF_MS():
    lista_x=[]
    lista_y=[]
    for k in range(n_layers):
        if_k=float(IFMaxStress(rk[k],k))
        lista_x.append(rk[k]*10**3)
        lista_y.append(if_k)
        if_k1=float(IFMaxStress(rk[k+1],k))
        lista_x.append(rk[k+1]*10**3)
        lista_y.append(if_k1)
    plt.plot(lista_x,lista_y)
    plt.grid(color='gray',linestyle='--',linewidth=0.5)
    plt.title('Índice de Falha x Raio [mm]')
    plt.xlabel('Raio [mm]')
    plt.ylabel('IF')
    plt.show()
########################################################################

print(IFMaxStress(rk[2],2))
print(IFTsaiHill(rk[2],2))
plot_IF_MS()



