# -*- coding: utf-8 -*-

from math import*
import numpy as np
import matplotlib.pyplot as plt

########################################################################
#Parâmetros iniciais
n_layers=3
angulos=[-20,0.001,20]
rk=[23.00*10**(-3),23.10*10**(-3),23.90*10**(-3),24.00*10**(-3)]

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
#Criando uma matriz de rigidez transformada (CB) para cada camada do
#laminado
lista_cb=[]
for k in range(n_layers):
    theta=angulos[k]
    cb_k=CB(theta)
    lista_cb.append(cb_k)

###########################################################################
#Cálculo das matriz de rigidez do laminado
def matriz_acoplamento():
    ry=(rk[0]+rk[-1])/2
    a=np.zeros(shape=(3,3))
    b=np.zeros(shape=(3,3))
    d=np.zeros(shape=(3,3))
    for j in range(len(lista_cb)):
        i=lista_cb[j]
        cb=np.array([[i[0][0],i[0][1],i[0][5]],
                    [i[1][0],i[1][1],i[1][5]],
                    [i[5][0],i[5][1],i[5][5]]])
        zk=rk[j+1]-ry
        zk_1=rk[j]-ry
        a=a+cb*(zk-zk_1)
        b=b+0.5*cb*((zk**(2))-((zk_1)**(2)))
        d=d+(1/3)*cb*((zk**(3))-((zk_1)**(3)))
    return [a,b,d]

m_acoplamento=matriz_acoplamento()
########################################################################
#Cálculo da carga crítica de flambagem pela teoria de casca
def flambagem(alfa,beta,c1,c2):
    Nx,Ny,Nxy=1,0,0
    ry=(rk[0]+rk[-1])/2
    A=m_acoplamento[0]
    B=m_acoplamento[1]
    D=m_acoplamento[2]
    O=np.array([[-alfa,0,beta,0,0,0],
                [0,-beta,alfa,0,0,0],
                [0,1/ry,0,(alfa**(2))+(beta**(2)*c2**(2)),
                 (beta**(2))+(alfa**(2)*c1**(2)),2*alfa*beta*(1+c1*c2)]])
    L=np.array([[beta*c2,0,-alfa*c1,0,0,0],
                [0,alfa*c1,-beta*c2,0,0,0],
                [0,0,0,-2*alfa*beta*c2,-2*alfa*beta*c1,
                 2*(c1+alfa**2+c2*beta**2)]])
    Mo=np.array([[A[0][0],A[0][1],0,B[0][0],B[0][1],0],
                 [A[0][1],A[1][1],0,B[0][1],B[1][1],0],
                 [0,0,A[2][2],0,0,B[2][2]],
                 [B[0][0],B[0][1],0,D[0][0],D[0][1],0],
                 [B[0][1],B[1][1],0,D[0][1],D[1][1],0],
                 [0,0,B[2][2],0,0,D[2][2]]])
    M=np.array([[A[0][0],A[0][1],A[0][2],B[0][0],B[0][1],B[0][2]],
                 [A[1][0],A[1][1],A[1][2],B[1][0],B[1][1],B[1][2]],
                 [A[2][0],A[2][1],A[2][2],B[2][0],B[2][1],B[2][2]],
                 [B[0][0],B[0][1],B[0][2],D[0][0],D[0][1],D[0][2]],
                 [B[1][0],B[1][1],B[1][2],D[1][0],D[1][1],D[1][2]],
                 [B[2][0],B[2][1],B[2][2],D[2][0],D[2][1],D[2][2]]])
    Mn=M-Mo
    J=np.array([[0,0,0],[0,0,0],[0,0,1]])
    fi_1=Nx*((alfa**2)+(beta**2)*(c2**2))+Nxy*2*(c1*alfa**(2)+beta**(2)*c2**2)
    fi_1=fi_1+Ny*(c1*alfa**(2)+beta**2)
    fi_2=-2*alfa*beta*(c2*Nx+Nxy*(1+c1*c2)+Ny*c1)
    X=np.zeros(shape=(12,12))
    for i in range(6):
        for j in range(6):
            X[i][j]=Mo[i][j]
    for i in range(6):
        for j in range(6,12):
            X[i][j]=Mn[i][j-6]
    for i in range(6,12):
        for j in range(6):
            X[i][j]=Mn[i-6][j]
    for i in range(6,12):
        for j in range(6,12):
            X[i][j]=Mo[i-6][j-6]
    Y=np.zeros(shape=(6,12))
    for i in range(3):
        for j in range(6):
            Y[i][j]=O[i][j]
    for i in range(3):
        for j in range(6,12):
            Y[i][j]=L[i][j-6]
    for i in range(3,6):
        for j in range(6):
            Y[i][j]=L[i-3][j]
    for i in range(3,6):
        for j in range(6,12):
            Y[i][j]=O[i-3][j-6]
    Yt=np.transpose(Y)
    G=np.matmul(np.matmul(Y,X),Yt)
    J=np.zeros(shape=(6,6))
    J[2][2],J[5][5]=fi_1,fi_1
    J[2][5],J[5][2]=fi_2,fi_2
    c3=G[2][2]
    f3=G[2][5]
    c6=G[5][2]
    f6=G[5][5]
    G2=np.array([[c3,f3],
                 [c6,f6]])
    J2=np.array([[fi_1,fi_2],
                 [fi_2,fi_1]])
    try:
        a_valor,a_vetor=np.linalg.eig(np.dot(G2,np.linalg.inv(J2)))
    except:
        a_valor=[0,0]        
    return a_valor

    #autovalores,autovetores=np.linalg.eig(np.linalg.inv(np.dot(J,np.linalg.inv(G))))
    #print(autovetores)
########################################################################
#Cálculo da carga crítica de flambagem pela teoria de casca
def flambagem2(alfa,beta):
    ry=(rk[0]+rk[-1])/2
    A=m_acoplamento[0]
    B=m_acoplamento[1]
    D=m_acoplamento[2]
    Nx=1
    Ny=0
    f11=(A[0][0]*alfa**(2))+(A[2][2]*beta**(2))
    f12=(A[0][1]+A[2][2])*alfa*beta
    f13=-(A[0][1]/ry)*alfa-B[0][0]*alfa**(3)-B[0][1]*alfa*beta**(2)-B[2][2]*2*alfa*beta**2
    f22=(A[1][1]*beta**2)+A[2][2]*alfa**2
    f23=-B[1][1]*beta**(3)-(B[0][1]+2*B[2][2])*beta*alfa**(2)-A[1][1]*(beta/ry)
    f33=D[0][0]*alfa**(4)+2*(D[0][1]+2*D[2][2])*beta**(2)*alfa**(2)+D[1][1]*beta**(4)
    f33=f33+(1/ry)*(A[1][1]*(1/ry)+2*B[1][1]*beta**(2)+2*B[0][1]*alfa**(2))
    F3=np.array([[f11,f12,f13],
                 [f12,f22,f23],
                 [f13,f23,f33]])
    F2=np.array([[f11,f12],
                 [f12,f22]])
    lamb2=(1/(Nx*alfa**(2)+Ny*beta**(2)))*(np.linalg.det(F3)/np.linalg.det(F2))
    return lamb2


############################################################################
#Solução para laminados ortrópicos
def lambda_orto():
    rm=(rk[0]+rk[-1])*0.5
    lista_beta=np.linspace(1,200,2000)
    lista_alfa=[]
    for k in np.linspace(1,200,2000):
        lista_alfa.append(k*np.pi)
    lista_lamb=[]
    loc=[]
    for b in lista_beta:
        for a in lista_alfa:
            if flambagem2(a,b)>0:
                lamb_c=flambagem2(a,b)*2*np.pi*rm
                lista_lamb.append([lamb_c])
                loc.append([a,b,lamb_c])
    return (loc[lista_lamb.index(min(lista_lamb))])
############################################################################
#Solução para laminados anisotrópicos
def lambda_aniso():
    rm=(rk[0]+rk[-1])*0.5
    lista_beta=[]
    lista_alfa=[]
    for k in range(0,60):
        lista_alfa.append(k*np.pi)
        lista_beta.append(k)
    lista_c1=np.linspace(0,5,25)
    lista_c2=np.linspace(0,5,25)
    lista_lamb=[10**10]
    loc=[]
    for b in lista_beta:
        for a in lista_alfa:
            for c1 in lista_c1:
                for c2 in lista_c2:
                    lamb=flambagem(a,b,c1,c2)
                    if lamb[0]>0 and lamb[1]>0:
                        lamb_select=(2*np.pi*rm)*min(lamb[0],lamb[1])
                        if lamb_select<lista_lamb[0]:
                            lista_lamb=[lamb_select]
                            print(lista_lamb)
                            loc=[a,b,c1,c2,lamb_select]
    return loc

print(lambda_orto())


