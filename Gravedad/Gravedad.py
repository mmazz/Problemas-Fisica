# -*- coding: utf-8 -*-
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import scipy as sc

FontLabel = 30
FontTicks = 30
FontLegend = 20

d = 0.25
n = 64
x = np.linspace(0,1,n)
b = np.zeros(len(x))
f = np.zeros(len(x))
error = 5*np.random.uniform(-1,1,size=(len(f)))/1000000

normError = np.linalg.norm(error,2)


def K(s,t):# El kernel, lo que me describe a mi sistema
    k = d/np.power((np.power(d,2)+np.power(s-t,2)),3/2)
    return k

def F2(t): # otra distribucion de masas
    f = np.sin(np.pi*t) + 0.5*np.sin(2*np.pi*t)
    return f

def g(s): # la componente de la gravedad perpendicular, en s, es el dato
    g, e = integrate.quad(lambda t: np.dot(K(s,t),F2(t)),0,1)
    return g

for i in range(len(x)):
    f[i] = F2(x[i])
    b[i] = g(x[i])


def Agrav(d,n):
    A = np.zeros([n,n])
    s = np.zeros(n)
    t = np.zeros(n)
    for i in range(n):
        s[i] = (i-0.5)/n
        t[i] = (i-0.5)/n
    for i in range(n):
        for j in range(n):
            A[i][j] = (d/n)*(np.power(np.power(d,2)+np.power(((i-j)/n),2),-3/2))
    return A


def matrizFiltroTikh(sigma,lamb):
    filtro = np.zeros(len(sigma))
    for j in range(len(sigma)):
        filtro[j] =  np.power(sigma[j],2)/(np.power(sigma[j],2)+np.power(lamb,2))
    return filtro

def matrizFiltroTSVD(sigma,lamb):
    filtro = np.zeros(len(sigma))
    for j in range(lamb):
        filtro[j] =  1
    return filtro


def TikhonovSVD(A,b,param):
    global flag
    n = A.shape[1]
    r = np.linalg.matrix_rank(A)
    U, sigma, VT = sc.linalg.svd(A, full_matrices=False,overwrite_a=True,lapack_driver='gesvd')
    sigma_inv = np.diag(np.hstack([1/sigma]))
    psi = matrizFiltroTikh(sigma,param)
    Psi = np.diag(np.hstack([psi]))
    V = VT.T
    sigma_inv = np.diag(np.hstack([1/sigma[:r], np.zeros(n-r)]))
    Psisig = np.dot(Psi,sigma_inv)
    VPsisig =   np.dot(V,Psisig)
    VPsisigUT = np.dot(VPsisig,U.T)
    xsol = np.dot(VPsisigUT,b)
    bTik = np.dot(A,xsol)
    eps = np.linalg.norm(b-bTik)# Error of solution ||b-A*x||
    #print('\nError of the Tikhonov, lambda = {}, solution: ||b-A*x|| ='.format(param), eps)

    return xsol, bTik, sigma, U, Psi, sigma_inv, V

def TSVD(A,b,param):
    global flag
    n = A.shape[1]
    r = np.linalg.matrix_rank(A)
    U, sigma, VT = np.linalg.svd(A, full_matrices=True)
    sigma_inv = np.diag(np.hstack([1/sigma]))
    psi = matrizFiltroTSVD(sigma,param)
    Psi = np.diag(np.hstack([psi]))
    V = VT.T
    sigma_inv = np.diag(np.hstack([1/sigma[:r], np.zeros(n-r)]))
    Psisig = np.dot(Psi,sigma_inv)
    VPsisig =   np.dot(V,Psisig)
    VPsisigUT = np.dot(VPsisig,U.T)
    xsol = np.dot(VPsisigUT,b)
    bTik = np.dot(A,xsol)
    eps = np.linalg.norm(b-bTik)# Error of solution ||b-A*x||
    #print('\nError of the Tikhonov, lambda = {}, solution: ||b-A*x|| ='.format(param), eps)

    return xsol, bTik, sigma, U, Psi, sigma_inv, V

#%%
################################################################################
################ Veamos si logro tener el A de Ax=b
################################################################################
index = np.linspace(1,n,n)
AGrav = Agrav(0.25,n)  # Matriz A
Ai = np.linalg.inv(AGrav)
xsol = np.linalg.tensorsolve(Ai,b)  # El X, es decir mi distribucion f
bsol = np.dot(AGrav,xsol)  #
solExacta = np.dot(AGrav,f) # Pruebo que efectivamente esta A es lo que quiero
#plt.plot(i,solExacta)  # Me fijo si recupero la b (es decir la g)
#plt.plot(i,b) # la b posta
#plt.plot(i,bsol)
#RECUPERE! TENGO LA A! pero no estaria teniendo la f, tengo un factor 40
# la cond(A) es gigante.... pero me esperaba cualquier cosa pero sin ese factor de escala

b = np.dot(AGrav,f)
bError = b + error
################################################################################
############################# Veamos el grafico de Picard
################################################################################
xsolTik, bTik, sigma, U, Psi, sigma_inv , V= TikhonovSVD(AGrav,b,0)
uTb = np.abs(np.dot(U.T,b))
plt.figure(1)
sig = r" $\sigma_i$"
uTbstring = r" $\|u^Tb\|$"
#plt.title('Grafico de Picard',fontsize=34)
plt.yscale('log')
plt.scatter(index,sigma,label=sig,s=100)
plt.scatter(index,uTb,label=uTbstring ,s=100)
plt.scatter(index,uTb/sigma,label=uTbstring +'/'+sig,s=100)
plt.legend(fontsize=FontLabel)
plt.xlabel("Eje X",fontsize=FontLabel)
plt.ylabel("Eje Y",fontsize=FontLabel)
plt.xticks(fontsize=FontTicks)
plt.yticks(fontsize=FontTicks)
plt.savefig('Python-Picard.png')
################################################################################
############################# Veamos el grafico de Picard con Error
################################################################################
xsolTik, bTik, sigma, U, Psi, sigma_inv , V= TikhonovSVD(AGrav,bError,0)
uTb = np.abs(np.dot(U.T,bError))
xnormerror = np.linspace(0,index[-1])
normErrorVec = np.zeros(len(xnormerror))
normErrorString = r'$\Vert e\Vert_2$'


for i in range(len(xnormerror)):
    normErrorVec[i] = error.std()
plt.figure(2)
sig = r" $\sigma_i$"
uTbstring = r" $\|u^Tb\|$"
#plt.title('Grafico de Picard con error')
plt.yscale('log')

plt.plot(xnormerror, normErrorVec, '--',  linewidth=5)
plt.scatter(index,sigma,label=sig,s=100)
plt.scatter(index,uTb,label=uTbstring ,s=100)
plt.scatter(index,uTb/sigma,label=uTbstring +'/'+sig,s=100)
plt.legend(fontsize=FontLegend)
plt.xlabel("Eje X",fontsize=FontLabel)
plt.ylabel("Eje Y",fontsize=FontLabel)
plt.xticks(fontsize=FontTicks)
plt.yticks(fontsize=FontTicks)
plt.savefig('Python-PicardError.png')
#%%
################################################################################
############################# Comportamiento del factor de filtracion de Tik
################################################################################
lambda_prueba = 1
sigma_continuo = np.linspace(7,0,1000)
filtro = matrizFiltroTikh(sigma_continuo,lambda_prueba)
lambdaString = r" $\lambda$"
factorFiltro = r" $\varphi(\lambda)$"
plt.figure(3)
#plt.title('Factor de filtracion de Tikhonov')
plt.plot(sigma_continuo,filtro,marker="o",  linewidth=10)
plt.xlabel(lambdaString,fontsize=FontLabel)
plt.ylabel(factorFiltro ,fontsize=FontLabel)
plt.xticks(fontsize=FontTicks)
plt.yticks(fontsize=FontTicks)
plt.savefig('Python-FactirTik.png')
################################################################################
############################# Probemos Tikhonov!!! FUNCIONA!
################################################################################
#%%
plt.figure(4)
#plt.title('Tikhonov')
plt.plot(index,f,'*')

 # mejor 0.001
parametros = [0.00001,0.001,0.01,0.1,1,3]
for j in parametros:
    xsolTik, bTik, sigma, U, Psi, sigma_inv , V= TikhonovSVD(AGrav,bError,j)
    plt.plot(index,xsolTik,label='Tikhonov lambda={}'.format(j),  linewidth=5)

plt.plot(index,f, 'b*',label='Exacto', markersize=10)
plt.legend(fontsize=FontLegend)
plt.xlabel("X",fontsize=FontLabel)
plt.ylabel("Densidad de masa f(x)",fontsize=FontLabel)
plt.xticks(fontsize=FontTicks)
plt.yticks(fontsize=FontTicks)
plt.savefig('Python-Tikhonov.png')
#%%
################################################################################
#############################  Probemos Trucando!
################################################################################

parametrosK = [5,15,20,23] # 15 segun discrepancia
plt.figure(5)
#plt.title('TSVD')
for k in parametrosK:
    xsolTik, bTik, sigma, U, Psi, sigma_inv, V = TSVD(AGrav,bError,k)
    plt.plot(index,xsolTik,label='TSVD k={}'.format(k),  linewidth=5)

plt.plot(index,f,'*',label='Exacto',  linewidth=5)
plt.legend(fontsize=FontLegend)
plt.xlabel("X",fontsize=FontLabel)
plt.ylabel("Densidad de masa f(x)",fontsize=FontLabel)
plt.xticks(fontsize=FontTicks)
plt.yticks(fontsize=FontTicks)

#plt.savefig('Python-TSVD.png')

################################################################################
############################# TSVD Errores
################################################################################
#%%
plt.figure(12)
error = 5*np.random.uniform(-1,1,size=(len(f)))/1000000
xbiasString= r" $\Vert\Delta x_{bias}\Vert_2$"
xpertString= r" $\Vert\Delta x_{pert}\Vert_2$"
xString = r" $\Vert\Delta x_{bias}\Vert_2+\Vert\Delta x_{pert}\Vert_2$"
parametrosK = np.linspace(2,30,29,dtype=int) # 15 segun discrepancia

#plt.title('TSVD')
NormPert = np.zeros(len(parametrosK))
NormBias = np.zeros(len(parametrosK))
ind=0
for k in parametrosK:
    xsolTik, bTik, sigma, U, Psi, sigma_inv, V = TSVD(AGrav,bError,k)
    VSisig =   np.dot(V[:,:k],sigma_inv[:k,:k])
    VSisigUT = np.dot(VSisig,U.T[:k,:])
    xpert = np.dot(VSisigUT[:k],error)
    NormPert[ind] = np.linalg.norm(xpert,2)
    

    VTx = np.dot(V[:,k+1:],U.T[k+1:,:]) # f es la solucion exacta
    xbias = np.dot(VTx,f)
    NormBias[ind] = np.linalg.norm(xbias,2)
    ind = ind + 1
    
plt.yscale('log')

plt.scatter(parametrosK,NormPert,label=xpertString,s=100)
plt.plot(parametrosK,NormPert,  linewidth=5)
plt.scatter(parametrosK,NormBias ,label=xbiasString,s=100)
plt.plot(parametrosK,NormBias ,  linewidth=5)
plt.plot(parametrosK,NormBias+NormPert, label=xString,  linewidth=5)


#plt.plot(index,f,'*',label='Exacto')
plt.legend(fontsize=FontLegend)
plt.xlabel("k",fontsize=FontLabel)

plt.xticks(fontsize=FontTicks)
plt.yticks(fontsize=FontTicks)

#%%
################################################################################
############################# Curva L
################################################################################
parametros = [0.00000001,0.0000001,0.00001,0.0001,0.001,0.01,0.1,1,2,5]
xnormerror = np.linspace(0,max(parametros))
normErrorVec = np.zeros(len(xnormerror))
normxString = r'$\Vert x_{\lambda}\Vert_2$'
normAxbString = r'$\Vert Ax_{\lambda}-b\Vert_2$'
normErrorString = r'$\Vert e\Vert_2$'
for i in range(len(xnormerror)):
    normErrorVec[i] = normError
tick = -1

normx = np.zeros(len(parametros))
normAxb = np.zeros(len(parametros))
for j in parametros:
    tick = tick + 1
    xsolTik, bTik, sigma, U, Psi, sigma_inv, V = TikhonovSVD(AGrav,bError,j)
    normx[tick] = np.linalg.norm(xsolTik,2)
    normAxb[tick] = np.linalg.norm(bTik-b,2)
plt.figure(6)
#plt.title('Curva-L')

plt.xscale('log')
plt.yscale('log')
plt.scatter(normAxb,normx,  s=200,c='r')
plt.loglog(normAxb,normx,  linewidth=8)
plt.xlabel(normAxbString,fontsize=FontLabel)
plt.ylabel(normxString,fontsize=FontLabel)
plt.xticks(fontsize=FontTicks)
plt.yticks(fontsize=FontTicks)
plt.savefig('Python-CurvaL.png')
################################################################################
#############################  Discrepancia
################################################################################
#%%
tick = -1
normAxbString2 = r'$\Vert Ax_k-b\Vert_2$'
normErrorString = r'$\Vert e\Vert_2$'
normErrorString2 = r'$(n-k_{\eta})^{1/2}\eta$'
parametrosk =  [7,8,10,12,15,17,20,25]
xnormerror = np.linspace(min(parametrosk),max(parametrosk))
normErrorVec = np.zeros(len(xnormerror))
eta = error.std()
for i in range(len(xnormerror)):
    normErrorVec[i] = normError
normErrorVecSup = np.zeros(len(xnormerror))

kn=15
etaerror = np.sqrt(n-kn)*eta

for i in range(len(xnormerror)):
    normErrorVecSup[i] = etaerror

normx = np.zeros(len(parametrosk))
normAxb = np.zeros(len(parametrosk))
for j in parametrosk:
    tick = tick + 1
    xsolTik, bTik, sigma, U, Psi, sigma_inv, V = TSVD(AGrav,bError,j)
    normx[tick] = np.linalg.norm(xsolTik,2)
    normAxb[tick] = np.linalg.norm(bTik-bError,2)
    #normAxb[tick] = np.linalg.norm(bTik-b,2)
plt.figure(8)
#plt.title('Discrepancia TSVD')

plt.scatter(parametrosk,normAxb,  s=100,c='r')
plt.plot(parametrosk,normAxb,  linewidth=5, label=normAxbString2)

plt.plot(xnormerror, normErrorVec, '--',  linewidth=5, label=normErrorString)
plt.plot(xnormerror, normErrorVecSup, '--',  linewidth=5, label=normErrorString2)
plt.yscale('log')
plt.xlabel("k",fontsize=FontLabel)
plt.xticks(fontsize=FontTicks)
plt.yticks(fontsize=FontTicks)
plt.legend(fontsize=FontLegend)
#plt.scatter(parametrosk,np.log(normAxb))
#plt.scatter(parametrosk,np.log(normAxb))
################################################################################
#############################  Discrepancia Tik
################################################################################
#%%
tick = -1
parametros = [0.000001,0.00001,0.0001,0.001,0.003,0.005,0.01,0.1,1,5,10,15]
xnormerror = np.linspace(0,max(parametros))
normErrorVec = np.zeros(len(xnormerror))
for i in range(len(xnormerror)):
    normErrorVec[i] = normError

normx = np.zeros(len(parametros))
normAxb = np.zeros(len(parametros))
for j in parametros:
    tick = tick + 1
    xsolTik, bTik, sigma, U, Psi, sigma_inv, V = TikhonovSVD(AGrav,bError,j)
    normx[tick] = np.linalg.norm(xsolTik,2)
    normAxb[tick] = np.linalg.norm(bTik-bError,2)
    #normAxb[tick] = np.linalg.norm(bTik-b,2)
plt.figure(11)
#plt.title('Discrepancia TSVD')

plt.scatter(parametros,normAxb,  s=100,c='r')
plt.plot(parametros,normAxb,  linewidth=5)

plt.plot(xnormerror, normErrorVec, '--')
plt.yscale('log')

plt.xlabel(lambdaString,fontsize=FontLabel)
plt.ylabel(normAxbString,fontsize=FontLabel)
plt.xticks(fontsize=FontTicks)
plt.yticks(fontsize=FontTicks)
plt.savefig('Python-Discrepancia.png')
#plt.scatter(parametrosk,np.log(normAxb))
#plt.scatter(parametrosk,np.log(normAxb))
#%%
################################################################################
############################# Gen Cross validation Tikhonov
################################################################################
#%%
tick = -1
parametros = [0.000001,0.00001,0.0001,0.001,0.003,0.005,0.01,0.1,1]
normAxb = np.zeros(len(parametros))
G = np.zeros(len(parametros))
traza = 0
for j in parametros:
    tick = tick + 1
    xsolTik, bTik, sigma, U, Psi, sigma_inv , V= TikhonovSVD(AGrav,bError,j)
    traza = np.sum(matrizFiltroTikh(sigma,j))
    divisor = np.power(len(b)-traza,2)
    normAxb[tick] = np.power(np.linalg.norm(bTik-b,2),2)
    G[tick] = normAxb[tick]/divisor
plt.figure(9)
#plt.title('GCV-Tikhonov')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(lambdaString,fontsize=FontLabel)
plt.ylabel("G({})".format(lambdaString),fontsize=FontLabel)
plt.plot(parametros,G,  linewidth=5)
plt.scatter(parametros,G,  s=100,c='r')
plt.xticks(fontsize=FontTicks)
plt.yticks(fontsize=FontTicks)
plt.savefig('Python-GCVTik.png')

################################################################################
############################# Gen Cross validation TSVD
################################################################################
tick = -1
parametrosk =  [3,5,7,8,10,12,15,17,19,22]
normAxb = np.zeros(len(parametrosk))
Gk = np.zeros(len(parametrosk))
traza = 0
for j in parametrosk:
    tick = tick + 1
    xsolTik, bTik, sigma, U, Psi, sigma_inv, V = TSVD(AGrav,bError,j)
    traza = np.sum(matrizFiltroTSVD(sigma,j))
    divisor = np.power(len(b)-traza,2)
    normAxb[tick] = np.power(np.linalg.norm(bTik-b,2),2)
    Gk[tick] = normAxb[tick]/divisor
plt.figure(10)
#plt.title('GCV-TSVD')
plt.yscale('log')
plt.plot(parametrosk,Gk,  linewidth=5)
plt.scatter(parametrosk,Gk,  s=100,c='r')
plt.xlabel("k",fontsize=FontLabel)
plt.ylabel("G(k)",fontsize=FontLabel)
plt.xticks(fontsize=FontTicks)
plt.yticks(fontsize=FontTicks)
plt.savefig('Python-GCVTSVD.png')

################################################################################
#############################
################################################################################
