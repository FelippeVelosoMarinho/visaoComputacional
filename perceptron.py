import numpy as np

#https://colab.research.google.com/drive/18BC82lz1xN1v5Mg5S4mSVvHL_DmLb0zF?usp=sharing#scrollTo=vSSBvnkyYEjL

x1 = np.array([0,0,1])
x2 = np.array([1,0,1])
x3 = np.array([1,1,1])
x4 = np.array([1,1,0])

values = [x1,x2,x3,x4,x1]

val_esperados = np.array([1,1,0,0,1])
Eta = 0.1
Bias = 0
W = np.array([0,0,0])

def somaPonderada(VETOR_x,VETOR_w):
    m = np.multiply(VETOR_x,VETOR_w)
    u = np.sum(m) + 1 * Bias
    return u

def funcaoAtivacao(u):
    if u > 0 :
        return 1
    else :
        return 0

def saida(vetorX, vetorW):
    u = somaPonderada(vetorX, vetorW)
    return funcaoAtivacao(u) #y do neuronio


#Ajuste de pesos
def ajustePesos(Eta, W, X, Bias, valorEsperado, saidaModelo):
    erro = valorEsperado - saidaModelo
    novoW = np.array([W[0] + Eta * X[0] * erro,
                    W[1] + Eta * X[1] * erro,
                    W[2] + Eta * X[2] * erro
                    ])
    novoBias = Bias + erro*Eta*1
    return novoW, novoBias


#iterando/testes
esperado_test = val_esperados[0]
print(f"esperado: {esperado_test}")
val_x_test = values[0]

y = saida(val_x_test, W)
print(f"y: {y}")
print(f"y: {W}")
pesos,bias = ajustePesos(Eta,W,val_x_test, Bias, esperado_test, y)

print(f"pesos: {pesos}")
print(f"bias: {bias}")