import numpy as np
import warnings as wr
from sklearn import neural_network as nn
from sklearn.exceptions import ConvergenceWarning

'''
        g: 00
        a: 01
        c: 10
        f: 11
'''


# taxa de aprendizado
lr = 0.01

# 1º: gfcgafg ( Valida )
X_primeira_sentenca_entrada = np.array([
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0],  # Q0 --> g --> Q1    
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,1],  # Q1 --> f --> Q2
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,0],  # Q2 --> c --> Q4
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0],  # Q4 --> g --> Q8
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,1],  # Q8 --> a --> Q12
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,1],  # Q12 --> f --> Q16
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,0],  # Q16 --> g --> Q18
         ])

Y_primeira_sentenca_saida = np.array([
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,1],  # Q0 --> g --> Q1    
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,0],  # Q1 --> f --> Q2
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0],  # Q2 --> c --> Q4
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,1],  # Q4 --> g --> Q8
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,1],  # Q8 --> a --> Q12
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,0],  # Q12 --> f --> Q16
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,0],  # Q16 --> g --> Q18
                ])

X_teste = np.array([
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0]  # Q0 --> g --> Q1    

         ])


mlp = nn.MLPClassifier(hidden_layer_sizes=(40,), max_iter=128, alpha=1e-4, solver='sgd', verbose=10, random_state=1, learning_rate_init=lr)

# treino
print('#################### EXECUCAO ####################')
print('Treinamento') 
with wr.catch_warnings():
    wr.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    mlp.fit(X_primeira_sentenca_entrada, Y_primeira_sentenca_saida)

# teste
print('Testes') 
Y = mlp.predict(X_primeira_sentenca_entrada)


# resultado 
print('Resultado procurado') 
print(Y_primeira_sentenca_saida)
print("Score de treino: %f" % mlp.score(X_primeira_sentenca_entrada, Y_primeira_sentenca_saida))


sumY = [sum(Y[i]) for i in range(np.shape(Y)[0])] # saida
sumT = [sum(Y_primeira_sentenca_saida[i]) for i in range(np.shape(Y_primeira_sentenca_saida)[0])] # target

print('Comparacao de resultados') 
print(np.logical_xor(sumY, sumT))


print('\n') 

# Testando a senten;a
print('Com a 1º Entrada a saida é:') 
Y = mlp.predict(X_teste)
print(Y)
print('\n') 

for i in range(2,7):
    print('Com a',i,'º Entrada a saida é:') 
    Y = mlp.predict(Y)
    print(Y)
    print('\n') 
