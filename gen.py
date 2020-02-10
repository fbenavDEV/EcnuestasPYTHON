import numpy as np
import matplotlib.pyplot as plt
import math

""" Crea una poblacion de N individuos cuyas preferencias se distribuyen
    entre un numero de candidatos igual a la longitud del vector P con los
    pesos indicados en ese vector. 
"""
def pob(N , P):
    n_cands = len(P)       
    P = P/sum(P)
    pop = np.zeros((N,n_cands))
    limits = np.zeros(n_cands+1)
    for k in range(1,n_cands+1):
        limits[k] = limits[k-1]+P[k-1]
    choices = np.random.uniform(0,1,N) 
    for k in range(n_cands):
        pop[:,k] = (choices < limits[k+1])*(choices >= limits[k]);
    return (pop)

""" Toma una subpoblacion del conjunto P, del tamano indicado
    por n_sample. La subpoblacion de toma de manera uniforme y equivale
    a la accion de realizar una encuesta con el numero de individuos indicado
    por n_sample
"""
def subpob(P,n_sample):
    sel = np.random.uniform(0,1,n_sample)
    tot_cands = P.shape[1];
    R = np.zeros((n_sample,tot_cands));
    for k in range(n_sample):
        idx = int(math.floor(sel[k]*(P.shape[0])))
        R[k] = P[idx]
    return(R)

""" Realiza encuestas con sample_size entrevistas de manera iterada,
    donde n_samples es el numer de iteraciones. La variable P representa la 
    poblacion. Esto permite obtener una distribucion de resultados asociados
    con el tamano de una ciert poblacion de la encuesta
"""
def subpob_s(P, n_samples, sample_size):
    n_cands = P.shape[1]
    R = np.zeros((n_samples,n_cands))
    for k in range(n_samples):
        S = subpob(P,sample_size)
        for s in range(n_cands):
            R[k][s] = round((sum(S[:,s])/S.shape[0])*100)        
    return(R)

""" Produce un vector de respuestas que puede afectar de manera diferente
    a cada candidato. P es necesario unicamente para obtener el tamano de 
    la poblacion, pues prob_r es del mismo tamano que el numero de candidatos
    y contiene el porcentaje de personas que responden la encuesta para cada 
    candidato. Esto permite crear desequilibrios en la tasa de respuesta
    a las encuestas para cada candidato, permitiendo esconder el resultado
    real
"""
def resp(P,prob_r):
    n_cands = P.shape[1]
    N = P.shape[0]
    if (len(prob_r) == n_cands):
        R = np.zeros((N,1))
        rul = np.random.uniform(0,1,N)
        for k in range(N):
            prob = 0;
            for t in range(n_cands):
                prob = prob + prob_r[t]*P[k][t]
            if rul[k] < prob:
                R[k] = 1
    return(R)

""" Coloca una fila de ceros a los individuos que no responden la encuesta.
    Esto permite crear una nueva poblacion en la que las deciciones de algunos
    individuos estan escondidas (con respecto a P) de acuerdo a las probabilidades
    indicadas en prob_r
"""
def esc(P,prob_r):
    R = resp(P,prob_r)
    PM = np.copy(P)
    N = P.shape[0]
    for k in range(N):
        PM[k] = P[k]*R[k]
    return(PM)

""" Dado el vector X, con valores continuos, esta funcion ajusta el valor de
    r_max para que todos los valores de X contenidos entre r_min y r_max tengan
    un peso porcentual igual al contenido en el vector peso. Esto se repite para
    todos los subintervalos definidos en el vector peso. 
"""
def limite_max(X,r_min,r_max,peso):
    err = 0.01
    ajuste = 0.25
    a_max = r_max #Valor inicial del limite superior del intervalo de la categoria
    c_peso = sum((X >= r_min)*(X < r_max))/len(X)   #Este es el peso de la
                                                    #categoria en el intervalo
                                                    #[r_min, a_max=r_max]
    iter = 0 #Contador de iteraciones
    signo_pos = True
    while ((abs(c_peso-peso) > err)and(iter < 50)): #Entra si el peso esta desajustado
        if (c_peso < peso):
            a_max = a_max + ajuste*(a_max-r_min) #Ajusta el limite superior hacia arriba 
            if (signo_pos == False):                
                ajuste = 0.5*ajuste
            signo_pos = True
        else:
            a_max = a_max - ajuste*(a_max-r_min) #Ajusta el limite superior hacia abajo
            if (signo_pos == True):                
                ajuste = 0.5*ajuste
            signo_pos = False
        c_peso = sum((X >= r_min)*(X < a_max))/len(X) #Recalcula el peso       
        iter = iter + 1
    return(a_max)
                
""" Retorna una variable que se correlaciona con V en un valor dado por
    corr_val y con una distribucion de m categorias discretas cuyos pesos
    son aproximadamente los mismos que se dan en el parametro pesos. 
"""
def correlac(V,corr_val,pesos):
    N = len(V)
    U = np.copy(V) #Se copia la variable base V en otra nueva, para normalizar
    U = U - U.mean()
    U = U/U.std() #Se normaliza U
    X = math.sqrt(12)*np.random.uniform(-0.5,0.5,N) 
    #Variable uniforme X, media 0 y desviacion 1
    
    cu = corr_val;
    cx = math.sqrt(1-corr_val*corr_val)
    #cu y cx son los coeficientes que producen una nueva variable correlacionada con U
    
    X = cu*U+cx*X
    X = X - X.min()
    X = X/X.max()
    #La variable producida, X se normaliza
    
    pesos = pesos/sum(pesos)
    ajuste = np.zeros(len(pesos)+1)
    ajuste[0] = 0
    c_cat = 0
    R = np.zeros(N)
    # Se procede a discretizar X de acuerdo con los pesos. 
    
    for k in range(len(pesos)):
        ajuste[k+1] = ajuste[k]+pesos[k] 
        #Inicialmente los pesos se distribuyen uniformemente       
        if ((ajuste[k+1] > 1)or(k == len(pesos)-1)):
            ajuste[k+1]=1
        else:            
            ajuste[k+1] = limite_max(X,ajuste[k],ajuste[k+1],pesos[k])  
            #Aqui se ajusta el limite superior de la categoria para que su
            #presencia dentro de X sea la misma que se da en el peso
        R = R + (X>=ajuste[k])*(X < ajuste[k+1])*c_cat
        #Se ha discretizado la variable
        c_cat = c_cat + 1
    R = R + 1  #Las categorias comienzan en 1      
    R = np.round(R)
    return(R)
    
""" Agrega la columna V a la poblacion P
    Equivale a ejecutar el comando [P V]  
"""
def agg_var(P,V):
    N = P.shape[0]
    n = P.shape[1]
    NP = np.zeros((N,n+1))
    for k in range(N):
        for j in range(n):
            NP[k][j] = P[k][j]
        NP[k][n]=V[k]
    return(NP)

""" Produce una re-estimacion ponderada de la decision de voto de cada individuo 
    de S, usando como base la distribucion de V que se compara con la distribucion
    de la misma variable en toda la poblacion (dada en el vector pesos).
    Se retorna una matriz del mismo tamano de S pero con los pesos ponderados (los valores
    no son necesariamente 1 o 0, sino que pueden tomar valores fuera del intervalo [0,1]) 
"""
def repes(pesos,S,V):
    n_cats = len(pesos) #Se obtiene el numero de categorias
    R = np.zeros(S.shape) #R contendra los valores modificados de S
    UM = np.zeros(V.shape) #Variable copia de la variable de control
    m_pesos = np.zeros(pesos.shape)
    for j in range(V.shape[0]):
        UM[j] = V[j]*sum(S[j])  #Se invisibilizan los valores de la variable de 
                                #control que tienen filas de ceros en S
    for k in range(n_cats):
        w = sum(UM==(k+1))/sum(UM!=0)   #Distribucion de la categoria
                                        #entre los que no estan indecisos
        p = pesos[k]
        if (w == 0):
            m_pesos[k]=1
        else:                
            m_pesos[k] = p/w #Se calcula el peso de la variable subrepresentada
                             #o sobrerepresentada
    for k in range(S.shape[0]):
        R[k] = S[k]
        j = int(V[k])
        R[k] = R[k]*m_pesos[j-1] #Crea una nueva matriz de encuestas con pesos 
                                 #modificados
    return(R)

"""*****************************************************************************"""
        
""" Codigo para producir los experimentos y las figuras del articulo
"""
PT = pob(3000000,np.array([6, 4]))  #Poblacion de 3 millones de habitanes
                                    #60% de ellos apoyan al candidato 1 y 
                                    #el resto al candidato 2

P = esc(PT,np.array([0.65,0.9])) #Esconde la decicion del 35%  de los que apoyan
                                 #al candidato 1 y el 10% de los que apoyan al 2

cats = np.array([2,2,2,2,1,1])  #Categorias de una variable de control. Son 6 distribuidas
                                #con los pesos indicados

VC = correlac(PT[:,0],0.6,cats) #Construccion de la variable de control
P = agg_var(P,VC) #Agregacion de la variable de control a la poblacion con indecisos
PT = agg_var(PT,VC) #Agregacion de la variable de control a la poblacion total (sin indecisos)

ps = np.zeros(cats.shape) 
for k in range(cats.shape[0]):
    ps[k] = sum(PT[:,2]==(k+1))/len(PT[:,0])
#Lo anterior es para recalcular las frecuencias de las categorias de VC, asumiendo
#solo 2 candidatos

n_samp = 100; #Se van a hacer 100 encuestas
R = np.zeros((100,PT.shape[1]-1)) #Para guardar los resultados de las encuestas redondeados
CC = np.zeros(100) #Para guardar correlaciones

for k in range(100):
    S = subpob(P,1000) #Subpoblacion de 1000 individuos       
    SW = repes(ps,S[:,range(2)],S[:,range(2,3)]) #Reajuste de acuerdo a los que contestaron
    ST = sum(SW[:,0]) +sum(SW[:,1]) #Numero total de individuos que contestaron
    for s in range(PT.shape[1]-1):
        R[k][s] = round((sum(SW[:,s])/ST)*100) #Porcentaje de apoyo a cada candidato       
    CC[k] = np.corrcoef(S[:,0],S[:,2])[0][1] #Calculo de correlacion
    
"""Codigo para plotear resultados
"""

plt.grid()
plt.xlabel("Índice de experimento")
plt.ylabel("Correlación entre el control y el apoyo al candidato 1")
plt.title("Valores de correlación detectables")
plt.plot(range(100),CC)
"""plt.grid()
plt.hist(R[:,1],range(100))
plt.hist(R[:,0],range(100))
"plt.hist(100-V[:,0]-V[:,1],range(100))"
plt.xlabel("Porcentaje de apoyo al candidato, con variable de control")
plt.ylabel("Número de encuestas")
plt.title("Resultados de 100 encuestas con 1000 entrevistas")
plt.savefig("Histogram.png")"""

  
    
    
        

