import numpy as np
from treelib import Node, Tree

#dato il nodo u, restituisce la sua posizione di figlio rispetto al padre
def pos(u):
    return [0]
#dato il nodo u, restituisce il padre
def pa(u):
    return [0]
#dato il nodo u e l, restituisce l-esimo figlio di u
def ch_l(u,l):
    return [0]

N_HIDDEN_STATES  = 10   #C
MAX_CHILD = 10          #L
N_TREE = 10             #NT
N_SYMBOLS = 10          #M
N_NODE = 100            #N
I_NODE = 70

#per ora la struttura dell'albero è ignorata e viene considerato un array
tree_test = np.zeros(N_NODE)

#model parameters
#positional prior probability matrix --- pg
pos_prior_p = np.zeros((N_HIDDEN_STATES,MAX_CHILD,N_NODE))
#SP probability matrix --- fi
sp_p = np.zeros((MAX_CHILD,N_NODE))
#positional state transiction probability matrix --- A
pos_st_tr_p = np.zeros((N_HIDDEN_STATES,N_HIDDEN_STATES,MAX_CHILD,N_NODE))
#multinomial emision --- bi
m_emission = np.zeros((N_HIDDEN_STATES,N_SYMBOLS,N_NODE))

#upward parameters beta
up_ward = np.zeros((N_NODE,N_HIDDEN_STATES))
a_up_ward = np.zeros((MAX_CHILD,N_HIDDEN_STATES,N_NODE))

#stater posterior €
post = np.zeros((N_NODE,N_HIDDEN_STATES))
#pairwwise smoothed posterior
s_post = np.zeros((MAX_CHILD,N_HIDDEN_STATES,N_HIDDEN_STATES,N_NODE))


for u in range(I_NODE,N_NODE):
    for i in range(0, N_HIDDEN_STATES):
        nu=0
        for j in range(0, N_HIDDEN_STATES):
            nu = nu + (m_emission[j,tree_test[u],u] * pos_prior_p[j,pos(u),u])

        up_ward[u,i] = ((m_emission[i,tree_test[u],u] * pos_prior_p[i,pos(u),u]) / nu)  #(17)

for u in range(I_NODE,1,-1):
    for i in range (0,N_HIDDEN_STATES):

        for l in range(0,MAX_CHILD):
            for j in range(0,N_HIDDEN_STATES):
                pos_prior_p[i,pos(u),u] =  sp_p[l,u] * pos_st_tr_p[i,j,l,u] * pos_prior_p[j , pos(ch_l(u,l)) ,ch_l(u,l)] # plausibile errore (20)

        d=0
        for l in range(0,MAX_CHILD):
            for j in range(0,N_HIDDEN_STATES):
                d = d +  (pos_st_tr_p[i,j,l,u] * up_ward[ch_l(u,l),j])

            a_up_ward[l,i,u] = d / pos_prior_p[i,pos(u),u]                  #(21) incertezza sul P(Qu = i) come prima

        aux1=0
        for l in range(0,MAX_CHILD):
            aux1 = aux1 + (sp_p(l,u) * a_up_ward[l,i,u] * pos_prior_p[i,pos(u),u] )

        aux2=0
        for j in range(0,N_HIDDEN_STATES):
            aux3=0
            for l in range(0,MAX_CHILD):
                aux3= aux3 +  (m_emission[j,tree_test[u],u] * a_up_ward[l,j,u] * pos_prior_p[j,pos(u),u] )

            aux2=aux2+aux3

        up_ward[u,i] = ( m_emission[i,tree_test[u],u] * aux1 ) / aux2   #(19)

# downward

#base case
for i in range(0,N_HIDDEN_STATES):
    post[1,i] = up_ward[1,i]


for u in range(2,N_NODE):
    for i in range(0,N_HIDDEN_STATES):
        for j in range(0,N_HIDDEN_STATES):

            aux1=0
            for l in range(0,MAX_CHILD):
                aux1 = aux1 + ( sp_p[l,pa(u) ] *  a_up_ward[l,i,pa(u)] )  # incertezza su fi di quale nodo si sta parlando sp_p[l,u ] o sp_p[l,ps(u) ]
            den = pos_prior_p[i,pos(pa(u)),pa(u)] * aux1

            s_post[pos(u),i,j,pa(u)] = ( post[i,pa(u)]*up_ward[u,j] * sp_p[pos(u),u] * pos_st_tr_p[i,j,pos(pa(u)),pa(u)] ) / den  #24

            post[u,i]=0
            for ii in range(1,N_HIDDEN_STATES):
                post[u,i] = post[u,i] + s_post[pos(u),ii,i,pa(u)]




