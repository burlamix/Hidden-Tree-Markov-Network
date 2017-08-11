
N_HIDDEN_STATES  = 3#C
MAX_CHILD = 6           #L
N_TREE = 10             #NT
N_SYMBOLS = 7          #M
N_NODE = 100            #N
I_NODE = 70

N_FOGLIE =3
TREE_DEEP = 8
MAX_NODE_LEVEL = 50
import numpy as np

class node:
    label = ""
    pos = (0,0)
    pa = 0
    ch_l =0

    def __init__(self,label,pos,pa,ch_l):
        self.label = label
        self.pos = pa
        self.ch_l = ch_l
def search_pa(mat,pa):
    find = False
    i = 0
    while( i < TREE_DEEP and  i > 0):
        j=0
        while (j < MAX_NODE_LEVEL and i > 0):
            if(mat[i,j].name == pa):
                #remove from leaves pa
                return (i,j)





class tree:
    mat = 0
    nome =""

    def __init__(self,TREE_DEEP,MAX_NODE_LEVEL):
        self.mat =  np.empty((MAX_NODE_LEVEL,TREE_DEEP),dtype=object)
        self.nome = "simo"

    def maketree(self,radice):
        self.mat[0,0]=node("radice", 0, 0)


    def add_node(self,label,name,pa):




n = node("simo", 1, 2, 3)

ste = tree()

ste.mat[0,1]= n

print(ste.mat[0,1].label)
