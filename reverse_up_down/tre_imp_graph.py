
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


lll=0

class Tree(object):
    "Generic tree node."
    global ll

    def __init__(self, name='root', children=None):


        self.name = name
        self.children = []
        self.level =[]
        self.l=ll

        if children is not None:
          #  self.level[0].append(children)
            for i in range(0,len(children)):
                self.add_child(children[i])

    def __repr__(self):
        return self.name

    def add_child(self, node):
        assert isinstance(node, Tree)
        print(node.name,self.l)
        self.children.append(node)
        self.children[len( self.children)-1].l=self.l+1
ll=0
t = Tree('*', [Tree('++', [Tree('33'),
                          Tree('44')]),
               Tree('2'),
               Tree('+', [Tree('3'),
                          Tree('4')])])
print(t.children[0].children)