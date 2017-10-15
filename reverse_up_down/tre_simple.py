import numpy as np
#dato il nodo u, restituisce la sua posizione di figlio rispetto al padre
def pos(u):
    return [0]
#dato il nodo u, restituisce il padre
def pa(u):
    return [0]
#dato il nodo u e l, restituisce l-esimo figlio di u
def ch_l(u,l):
    return [0]



class Node(object):
    def __init__(self,label,root,level_n=0):
        self.children = []
        self.father = None
        self.root = root
        self.label = label
        self.name = "-"
        self.level_n = level_n
        self.pos = -1



#    def __str__(self):
 #       return str(self.label)+"_"+str(self.name)
    def __str__(self, level=0):
        image = "\t" * level + repr(self.name) +"-" +repr(self.label) + "\n"
        for child in self.children:
            image += child.__str__(level + 1)
        return image

    def __repr__(self):
        return str(self.name)+"-"+str(self.label)

   # def __str__(self):
  #      return str(self.name)
 #   def __repr__(self):
#        return str(self.name)


    def add_node(self,label):
        node=Node(label,self.root,self.level_n+1)

        if len(self.root)<=node.level_n:
            self.root.append([])

        self.root[node.level_n].append(node)
        node.root=self.root
        self.children.append(node)
        node.father=self
        return node                                         #à---------------------------------------------test

    def put_name(self):
        stack = [self]
        i=0
        while stack:
            cur_node = stack[0]
            stack = stack[1:]
            cur_node.name = i
            i += 1
            for child in cur_node.children:
                stack.append(child)

    def get_node(self,name):
        stack = [self]
        while stack:
            cur_node = stack[0]
            if cur_node.name == name:
                return cur_node
            stack = stack[1:]
            for child in cur_node.children:
                stack.append(child)

        return None

    def get_label(self,name):
        aux = self.get_node(name)
        if  aux != None:
            return aux.label
        else:
            return None

    def pa(self,name):
        aux = self.get_node(name)
        if  aux != None:
            return aux.father
        else:
            return None

    def ch_l(self,name,l):  #occhio le posizioni iniziano da 0
        aux = self.get_node(name)
        if  aux != None and len(aux.children) > l:
            return aux.children[l].name
        else:
            return None

    def posizione(self):
        if  self != None:
            if self.father != None:
                return self.father.children.index(self) #inefficente va migliorata e partono da zero!
            else:
                return 0
        else:
            return None

    def get_reverse_children(self):
        children = self.children[:]
        children.reverse()
        return children

    def make_linear_tree(self,max_child,max_level,max_label):

        if(max_level>0):
            for i in range(0,max_child):
                #self.add_node(randint(0,max_label-1))

                self.add_node(((i*7)+3)%max_label)

                self.children[i].make_linear_tree(max_child,max_level-1,max_label)

class Tree(object):
    def __init__(self,label,classe,level_n=0):
        self.struct = []
        self.struct.append([])
        self.t = Node(label,self.struct)
        self.struct[0].append(self.t)
        self.size=None
        self.classe=classe
        self.max_child=32
        self.leaves_n=None
        self.no_leaves_n=None
        self.N_L=np.zeros(self.max_child)
        self.N_I=None
        self.N_II=np.zeros(self.max_child)


    def __str__(self):
        return  str(self.classe)+"-" +self.t.__str__()
    def __repr__(self):
        return self.t.__repr__()

#    def __str__(self):
 #       return str(self.name)
  #  def __repr__(self):
   #     return str(self.name)


    def set_name(l_t):
        i=0
        for level in l_t.struct:
            for node in level:
                #inoltre mi salvo il numero massimo di figli che ha l'albero così da agevolare i futuri calcoli
                #if len(node.children)> l_t.max_child:
                #   l_t.max_child=len(node.children)
                node.name=i
                if node.father != None:
                    node.pos = node.father.children.index(node)+1
                else:
                    node.pos = 0
                i=i+1
        l_t.size=i
        l_t.N_I=l_t.size -len(l_t.struct[-1])

    def divide_leaves(self):
        self.struct.append([])
        stack = [self.t]
        while stack:
            cur_node=stack[0]
            stack = stack[1:]
            if not cur_node.children:
                self.struct[-1].append(cur_node)
                self.struct[cur_node.level_n].remove(cur_node)
            else:
                for child in cur_node.get_reverse_children():
                    stack.insert(0, child)
        del self.struct[-2]

    def set_N_L(self):
        for node in self.struct[-1]:
            #self.N_L[node.father.children.index(node)]=self.N_L[node.father.children.index(node)] + 1
            self.N_L[node.pos-1]=self.N_L[node.pos-1] + 1
        for level in self.struct[:-1]:
            for node in level:
                for child in node.children:
                    self.N_II[child.pos - 1] = self.N_II[child.pos - 1] + 1




