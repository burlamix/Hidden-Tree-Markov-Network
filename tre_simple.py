import numpy as np

class Node(object):
    def __init__(self,label,root,level_n=0):
        self.children = []
        self.father = None
        self.root = root
        self.label = label
        self.name = "-"
        self.level_n = level_n
        self.pos = -1

    def __str__(self, level=0):
        image = "\t" * level + repr(self.name) +"-" +repr(self.label) + "\n"
        for child in self.children:
            image += child.__str__(level + 1)
        return image

    def __repr__(self):
        return str(self.name)+"-"+str(self.label)


    def add_node(self,label):
        node=Node(label,self.root,self.level_n+1)

        if len(self.root)<=node.level_n:
            self.root.append([])

        self.root[node.level_n].append(node)
        node.root=self.root
        self.children.append(node)
        node.father=self
        return node                                      

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

    def ch_l(self,name,l):  
        aux = self.get_node(name)
        if  aux != None and len(aux.children) > l:
            return aux.children[l].name
        else:
            return None

    def posizione(self):
        if  self != None:
            if self.father != None:
                return self.father.children.index(self)
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
        self.max_child=66
        self.leaves_n=None
        self.no_leaves_n=None
        self.N_L=np.zeros(self.max_child)
        self.N_I=None
        self.N_II=np.zeros(self.max_child)


    def __str__(self):
        return  str(self.classe)+"-" +self.t.__str__()
    def __repr__(self):
        return self.t.__repr__()


    def set_name(l_t):
        i=0
        for level in l_t.struct:
            for node in level:
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
            self.N_L[node.pos-1]=self.N_L[node.pos-1] + 1
        for level in self.struct[:-1]:
            for node in level:
                for child in node.children:
                    self.N_II[child.pos - 1] = self.N_II[child.pos - 1] + 1
