from random import randint

#dato il nodo u, restituisce la sua posizione di figlio rispetto al padre
def pos(u):
    return [0]
#dato il nodo u, restituisce il padre
def pa(u):
    return [0]
#dato il nodo u e l, restituisce l-esimo figlio di u
def ch_l(u,l):
    return [0]



class Tree(object):
    def __init__(self,label,root,level_n=0):
        self.children = []
        self.father = None
        self.root = root
        self.label = label
        self.name = ""
        self.level_n = level_n

#    def __str__(self):
 #       return str(self.label)+"_"+str(self.name)
  #  def __repr__(self):
   #     return str(self.label)+"_"+str(self.name)

    def __str__(self):
        return str(self.name)
    def __repr__(self):
        return str(self.name)


    def add_node(self,label):
        node=Tree(label,self.root,self.level_n+1)

        if len(self.root)<=node.level_n:
            self.root.append([])

        self.root[node.level_n].append(node)
        node.root=self.root
        self.children.append(node)
        node.father=self

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

    def pos(self,name):
        aux = self.get_node(name)
        if  aux != None:
            if aux.father != None:
                return aux.father.children.index(aux) #inefficente va migliorata e partono da zero!
            else:
                return -1
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

class List_tree(object):
    def __init__(self,label,level_n=0):
        self.struct = []
        self.struct.append([])
        self.t = Tree(label,self.struct)
        self.struct[0].append(self.t)
        self.size=None

   # def __str__(self):
  #      return self.t.label+"_"+str(self.name)
 #   def __repr__(self):
#        return self.t.label+"_"+str(self.name)

    def __str__(self):
        return str(self.name)
    def __repr__(self):
        return str(self.name)


    def set_name(l_t):
        i=0
        for level in l_t.struct:
            for node in level:
                node.name=i
                i = i+ 1
        l_t.size=i

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


"""t = List_tree("radice")
t.t.add_node("1")
t.t.add_node("2")
t.t.add_node("3")
t.t.children[0].add_node("11")
t.t.children[2].add_node("31")
t.t.children[2].add_node("32")
t.t.children[2].add_node("33")
t.t.children[2].children[0].add_node("311")
t.t.add_node("4")
t.t.put_name()

t2 = List_tree(0)
t2.t.make_linear_tree(3,4,9)
t2.divide_leaves()
t2.set_name()
print(t2.struct)

print(t2.t.get_label(1))  """


