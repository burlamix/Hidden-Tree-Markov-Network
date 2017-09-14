import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from tre_simple import *


with open("inex05.train.elastic.tree", "r") as ins:
    line_tree = []
    for line in ins:
        line_tree.append(line)
ins.close()

str_label=''


#itero su ogni linea
for line in line_tree:

    str_label = ""

    #divido la classe dell'albero dall'albero sui :
    line_div = line.split(':')

    #prelevo il label del primo nodo così da inizializzare l'albero
    my_iter = iter(line_div[1])
    s = next(my_iter)

    while s != '(':
        str_label = str_label + s
        s = next(my_iter)

    line_div[1] = line_div[1][len(str_label)+1:]

    # inizzializzo l'albero
    lt = List_tree(str_label,line_div[0])

    nodo_in_esame = lt.t
    int_label = 0
    str_label = ''



    for s in line_div[1]:

        if s in ["1", "2", "3", "4", "5", "6", "7", "8","9","0"]:
            str_label = str_label + s
        elif s == '(':
            in_label=False
            int_label = int(str_label)
            nodo_in_esame = nodo_in_esame.add_node(int_label)
            str_label = ""
        elif s == ')':
            nodo_in_esame = nodo_in_esame.father


    lt.divide_leaves()
    lt.set_name()
    print(lt)