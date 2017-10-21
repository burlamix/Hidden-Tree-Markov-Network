import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


#FILE = "inex05.train.elastic.tree"
FILE = "test_500.tree"

from tre_simple import *

def dataset_parser():

    str_label=''
    tre_list =[]
    with open(FILE, "r") as ins:
        line_tree = []
        for line in ins:
            line_tree.append(line)
    ins.close()


    #itero su ogni linea
    for line in line_tree:

        str_label = ''

        #divido la classe dell'albero dalla rappresentazione dell'albero sui :
        line_div = line.split(':')

        #prelevo il label del primo nodo cosi da inizializzare l'albero
        my_iter = iter(line_div[1])
        s = next(my_iter)
        while s != '(':
            str_label = str_label + s
            s = next(my_iter)
        line_div[1] = line_div[1][len(str_label)+1:]

        # inizzializzo l'albero, con la classe e il nodo radice
        lt = Tree(int(str_label),line_div[0])

        nodo_in_esame = lt.t
        str_label = ''

        #ciclo su ogni nodo
        for s in line_div[1]:

            if s in ["1", "2", "3", "4", "5", "6", "7", "8","9","0"]:
                str_label = str_label + s
            elif s == '(':
                in_label=False
                nodo_in_esame = nodo_in_esame.add_node(int(str_label))
                str_label = ""
            elif s == ')':
                nodo_in_esame = nodo_in_esame.father

        lt.divide_leaves()
        lt.set_name()
        lt.set_N_L()
        tre_list.append(lt)

    return tre_list

