

str_label=''
tre_list =[]

c1=6
c2=10
c3=11

with open("data/inex05.train.elastic.tree", "r") as ins:
    line_tree = []
    for line in ins:
        line_tree.append(line)
ins.close()


#itero su ogni linea
for line in line_tree:

    str_label = ''

    #divido la classe dell'albero dalla rappresentazione dell'albero sui :
    line_div = line.split(':')
    
    with open("data/3_train", "a") as myfile:

        if(int(line_div[0]) == c1 or int(line_div[0]) == c2 or int(line_div[0]) == c3 ):
                myfile.write(line)


