str_label=''
tre_list =[]

c1=1
c2=2
c3=8

with open("data/inex05.test.elastic.tree", "r") as ins:
    line_tree = []
    for line in ins:
        line_tree.append(line)
ins.close()


#itero su ogni linea
for line in line_tree:

    str_label = ''

    #divido la classe dell'albero dalla rappresentazione dell'albero sui :
    line_div = line.split(':')
    
    with open("data/3f_test", "a") as myfile:

        if(int(line_div[0]) == c1 or int(line_div[0]) == c2 or int(line_div[0]) == c3 ):
                myfile.write(line)


