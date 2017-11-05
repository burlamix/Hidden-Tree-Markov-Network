
bs = 4
for j in range(0,15):

	print("--",j)

	print("---------",j%bs)

	if(j%bs == bs-1):
		print("			gradiente!",j)