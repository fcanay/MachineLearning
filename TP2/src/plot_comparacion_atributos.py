import numpy as np
import matplotlib.pyplot as plt


def main():
	x = range(1,10)
	y = [[0.618,0.582,0.576,0.56,0.56,0.586,0.594,0.552,0],[0.508,0.518,0.584,0.584,0.572,0.558,0.556,0.606,0.552],[48, 143, 16, 63, 8, 25, 22, 18, 7],[0.1875, 0.18619791666666666, 1.0, 0.123046875, 1.0, 0.4166666666666667, 0.36666666666666664, 0.3, 0.875],[0.12430044676328877, 0.39838284003302066, 0.014873884486647504, 0.35875888461242939, 0.0049779276480328895, 0.028127599418183897, 0.03105365990931936, 0.034551961531363543, 0.0049727955977149057],[0.00048554862016909677, 0.00051872765629299561, 0.000929617780415469, 0.00070070094650865114, 0.00062224095600411119, 0.00046879332363639829, 0.00051756099848865604, 0.00057586602552272572, 0.00062159944971436321]]
	titles = ["Sacando un atributo","Entrenando con solo un atributo","Cantidad de variables seleccionas en el 20 percentil univariado","Porcentaje de variables seleccionas en el 20 percentil univariado","Importancia porcentual acumulada de la seleccion multivariada","Importancia porcentual promedio por variable de la seleccion multivariada"]
	file_name = ['comparacion_atributos_sacando_uno','comparacion_atributos_de_a_uno','univar_cant_var','univar_porcentaj_var','multivar_porcentaj_acum','multivar_porcentaj_prom']
	labels=['histByN','histColor','pat 2x2','pat 3x3','Circ 2 5','Circ 2 9','Circ 3 9','Circ 5 9','Circ 3 5']
	for i in xrange(0,len(y)):
		fig, ax = plt.subplots()
		plt.title(titles[i])
		plt.ylabel("Score")
		ax.bar(x,y[i],align='center')
		#ax.set_ylim([0.50,0.75])
		ax.set_xticks(x)
		ax.set_xticklabels(labels,rotation=30)
		ax.legend()
		#plt.show()
		#return
		file_path = "../imagenes/"+file_name[i]

		plt.savefig(file_path, dpi=None, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format=None,transparent=False, bbox_inches=None, pad_inches=0.1,frameon=None)
		plt.close()

main()