import numpy as np
import matplotlib.pyplot as plt


def main():
	x = range(1,9)
	y = [0.618,0.582,0.576,0.56,0.56,0.586,0.594,0.552]
	labels=['histogramaByN','histogramaColor','patrones2x2','patrones3x3','patronesCirculaes_2_5','patronesCirculaes_2_9','patronesCirculaes_3_9','patronesCirculaes_5_9']
	fig, ax = plt.subplots()
	plt.title("Comparacion Atributos")
	plt.ylabel("Score")
	ax.bar(x,y,align='center')
	ax.set_ylim([0.50,0.75])
	ax.set_xticks(x)
	ax.set_xticklabels(labels,rotation=30)
	ax.legend()
	#plt.show()
	#return
	file_path = "../imagenes/comparacion_atributos_sacando_uno"

	plt.savefig(file_path, dpi=None, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format=None,transparent=False, bbox_inches=None, pad_inches=0.1,frameon=None)
	plt.close()

main()