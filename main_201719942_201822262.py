#Pamela Ramírez González #Código: 201822262
#Manuel Gallegos Bustamante #Código: 201719942
#Análisis y procesamiento de imágenes: Proyecto3 Entrega2
#Se importan librerías que se utilizarán para el desarrollo del laboratorio
import random
import numpy as np
from skimage.filters import threshold_otsu
import skimage.io as io
import skimage.morphology as morfo
import skimage.segmentation as segmen
from scipy.ndimage import binary_dilation
import sklearn.metrics as skmetr
import matplotlib.pyplot as plt
import os
import cv2
import glob
import pandas
##input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
img = glob.glob(os.path.join('data_mp3','blood_cell_dataset', 'noisy_data', '*.png')) # se importan las imágenes
preprocesadas=[] # lista para almacenar imágenes
for i in img:
    carga_color_image=io.imread(i)
    imagen_en_Lab = cv2.cvtColor(carga_color_image, cv2.COLOR_BGR2LAB)  # se convierte de rgb a Lab con librería cv2
    filtrado= cv2.medianBlur(imagen_en_Lab,7)
    espacio_L = filtrado[:, :, 0]  # se extraen canales de la imagen en el espacio de color La*b
    espacio_a = filtrado[:, :, 1]
    espacio_b = filtrado[:, :, 2]
    umbral = threshold_otsu(espacio_b) #cálculo del umbral de otsu y binarización de la imagen
    binarizada = espacio_b < umbral
    preprocesadas.append(binarizada) #se añade nueva imagen preprocesada a arreglo
##input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
prueba_star=io.imread(os.path.join('data_mp3', 'star_binary.png')) # se importa la imagen de prueba
def ImgComplemento(img): # funcion para calcular el complemento
    complemento = img.copy() # se copia la imagen ingresada por parametro
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j] == 0:
                complemento[i][j] = 1 # se intercambian los valores de 1 con 0
            else:
                complemento[i][j] = 0 # se intercambian los valores de 0 con 1
    return complemento
def MyHoleFiller_201719942_201822262(bin_img):
    complemento = ImgComplemento(bin_img) # se saca el complemento de la imagen ingresada por parámetro
    conbordecomplemento = np.zeros((len(complemento), len(complemento[0]))) # se crea matriz de ceros
    for i in range(len(bin_img)):
        for j in range(len(bin_img[0])):
            if i == 0 or i == len(complemento) - 1 or j == 0 or j == len(complemento[0]) - 1:
                conbordecomplemento[i][j] = complemento[i][j] # se llenan los bordes de la matriz con el complemento de la imegen original
    marcador_inicialF = conbordecomplemento # variable para el marcador
    mask = complemento # variable para la máscara
    elemt_struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]) # elemento estructurante seleccionado arbitrariamente
    marcador = marcador_inicialF
    fin = False
    while fin == False:
        previo = np.copy(marcador) # se copia el marcador anterior
        marcador = binary_dilation(marcador, elemt_struct).astype(marcador.dtype) # se realiza una dilatación del marcador con el elemento estructurante
        marcador = marcador + mask # se hace una suma de las matrices para revisar la intersección
        for i in range(len(marcador)):
            for j in range(len(marcador[0])):
                if marcador[i][j] > 1: # dónde haya interseccion se preserva con un 1
                    marcador[i][j] = 1
                else:
                    marcador[i][j] = 0 # dónde no hay intersección se convierte en fondo con un 0
        if np.array_equal(previo, marcador):
            fin = True # condición de parada: si el marcador anterior es igual al actual
    whole_img=ImgComplemento(marcador)
    return whole_img # se retorna el resultado
##input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
#cálculo de índices de Jaccard y hematocritos
def hematocrito(img): # función para calculra hematocrito
    red_cells=np.count_nonzero(img == 1) # se cuentarn los pixeles de celula de la imagen
    total_pixels=img.shape[0]
    porcentaje_hematocrito=red_cells/total_pixels # aplicacion de la fórmula propuesta en la guía
    return porcentaje_hematocrito
img_anot = glob.glob(os.path.join('data_mp3','blood_cell_dataset', 'groundtruth', '*.png')) # lectura de imagenes
anota,errores=[],{}
jaccards_huecos,jaccards_llenas={},{}
hematocrito_anota,hematocrito_llenas={},{} # diccionario para guardar las métricas y hematocritos como valores de las llaves que indican el número de la imagen de esta métrica
for i in img_anot:
    carga=io.imread(i)>0
    anota.append(carga) # se realiza la carga y umbralización de imagenes
for i in range(len(preprocesadas)):
    llena=MyHoleFiller_201719942_201822262(preprocesadas[i]).flatten() # se rellenan los huecos
    i_dict,i_anota=i,i #indices correspondientes al número de la imagen
    if i == 1: # condicionales para excepciones en el orden correspondiente
        i_dict=10
        i_anota=0
    elif i==0:
        i_dict = i+1
        i_anota=1
    jaccards_huecos[i_dict] = skmetr.jaccard_score(anota[i_anota].flatten(), preprocesadas[i].flatten()) #cálculo de métricas y asignación de valores en los diccionarios haciendo uso de las anotaciones e imagenes correspondientes
    jaccards_llenas[i_dict] = skmetr.jaccard_score(anota[i_anota].flatten(), llena)
    hematocrito_llenas[i_dict] = hematocrito(llena)
    hematocrito_anota[i_dict] = hematocrito(anota[i_anota].flatten())
    errores[i_dict]=(hematocrito_anota[i_dict]-hematocrito_llenas[i_dict])**2
# se muestran las métricas
valores_llenos=np.array(list(jaccards_llenas.values()))
valores_huecos=np.array(list(jaccards_huecos.values()))
print("\nÍndices de Jaccard imágenes con huecos:\n",jaccards_huecos)
print("prom huecos",np.mean(valores_huecos),"desv.est",np.std(valores_huecos))
print("\nÍndices de Jaccard imágenes sin huecos:\n",jaccards_llenas)
print("prom llenos",np.mean(valores_llenos),"desv.est",np.std(valores_llenos))
print("\n% Hematocrito predicciones sin huecos:\n",hematocrito_llenas)
print("\n% Hematocrito anotaciones:\n",hematocrito_anota)
print("\nErrores cuadráticos:\n",errores)
print("\nMSE:\n",np.mean(np.array(list(errores.values()))))
##input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
def interseccion(cor_x1_anotacion, cor_y1_anotacion, cor_x1_prediccion, cor_y1_prediccion, dim_anot, dim_pred): # funcion para calcularla intersección dadas las cordenadas y las dimensiones de dos cuadros
    # caso 1: en el que hay un cuadrado en la esquina superior izquierda
    def calculo1(izq_x, izq_y, der_x, der_y): #calculo de área compartida entre ambos cuadrados
        return (izq_x + dim_anot - der_x) * (izq_y + dim_pred - der_y)
    # caso 2: en el que hay un cuadrado en la esquina superior derecha
    def calculo2(izq_x, izq_y, der_x, der_y):#calculo de área compartida entre ambos cuadrados
        return (izq_x + dim_anot - der_x) * (der_y + dim_pred - izq_y)
    # variables para almacenan coordenadas de los cuadrados
    c_1 = np.array([cor_x1_anotacion,cor_y1_anotacion])
    c_2 = np.array([cor_x1_prediccion, cor_y1_prediccion])
    # caso en el que no hay intersección
    if (c_1[0] + dim_anot < c_2[0] and c_1[1] + dim_anot < c_2[1]) or (c_2[0] + dim_pred < c_1[0] and c_2[1] + dim_pred < c_1[1]):
        return 0
    # cálculos del caso 1
    if min(cor_x1_anotacion, cor_x1_prediccion) == c_1[0] and min(cor_y1_anotacion, cor_y1_prediccion) == c_1[1]:
        return calculo1(c_1[0], c_1[1], c_2[0], c_2[1]) # la anotación es el cuadro de la esquina superior izquierda
    elif min(cor_x1_anotacion, cor_x1_prediccion) == c_2[0] and min(cor_y1_anotacion, cor_y1_prediccion) == c_2[1]:
        return calculo1(c_2[0], c_2[1], c_1[0], c_1[1]) # la predicción es el cuadro de la esquina superior izquierda
    # cálculos del caso 2
    if max(cor_x1_anotacion, cor_x1_prediccion) == c_1[0] and min(cor_y1_anotacion, cor_y1_prediccion) == c_1[1]:
        return calculo2(c_2[0], c_2[1], c_1[0], c_1[1]) # la anotación es el cuadro de la esquina superior derecha
    elif max(cor_x1_anotacion, cor_x1_prediccion) == c_2[0] and min(cor_y1_anotacion, cor_y1_prediccion) == c_2[1]:
        return calculo2(c_1[0], c_1[1], c_2[0], c_2[1]) # la predicción es el cuadro de la esquina superior derecha
# Unión: se resta el área total de los cuadrados con la intersección
def union(cor_x1_anotacion, cor_y1_anotacion, cor_x1_prediccion, cor_y1_prediccion, dim_anot, dim_pred):
    return 5000 - interseccion(cor_x1_anotacion, cor_y1_anotacion, cor_x1_prediccion, cor_y1_prediccion, dim_anot, dim_pred)
# índice Jaccard con la fórmula: intersección/unión
def jaccard(cor_x1_anotacion, cor_y1_anotacion, cor_x1_prediccion, cor_y1_prediccion, dim_anot, dim_pred):
    return interseccion(cor_x1_anotacion, cor_y1_anotacion, cor_x1_prediccion, cor_y1_prediccion, dim_anot, dim_pred)/union(cor_x1_anotacion, cor_y1_anotacion, cor_x1_prediccion, cor_y1_prediccion, dim_anot, dim_pred)
excel = pandas.read_csv(os.path.join('data_mp3', 'detection_groundtruth.csv')) # uso de librería pandas para importar datos
pred, anot, scores = excel['predictions'], excel['annotations'], excel['scores'] # datos para anotaciones, predicciones y scores
pred, anot = list(pred), list(anot) #cambio a variables tipo list
scores, indices_jaccard = np.array(scores), np.zeros(len(pred)) # inicialización de arreglos
for i in range(len(pred)):
    prueba_anot, prueba_pred = anot[i].split(','), pred[i].split(',') # split de los datos en las comas
    dim_anot = int(prueba_anot[3]) # extracción de las dimensiones de la anotación
    dim_pred = int(prueba_pred[3]) # extraccion de las dimensiones de la predicción
    indices_jaccard[i] = jaccard(int(prueba_anot[0]), int(prueba_anot[1]), int(prueba_pred[0]), int(prueba_pred[1]), dim_anot, dim_pred)# calculo de IoU para todas las predicciones
def precision(list):
    TP = 0
    FP = 0
    for i in list:
        if i == "TP":
            TP += 1
        elif i == "FP":
            FP += 1
    return TP/(TP + FP) # se utiliza la fórmula de precisión
def cobertura(list):
    TP = 0
    for i in list:
        if i == "TP":
            TP += 1
    return TP/len(list) # se utiliza la definición global de la cobertura
def resultados(umbral):
    resultados_precision = np.zeros(100)
    resultados_cobertura = np.zeros(100) # inicialización de variables
    fs = np.zeros(100)
    for i in range(0, 100, 1):
        resultados_umbral1 = [0] * 100
        for j in range(len(indices_jaccard)):
            if indices_jaccard[j] > umbral: # si se sobre pasa el umbral son verdaderas (T)
                if scores[j] > i / 100: # si se sobrepasa el score son positivas
                    resultados_umbral1[j] = "TP"
                else:
                    resultados_umbral1[j] = "TN" # si no se sobrepasa el score son negativas
            else: # si no se sobre pasa el umbral son falsas (F)
                if scores[j] > i / 100:
                    resultados_umbral1[j] = "FP" # si se sobrepasa el score son positivas
                else:
                    resultados_umbral1[j] = "FN" # si no se sobrepasa el score son negativas
        resultados_precision[i] = precision(resultados_umbral1) # calculo de la precisión
        resultados_cobertura[i] = cobertura(resultados_umbral1) # cálculo de la cobretura
        fs[i] = f_medida(resultados_precision[i], resultados_cobertura[i]) # se llena un arreglo con las f medidas
    prom = abs(np.trapz(resultados_precision, resultados_cobertura)) # se calcula el área bajo la curva
    return resultados_cobertura, resultados_precision, max(fs), prom  # se retorna la f medida máxima y los otros resultados
def f_medida(precision, cobertura):
    return (2 * precision * cobertura)/(precision + cobertura) # cálculo de f-medida con la fórmula
resultado_05_fs, resultado_05_prom = resultados(0.5)[2], resultados(0.5)[3] # cálculo resultados para distintos umbrales
resultado_75_fs, resultado_75_prom = resultados(0.75)[2], resultados(0.75)[3]
resultado_95_fs, resultado_95_prom = resultados(0.95)[2], resultados(0.95)[3]
print(resultado_05_fs, resultado_05_prom) # visualización de resultados
print(resultado_75_fs, resultado_75_prom)
print(resultado_95_fs, resultado_95_prom)
##input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
def MyConnComp_201719942_201822262(binary_image, conn = 4):
    elemt_struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])  # elemento estructurante para conectividad default 4
    if conn == 8:
        elemt_struct = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])  # elemento estructurante para conectividad 8
    mask = np.copy(binary_image)
    labeled_image = np.zeros((len(binary_image), len(binary_image[0])))
    tatuaje = 1
    for i in range(len(binary_image)):
        for j in range(len(binary_image[0])):
            if mask[i][j] == 1:
                papel_cal = np.zeros((len(binary_image), len(binary_image[0])))
                papel_cal[i][j] = mask[i][j]
                fin = False
                while fin == False:
                    previo = np.copy(papel_cal)  # se copia el marcador anterior
                    papel_cal = binary_dilation(papel_cal, elemt_struct).astype(papel_cal.dtype)  # se realiza una dilatación del marcador con el elemento estructurante
                    papel_cal = papel_cal + mask  # se hace una suma de las matrices para revisar la intersección
                    for ii in range(len(papel_cal)):
                        for jj in range(len(papel_cal[0])):
                            if papel_cal[ii][jj] > 1:  # dónde haya interseccion se preserva con un 1
                                papel_cal[ii][jj] = 1
                            else:
                                papel_cal[ii][jj] = 0  # dónde no hay intersección se convierte en fondo con un 0
                    if np.array_equal(previo, papel_cal):
                        fin = True  # condición de parada: si el marcador anterior es igual al actual
                mask = mask - papel_cal # se quita el componente conexo
                labeled_image += tatuaje * papel_cal
                tatuaje += 1
    pixel_labels = np.array([])
    for t in range(1, tatuaje):
        xces = []
        yes = []
        for z in range(len(labeled_image)):
            for j in range(len(labeled_image[0])):
                if t == labeled_image[z][j]:
                    xces.append(z)
                    yes.append(j)
        pixel_labels = np.append(pixel_labels, np.ravel_multi_index((xces, yes), (len(binary_image), len(binary_image[0]))))
    return labeled_image, pixel_labels
##
prueba_frutas=io.imread(os.path.join('data_mp3', 'fruits_binary.png')) # se importa la imagen de prueba
for i in range(len(prueba_frutas)):
    for j in range(len(prueba_frutas[0])):
        if prueba_frutas[i][j]==255:
            prueba_frutas[i][j]=1
plt.figure()
plt.subplot(1,2,1)
plt.title("Imagen original")
plt.axis("off")
plt.imshow(prueba_frutas,cmap="gray")
plt.subplot(1,2,2)
plt.title("Imagen componentes conexos")
plt.axis("off")
plt.imshow(MyConnComp_201719942_201822262(prueba_frutas)[0],cmap="gray")
plt.show()
##
imagen_creada1=np.array([[0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0],[0,1,1,0,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0],[1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],[1,0,1,1,1,1,1,1,1,0,1,0,0,0,0,0,0,0,0,0],[1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0],[0,0,0,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0],[0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,1,0,1,1,0],[0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1],[0,0,0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,0,1],[0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1],[0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,0,0],[0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],[1,1,0,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0]])
imagen_creada2=np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1],[0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1],[0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1],[0,1,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0],[0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0],[0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0]])
##input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
plt.figure("vecindad_4_8_Dif")
plt.subplot(1,3,1)
plt.title("Imagen original")
plt.axis("off")
plt.imshow(imagen_creada1,cmap="gray")
plt.subplot(1,3,2)
plt.title("Imagen componentes\nconexos con 4 vecindad")
plt.axis("off")
plt.imshow(MyConnComp_201719942_201822262(imagen_creada1)[0],cmap="gray")
plt.subplot(1,3,3)
plt.title("Imagen componentes\nconexos con 8 vecindad")
plt.axis("off")
plt.imshow(MyConnComp_201719942_201822262(imagen_creada1,conn=8)[0],cmap="gray")
plt.show()
##input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
plt.figure("vecindad_4_8_Igual")
plt.subplot(1,3,1)
plt.title("Imagen original")
plt.axis("off")
plt.imshow(imagen_creada2,cmap="gray")
plt.subplot(1,3,2)
plt.title("Imagen componentes\nconexos con 4 vecindad")
plt.axis("off")
plt.imshow(MyConnComp_201719942_201822262(imagen_creada2)[0],cmap="gray")
plt.subplot(1,3,3)
plt.title("Imagen componentes\nconexos con 8 vecindad")
plt.axis("off")
plt.imshow(MyConnComp_201719942_201822262(imagen_creada2,conn=8)[0],cmap="gray")
plt.show()
##input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
img = glob.glob(os.path.join('data_mp3','blood_cell_dataset', 'noisy_data', '*.png')) # se importan las imágenes
#num_rand=random.randint(0,9) #proceso para hallar imagen de prueba de forma aleatoria
#print(num_rand) #=7
carga_prueba=io.imread(img[7])
def preprocesamiento(carga_color_image):
    imagen_en_Lab = cv2.cvtColor(carga_color_image, cv2.COLOR_BGR2LAB)  # se convierte de rgb a Lab con librería cv2
    filtrado = cv2.medianBlur(imagen_en_Lab, 7)
    filtrado_grises = cv2.cvtColor(cv2.cvtColor(filtrado, cv2.COLOR_LAB2RGB),cv2.COLOR_RGB2GRAY)  # se convierte de rgb a Lab con librería cv2
    return filtrado_grises
def gradiente_morfo(image):
    dilatacion=morfo.dilation(image)
    erosion=morfo.erosion(image)
    return dilatacion-erosion
def watershed_sinmarcador(image,marcadores):
    return segmen.watershed(image,watershed_line=True)
plt.figure()
plt.subplot(1,3,1)
plt.title("Imagen original preprocesada")
plt.axis("off")
plt.imshow(preprocesamiento(carga_prueba),cmap="gray")
plt.subplot(1,3,2)
plt.title("Gradiente morfológico")
plt.axis("off")
plt.imshow(gradiente_morfo(preprocesamiento(carga_prueba)),cmap="gray")
plt.subplot(1,3,3)
plt.title("Watershed sin\nmardadores definidos")
plt.axis("off")
plt.imshow(watershed_sinmarcador(gradiente_morfo(preprocesamiento(carga_prueba))),cmap="gray")
plt.show()
