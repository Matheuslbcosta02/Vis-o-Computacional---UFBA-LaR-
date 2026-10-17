import cv2
import numpy as np
import matplotlib.pyplot as plt

imagem_original = cv2.imread("homografia_imagem_papel.jpg") #opencv lê img em bgr precisa converter depois pra rgb

imagem_original_rgb = cv2.cvtColor(imagem_original,cv2.COLOR_BGR2RGB)

plt.imshow(imagem_original_rgb)

Ponto_esq_sup, Ponto_sup_dir, Ponto_inf_dir, Ponto_inf_esq = plt.ginput(4)

pontos = [Ponto_esq_sup,Ponto_sup_dir,Ponto_inf_dir,Ponto_inf_esq]
print("pontos capturados:")
print(pontos)
print()
print()

pontos_desenho = np.array(pontos, dtype=np.int32)

print("pontos inteiros:")
print(pontos)
print()
print()
imagem_original_copia_desenho = cv2.cvtColor(imagem_original,cv2.COLOR_BGR2RGB)
#breakpoint()

for ponto in pontos_desenho:
    imagem_desenhada = cv2.circle(imagem_original_copia_desenho,ponto,8,(242,74,41),-1)

plt.imshow(imagem_desenhada)
plt.show()

pontos = np.array(pontos, dtype=np.float32)

largura = 700
altura = 1000

p1 = (0,0)
p2 = (largura-1,0)
p3 = (largura-1, altura-1)
p4 = (0, altura-1)

novos_pontos = [p1,p2,p3,p4]

novos_pontos = np.array(novos_pontos, dtype=np.float32)

matriz_homografia = cv2.getPerspectiveTransform(pontos,novos_pontos)
print("MATRIZ DE HOMOGRAFIA:")
print(matriz_homografia)
print()
print()

nova_imagem_retificada = cv2.warpPerspective(imagem_original_rgb,matriz_homografia,(largura,altura))

plt.imshow(nova_imagem_retificada)
plt.show()

plt.imshow(imagem_original_rgb)
ponto_verificacao = plt.ginput(1)
plt.close()
ponto_verificacao = np.array(ponto_verificacao, dtype=np.int32)
x = ponto_verificacao[0][0]
y = ponto_verificacao[0][1]
ponto_verificao_para_desenho = [x,y]
print("ponto de verificação pra desenho original:")
print(ponto_verificao_para_desenho)

ponto_verificacao_homogeneo = np.append(ponto_verificacao,[[1]], axis=1)
print("ponto de verificação homogeno pra calculo com matriz homografia pra gerar ponto transformado")
print(ponto_verificacao_homogeneo)

ponto_verificacao_transformado = matriz_homografia @ ponto_verificacao_homogeneo.T
print("ponto transformado erado após calculo:")
print(ponto_verificacao_transformado)

coordenada_A = (ponto_verificacao_transformado[0][0])/(ponto_verificacao_transformado[-1][0])
coordenada_B = (ponto_verificacao_transformado[1][0])/(ponto_verificacao_transformado[-1][0])

pontos_transformados_finais = [coordenada_A,coordenada_B]
print("pontos transformados obtidos x e y")
print(pontos_transformados_finais)


imagem_original_verificacao = cv2.circle(imagem_original_copia_desenho, ponto_verificao_para_desenho, 8,(242,74,41),-1)
plt.subplot(1, 2, 1) #linha, coluna, posicao --> os dois primeiros definem layout e o ultimo posicao do elemento
plt.imshow(imagem_original_verificacao)


ponto_verificacao_transformado_desenho =  np.array(pontos_transformados_finais, dtype= np.int32)
nova_imagem_retificada_verificacao = cv2.circle(nova_imagem_retificada,ponto_verificacao_transformado_desenho,8,(242,74,41),-1)
plt.subplot(1, 2, 2)
plt.imshow(nova_imagem_retificada_verificacao)
plt.show()
