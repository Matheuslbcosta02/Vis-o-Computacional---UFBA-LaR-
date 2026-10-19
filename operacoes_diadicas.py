import cv2
import numpy as np
import matplotlib.pyplot as plt

imagem1 = cv2.imread('foto1_diadica.jpg', cv2.IMREAD_GRAYSCALE)
imagem2 = cv2.imread('foto2_diadica.jpg', cv2.IMREAD_GRAYSCALE)

largura1, altura1 = imagem1.shape
largura2, altura2 = imagem2.shape
tipo_img1 = imagem1.dtype
tipo_img2 = imagem2.dtype

menor_intensidade1 = np.min(imagem1)
maior_intensidade1 = np.max(imagem1)
media_intensidade1 = np.mean(imagem1)

menor_intensidade2 = np.min(imagem2)
maior_intensidade2 = np.max(imagem2)
media_intensidade2 = np.mean(imagem2)

print(f" infos img1: tipo:{tipo_img1},largura:{largura1}, altura:{altura1} min intensidade:{menor_intensidade1}, max intensidade: {maior_intensidade1}, media intensidade: {media_intensidade1}")
print()
print(f" infos img2: tipo:{tipo_img2},largura:{largura2}, altura:{altura2} min intensidade:{menor_intensidade2}, max intensidade: {maior_intensidade2}, media intensidade: {media_intensidade2}")

dimensao_img1 = largura1 * altura1
dimensao_img2 = largura2 * altura2
print(f'dimensão imagem1: {dimensao_img1}    dimensão imagem2: {dimensao_img2}')

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.title('Imagem 1')
plt.imshow(imagem1, cmap='gray')
plt.subplot(1,2,2)
plt.title('Imagem 2')
plt.imshow(imagem2, cmap='gray')
plt.show()

#converter pra uma representacao numerica que aceita resultados sem saturar
soma_imagens = imagem1.astype(np.int16) + imagem2.astype(np.int16)

menor_valor_soma = np.min(soma_imagens)
maior_valor_soma = np.max(soma_imagens)
qtd_maiores_255 = np.sum(soma_imagens > 255)

print(f"Maior valor da soma: {maior_valor_soma}, menor valor da soma: {menor_valor_soma}, qtd maiores que 255: {qtd_maiores_255}")

plt.figure(figsize=(12,5))
plt.subplot(1,3,1)
plt.title('Imagem 1')
plt.imshow(imagem1, cmap='gray')
plt.subplot(1,3,2)
plt.title('Imagem 2')
plt.imshow(imagem2, cmap='gray')
plt.subplot(1,3,3)
plt.title('Imagem soma')
plt.imshow(soma_imagens, cmap='gray')
plt.show()

subtracao_imagem = imagem1.astype(np.int16) - imagem2.astype(np.int16)
menor_valor_subtracao = np.min(subtracao_imagem)
maior_valor_subtracao = np.max(subtracao_imagem)
valores_negativos_subtracao = np.sum(subtracao_imagem<0)
valores_positivos_subtracao = np.sum(subtracao_imagem>0)

print(f'menor valor sub: {menor_valor_subtracao}, maior valor sub: {maior_valor_subtracao}, valores postivos da sub: {valores_positivos_subtracao}, valores negativos da sub: {valores_negativos_subtracao}')

plt.figure(figsize=(12,5))
plt.subplot(1,3,1)
plt.title('Imagem 1')
plt.imshow(imagem1, cmap='gray')
plt.subplot(1,3,2)
plt.title('Imagem 2')
plt.imshow(imagem2, cmap='gray')
plt.subplot(1,3,3)
plt.title('Imagem subtração')
plt.imshow(subtracao_imagem, cmap='gray')
plt.show()

diferenca_absoluta = np.abs(subtracao_imagem).astype(np.uint8)

plt.figure(figsize=(14,5))
plt.subplot(1,4,1)
plt.title('Imagem 1')
plt.imshow(imagem1, cmap='gray')
plt.subplot(1,4,2)
plt.title('Imagem 2')
plt.imshow(imagem2, cmap='gray')
plt.subplot(1,4,3)
plt.title('Imagem subtração')
plt.imshow(subtracao_imagem, cmap='gray')
plt.subplot(1,4,4)
plt.title('Imagem subtração absoluta')
plt.imshow(diferenca_absoluta, cmap='gray')
plt.show()

#limiarizar imagem

histograma, bordas = np.histogram(diferenca_absoluta, bins=256, range=(0,256))
intesindades_histograma = np.arange(256)

plt.figure(figsize=(14,5))
plt.plot(intesindades_histograma,histograma)
plt.xlabel('intensidades')
plt.ylabel('qtd. pixels')
plt.title('histograma')
plt.show()

#olhando grafico escolho limiar = 25

#limiarização da imagem para obter uma inmagem no estilo binário com apenas dois valores 0 e 1

limiar_escolhido = 27

limiar_usei, imagem_binarizada = cv2.threshold(diferenca_absoluta, limiar_escolhido, 255, cv2.THRESH_BINARY)
#MAIOR QUE 25 vira 255 e menor ou igual vira zero

plt.figure(figsize=(14,5))
plt.subplot(1,3,1)
plt.title('Imagem diferença absoluta')
plt.imshow(diferenca_absoluta,cmap='gray')
plt.subplot(1,3,2)
plt.title(f'Imagem binária usando limiar = {limiar_escolhido}')
plt.imshow(imagem_binarizada, cmap='gray')
plt.show()

alpha1= 0.25
alpha2 = 0.5
alpha3 = 0.75

combinacao_ponderada1 = cv2.addWeighted(imagem1,alpha1,imagem2,1-alpha1,0.0) #sem brilho 0.0
combinacao_ponderada2 = cv2.addWeighted(imagem1,alpha2,imagem2,1-alpha2,0.0) #sem brilho 0.0
combinacao_ponderada3 = cv2.addWeighted(imagem1,alpha3,imagem2,1-alpha3,0.0) #sem brilho 0.0

plt.figure(figsize=(16,5))
plt.subplot(1,3,1)
plt.title(f'combinação ponderada com alpha = {alpha1}')
plt.imshow(combinacao_ponderada1, cmap='gray')
plt.subplot(1,3,2)
plt.title(f'combinação ponderada com alpha = {alpha2}')
plt.imshow(combinacao_ponderada2, cmap='gray')
plt.subplot(1,3,3)
plt.title(f'combinação ponderada com alpha = {alpha3}')
plt.imshow(combinacao_ponderada3, cmap='gray')
plt.subplots_adjust(hspace=0.7)
plt.show()

#mais translucido pois dei mais peso pra foto 1 sem o objeto, foi aumentando alpha e aumentando peso pra foto sem o corretivo

#criando minhas mascara

M = np.zeros(imagem1.shape, dtype=np.uint8)
#regiao retangular
y1,y2 = 1500,2000
x1,x2 = 1000,2000
M[y1:y2, x1:x2]=255

#aplicando mascara na imagem1

Im = imagem1 * M

plt.figure(figsize=(14,5))
plt.subplot(1,3,1)
plt.title("Imagem original 1")
plt.imshow(imagem1, cmap='gray')
plt.subplot(1,3,2)
plt.title("Máscara usada")
plt.imshow(M, cmap='gray')
plt.subplot(1,3,3)
plt.title("Imagem resultante depois da máscara")
plt.imshow(Im, cmap='gray')
plt.show()

plt.figure(figsize=(16,8))
plt.subplot(2,3,1)
plt.title("Imagem original 1")
plt.imshow(imagem1, cmap='gray')
plt.subplot(2,3,2)
plt.title("Imagem 2")
plt.imshow(imagem2, cmap='gray')
plt.subplot(2,3,3)
plt.title("Soma das imagens")
plt.imshow(soma_imagens, cmap='gray')
plt.subplot(2,3,4)
plt.title("Diferença absoluta")
plt.imshow(diferenca_absoluta, cmap='gray')
plt.subplot(2,3,5)
plt.title(f'combinação ponderada com alpha = {alpha2}')
plt.imshow(combinacao_ponderada2, cmap='gray')
plt.subplot(2,3,6)
plt.title("Imagem resultante depois da máscara")
plt.imshow(Im, cmap='gray')
plt.subplots_adjust(hspace=0.6)
plt.show()


