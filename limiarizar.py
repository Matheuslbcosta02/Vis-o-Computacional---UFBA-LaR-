import cv2
import numpy as np
import matplotlib.pyplot as plt

imagem_original = cv2.imread('imagem_original_limiarizar.jpg', cv2.IMREAD_GRAYSCALE)
plt.imshow(imagem_original, cmap='gray')
plt.show()

altura, largura = imagem_original.shape
print(f"Altura {altura} e largura {largura}")

total_pixels = altura * largura
print(f"Total de pixels: {total_pixels}")

menor_intensidade = imagem_original.min()
maior_intensidade = imagem_original.max()
intensidade_media = imagem_original.mean()

print(f'Menor intensidade encontrada: {menor_intensidade}, maior intesidade encontrada: {maior_intensidade}, intensidade media: {intensidade_media}')

print("Total de intensidades distintas:", np.unique(imagem_original).size)

breakpoint()
histograma, bordas = np.histogram(imagem_original, bins=256,range=(0,256))
intensidades = np.arange(256)

plt.figure(figsize=(14,5))
plt.plot(intensidades,histograma)
plt.xlabel("intensidades")
plt.ylabel("Qtd. Pixels")
plt.title("Histograma")
plt.show()

print("Soma do histograma:", np.sum(histograma))


histograma_normalizado = histograma/total_pixels


plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(intensidades,histograma)
plt.xlabel("intensidades")
plt.ylabel("Qtd. Pixels")
plt.title("Histograma")
plt.subplot(1,2,2)
plt.plot(intensidades, histograma_normalizado)
plt.xlabel("intensidades")
plt.ylabel("frequência relativa")
plt.title("Histograma Normalizado")
plt.show()


plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(intensidades,histograma)
plt.yscale("log")
plt.xlabel("intensidades")
plt.ylabel("Qtd. Pixels")
plt.title("Histograma")
plt.subplot(1,2,2)
plt.plot(intensidades, histograma_normalizado)
plt.yscale("log")
plt.xlabel("intensidades")
plt.ylabel("frequência relativa")
plt.title("Histograma Normalizado")
plt.show()

soma_histo_normalizado = np.sum(histograma_normalizado)
print(f"histograma normalizado: {soma_histo_normalizado}")

plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(intensidades,histograma)
plt.axvline(x=235, color='red', linestyle='-', label='Limiar T = 235')
plt.legend()
plt.yscale("log")
plt.xlabel("intensidades")
plt.ylabel("Qtd. Pixels")
plt.title("Histograma")
plt.subplot(1,2,2)
plt.plot(intensidades,histograma)
plt.axvline(x=235, color='red', linestyle='-', label='Limiar T = 235')
plt.legend()
plt.xlabel("intensidades")
plt.ylabel("Qtd. Pixels")
plt.title("Histograma")
plt.show()

#aplicando minha limiaarização na imagem cinza pra obter a img binária
limiar= 235
limiar_usei,imagem_binaria = cv2.threshold(imagem_original,limiar,255, cv2.THRESH_BINARY)
print(limiar_usei)
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.title("Imagem Original cinza")
plt.imshow(imagem_original, cmap='gray')
plt.subplot(1,2,2)
plt.title("Imagem Binária")
plt.imshow(imagem_binaria, cmap='binary')
plt.show()

limiar1 = 220
limiar_usei,imagem_binaria1 = cv2.threshold(imagem_original,limiar1,255, cv2.THRESH_BINARY)

limiar2 = 250
limiar_usei,imagem_binaria2 = cv2.threshold(imagem_original,limiar2,255, cv2.THRESH_BINARY)

plt.figure(figsize=(14,5))
plt.subplot(1,3,1)
plt.title("Imagem Binária - Limiar 220")
plt.imshow(imagem_binaria1, cmap='binary')
plt.subplot(1,3,2)
plt.title("Imagem Binária - Limiar 235")
plt.imshow(imagem_binaria, cmap='binary')
plt.subplot(1,3,3)
plt.title("Imagem Binária - Limiar 250")
plt.imshow(imagem_binaria2, cmap='binary')
plt.show()

T_otsu , binaria_otsu = cv2.threshold(imagem_original,0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.title("Imagem binária usando limiar que eu escolhi - T = 235")
plt.imshow(imagem_binaria, cmap='binary')
plt.subplot(1,2,2)
plt.title(f"Imagem binária usando limiar Otsu - T = {T_otsu}")
plt.imshow(binaria_otsu, cmap='binary')
plt.show()

n_0 =np.count_nonzero(imagem_binaria == 0)
n_1=np.count_nonzero(imagem_binaria == 255)

print(f"quantidade pixels na imagem binária com intesidade 0: {n_0}")
print(f"quantidade pixels na imagem binária com intesidade 255: {n_1}")

altura_binaria, largura_binaria = imagem_binaria.shape

total_pixels_binaria = altura_binaria * largura_binaria

p0 = n_0/total_pixels_binaria
p1 = n_1/total_pixels_binaria

print(f"proporções: {p0} e {p1}")

soma_proporcoes = p0 + p1
print(soma_proporcoes)
