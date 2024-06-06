from keras.models import load_model
import cv2
import numpy as np

np.set_printoptions(suppress=True)
modelo = load_model("C:/keras_Model.h5", compile=False)
nomes_classes = open("C:/labels.txt", "r").readlines()
caminho_video = "C:/video_teste.mkv"
camera = cv2.VideoCapture(caminho_video)
rastreador = cv2.TrackerCSRT_create() if hasattr(cv2, 'TrackerCSRT_create') else cv2.TrackerKCF_create()
rastreando = False
caixa_delimitadora = None
nome_classe = ""
pontuacao_confianca = 0.0

detecoes = []

while True:
    ret, imagem = camera.read()
    if ret:
        imagem_redimensionada = cv2.resize(imagem, (224, 224), interpolation=cv2.INTER_AREA)
        array_imagem = np.asarray(imagem_redimensionada, dtype=np.float32).reshape(1, 224, 224, 3)
        array_imagem = (array_imagem / 127.5) - 1
        previsao = modelo.predict(array_imagem)

        for i in range(len(previsao[0])):
            nome_classe_temp = nomes_classes[i].strip()
            pontuacao_confianca_temp = previsao[0][i]
            if pontuacao_confianca_temp > 0.75:
                detecoes.append((nome_classe_temp, pontuacao_confianca_temp))
                if pontuacao_confianca_temp > pontuacao_confianca:
                    nome_classe = nome_classe_temp
                    pontuacao_confianca = pontuacao_confianca_temp
                    if not rastreando:
                        caixa_delimitadora = (25, 25, 200, 200)
                        rastreador = cv2.TrackerCSRT_create() if hasattr(cv2, 'TrackerCSRT_create') else cv2.TrackerKCF_create()
                        rastreador.init(imagem, caixa_delimitadora)
                        rastreando = True

    if rastreando:
        sucesso, caixa_delimitadora = rastreador.update(imagem)
        if sucesso:
            p1 = (int(caixa_delimitadora[0]), int(caixa_delimitadora[1]))
            p2 = (int(caixa_delimitadora[0] + caixa_delimitadora[2]), int(caixa_delimitadora[1] + caixa_delimitadora[3]))
            cv2.rectangle(imagem, p1, p2, (0, 0, 255), 2, 1)
            cv2.putText(imagem, nome_classe, (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(imagem, f'{np.round(pontuacao_confianca * 100)}%', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            rastreando = False

    cv2.imshow("Quadro do Vídeo", imagem)
    tecla_teclado = cv2.waitKey(1)

    if tecla_teclado == 27:
        break


camera.release()
cv2.destroyAllWindows()

print(detecoes)


