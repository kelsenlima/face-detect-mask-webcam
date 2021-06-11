# importando bibliotecas necessarias
# caso retorne algum erro de biblioteca inexistente
# providencie a instalacao com comando
# py -m pip install nome_biblioteca
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
	# pegue as dimensões da moldura e, em seguida, construa uma bolha
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# passe o blob pela rede e obtenha as detecções de rosto
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# inicializar nossa lista de faces, suas localizações correspondentes,
	# e a lista de previsões da nossa rede de máscaras
	faces = []
	locs = []
	preds = []

	# fazer um loop sobre as detecções
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		# filtrar detecções fracas, garantindo que a confiança é
		# maior do que a confiança mínima
		if confidence > 0.5:
			# calcular as coordenadas (x, y) da caixa delimitadora para
			# o objeto
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# certifique-se de que as caixas delimitadoras estejam dentro das dimensões de
			# a moldura
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extrair o ROI do rosto, convertê-lo do canal BGR para RGB
			# ordenar, redimensionar para 224 x 224 e pré-processar
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# adicione a face e as caixas delimitadoras aos seus respectivos
			# listas
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# só faça previsões se pelo menos um rosto for detectado
	if len(faces) > 0:
		# para uma inferência mais rápida, faremos previsões em lote em * todos *
		# enfrenta ao mesmo tempo, em vez de previsões um por um
		# no loop `for` acima
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# retornar uma 2-tupla das localizações de face e seus correspondentes
	# localizacoes
	return (locs, preds)

# carregar nosso modelo de detector facial serializado do disco
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# carregar o modelo do detector de máscara facial do disco
maskNet = load_model("mask_detector.model")

# inicializar o stream de vídeo
print("[INFO] iniciando vídeo...")
vs = VideoStream(src=0).start()

# fazer um loop sobre os quadros do stream de vídeo
while True:
	# pega o quadro do stream de vídeo encadeado e redimensione-o
	# ter uma largura máxima de 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# detectar rostos no quadro e determinar se eles estão usando um
	# máscara facial ou não
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# fazer um loop sobre os locais de rosto detectados e seus correspondentes
	# localizacoes
	for (box, pred) in zip(locs, preds):
		# descompacte a caixa delimitadora e as previsões
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# determinar o rótulo da classe e a cor que usaremos para desenhar
		# a caixa delimitadora e o texto
		label = "Com mascara" if mask > withoutMask else "Sem mascara"
		color = (0, 255, 0) if label == "Com mascara" else (0, 0, 255)

		# inclua a probabilidade no rótulo
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# exibir o rótulo e o retângulo da caixa delimitadora na saída
		# quadro
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# mostre o quadro de saída
	cv2.imshow("Mascara", frame)
	key = cv2.waitKey(1) & 0xFF

	# se a tecla `q` foi pressionada, interrompa o loop
	if key == ord("q"):
		break

# faça uma pequena limpeza
cv2.destroyAllWindows()
vs.stop()
