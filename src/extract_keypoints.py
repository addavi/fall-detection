from ultralytics import YOLO      # biblioteca que contém o YOLO-Pose
import cv2                        # OpenCV: lê e escreve vídeos
import csv                        # para salvar os keypoints em tabela
from pathlib import Path          # forma moderna de lidar com caminhos de arquivos

# ---------- CONFIGURAÇÕES ----------
VIDEO_PATH = Path("data/raw/fall-01-cam0.mp4")  # troque pelo nome do seu vídeo
OUTPUT_DIR = Path("outputs") / VIDEO_PATH.stem  # cria outputs/fall-01-cam0/
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)   # cria a pasta se não existir

# ---------- CARREGAR O MODELO ----------
# "n" = nano, o menor e mais rápido. Baixa sozinho na primeira execução.
model = YOLO("yolo11n-pose.pt")

# ---------- ABRIR O VÍDEO ----------
cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    raise SystemExit(f"Não consegui abrir o vídeo: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)                    # frames por segundo
largura_total = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # largura do vídeo
altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))   # altura do vídeo

# O vídeo tem profundidade (esquerda) + RGB (direita), cada um com metade da largura
largura = largura_total // 2   # largura de UMA das imagens
print(f"Vídeo: {largura_total}x{altura} a {fps:.1f} FPS | usando só a metade RGB: {largura}x{altura}")

# ---------- PREPARAR AS SAÍDAS ----------
# Vídeo de saída (com esqueleto desenhado)
writer = cv2.VideoWriter(
    str(OUTPUT_DIR / "video_esqueleto.mp4"),
    cv2.VideoWriter_fourcc(*"mp4v"),   # codec do arquivo mp4
    fps,
    (largura, altura),
)

# CSV de saída (uma linha por frame)
csv_file = open(OUTPUT_DIR / "keypoints.csv", "w", newline="")
csv_writer = csv.writer(csv_file)

# Cabeçalho: frame, x0, y0, c0, x1, y1, c1, ... até o keypoint 16
# (c = confiança do modelo naquele ponto, de 0 a 1)
cabecalho = ["frame"] + [f"{eixo}{i}" for i in range(17) for eixo in ("x", "y", "c")]
csv_writer.writerow(cabecalho)

# ---------- LOOP PRINCIPAL: processa frame a frame ----------
frame_idx = 0
while True:
    ok, frame = cap.read()      # lê um frame; ok=False quando o vídeo acaba
    if not ok:
        break

        # Fica só com a metade direita (RGB). [linhas, colunas]: todas as linhas, da metade até o fim
    frame = frame[:, largura:]    
    # Roda o YOLO-Pose no frame. [0] pega o resultado da única imagem enviada.
    result = model(frame, verbose=False)[0]

    # Desenha caixa + esqueleto sobre o frame e grava no vídeo de saída
    writer.write(result.plot())

    # Se detectou pelo menos uma pessoa, guardamos os keypoints da primeira
    if result.keypoints is not None and len(result.keypoints) > 0:
        xy = result.keypoints.xy[0].cpu().numpy()      # shape (17, 2): posições x,y
        conf = result.keypoints.conf[0].cpu().numpy()  # shape (17,): confianças

        linha = [frame_idx]
        for i in range(17):
            linha += [xy[i][0], xy[i][1], conf[i]]
        csv_writer.writerow(linha)
    # Se não detectou ninguém, não grava linha (trataremos na Etapa 4)

    frame_idx += 1

# ---------- FINALIZAR ----------
cap.release()       # libera o vídeo de entrada
writer.release()    # fecha o vídeo de saída
csv_file.close()    # fecha o CSV
print(f"Pronto! {frame_idx} frames processados. Veja a pasta '{OUTPUT_DIR}'.")