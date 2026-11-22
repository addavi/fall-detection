import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FRAMES_BASE = 10   # primeiros frames usados como "posição inicial" do quadril
JANELA_VEL = 5     # suavização da velocidade (frames)


def calcular_features(pasta: Path):
    stem = pasta.name

    # Precisamos do FPS e do tamanho da imagem para o cálculo ficar correto
    cap = cv2.VideoCapture(str(Path("data/raw") / f"{stem}.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2   # só a metade RGB
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    df = pd.read_csv(pasta / "keypoints_tratado.csv", index_col="frame")

    # Ponto médio dos ombros (keypoints 5 e 6) e dos quadris (11 e 12)
    # mean(axis=1) ignora NaN: se um lado sumiu, usa o outro
    ombro_x = df[["x5", "x6"]].mean(axis=1) * largura   # volta para pixels
    ombro_y = df[["y5", "y6"]].mean(axis=1) * altura
    quadril_x = df[["x11", "x12"]].mean(axis=1) * largura
    quadril_y = df[["y11", "y12"]].mean(axis=1) * altura

    # 1) ÂNGULO DO TRONCO em relação à vertical
    # Voltamos para pixels antes de calcular, senão a normalização
    # (x/largura, y/altura) distorceria o ângulo.
    dx = (ombro_x - quadril_x).abs()
    dy = (ombro_y - quadril_y).abs()
    angulo = np.degrees(np.arctan2(dx, dy))   # 0° = vertical, 90° = horizontal

    # 2) DESCIDA DO QUADRIL em relação à posição inicial (fração da altura da imagem)
    quadril_y_norm = quadril_y / altura
    base = quadril_y_norm.iloc[:FRAMES_BASE].median()
    descida = quadril_y_norm - base            # positivo = quadril mais baixo

    # 3) VELOCIDADE VERTICAL do quadril (altura da imagem por segundo)
    # gradient = variação entre frames vizinhos; multiplicar por fps -> por segundo
    vel = pd.Series(np.gradient(quadril_y_norm.values) * fps, index=df.index)
    vel = vel.rolling(JANELA_VEL, center=True, min_periods=1).mean()

    feats = pd.DataFrame({
        "angulo_tronco": angulo,
        "descida_quadril": descida,
        "vel_vertical": vel,
    })
    feats.to_csv(pasta / "features.csv", index_label="frame")

    # Gráfico com 3 painéis
    fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    ax[0].plot(feats["angulo_tronco"]);   ax[0].set_ylabel("Ângulo tronco (°)")
    ax[1].plot(feats["descida_quadril"]); ax[1].set_ylabel("Descida quadril")
    ax[2].plot(feats["vel_vertical"]);    ax[2].set_ylabel("Velocidade (alt/s)")
    ax[2].set_xlabel("Frame")
    ax[0].set_title(stem)
    for a in ax:
        a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(pasta / "grafico_features.png", dpi=150)
    plt.close()

    return feats, fps


# ---------- Roda em todos os vídeos e monta o gráfico comparativo ----------
resumo = []
fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=False)

for pasta in sorted(Path("outputs").iterdir()):
    if not (pasta / "keypoints_tratado.csv").exists():
        continue
    feats, fps = calcular_features(pasta)
    eh_queda = pasta.name.startswith("fall")
    cor = "red" if eh_queda else "tab:blue"

    ax[0].plot(feats["angulo_tronco"], color=cor, alpha=0.6)
    ax[1].plot(feats["vel_vertical"], color=cor, alpha=0.6)

    resumo.append({
        "video": pasta.name,
        "classe": "queda" if eh_queda else "adl",
        "vel_max": feats["vel_vertical"].max(),
        "angulo_max": feats["angulo_tronco"].max(),
        "angulo_final": feats["angulo_tronco"].dropna().iloc[-15:].mean(),
    })

ax[0].set_ylabel("Ângulo tronco (°)")
ax[1].set_ylabel("Velocidade (alt/s)")
ax[1].set_xlabel("Frame")
ax[0].set_title("Queda (vermelho) x ADL (azul)")
plt.tight_layout()
Path("docs").mkdir(exist_ok=True)
plt.savefig("docs/comparativo_features.png", dpi=150)

tabela = pd.DataFrame(resumo).round(2)
print(tabela.to_string(index=False))
tabela.to_csv("docs/resumo_features.csv", index=False)