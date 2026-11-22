import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")              # permite salvar gráficos sem abrir janela
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- PARÂMETROS (você pode citar no documento) ----------
LIMIAR_CONF = 0.5     # keypoints abaixo disso são descartados
INTERP_MAX = 10       # só preenche buracos de até 10 frames seguidos
JANELA = 5            # tamanho da média móvel (em frames)


def tratar_video(pasta: Path):
    stem = pasta.name
    video_path = Path("data/raw") / f"{stem}.mp4"

    # Pegamos total de frames e tamanho da imagem a partir do vídeo original
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2   # só a metade RGB
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    df = pd.read_csv(pasta / "keypoints.csv")

    # 1) ALINHAR: garante uma linha para cada frame (os sem detecção ficam NaN)
    df = df.set_index("frame").reindex(range(total_frames))
    detectados = df["x0"].notna().sum()

    # Guardamos o quadril ANTES do tratamento, só para comparar no gráfico
    quadril_antes = (df["y11"] + df["y12"]) / 2

    # 2) FILTRAR: confiança baixa -> NaN
    for i in range(17):
        ruim = df[f"c{i}"] < LIMIAR_CONF
        df.loc[ruim, [f"x{i}", f"y{i}"]] = np.nan

    # 3) INTERPOLAR: preenche buracos curtos (limit_area="inside" não inventa
    #    dados no começo/fim do vídeo)
    cols_xy = [f"{e}{i}" for i in range(17) for e in ("x", "y")]
    df[cols_xy] = df[cols_xy].interpolate(
        method="linear", limit=INTERP_MAX, limit_area="inside"
    )

    # 4a) SUAVIZAR: média móvel centrada
    df[cols_xy] = df[cols_xy].rolling(JANELA, center=True, min_periods=1).mean()

    # 4b) NORMALIZAR: x / largura, y / altura -> valores entre 0 e 1
    for i in range(17):
        df[f"x{i}"] = df[f"x{i}"] / largura
        df[f"y{i}"] = df[f"y{i}"] / altura

    # Salva o CSV tratado (só coordenadas, sem confiança)
    df[cols_xy].to_csv(pasta / "keypoints_tratado.csv", index_label="frame")

    # Gráfico antes x depois (figura para o documento)
    quadril_depois = (df["y11"] + df["y12"]) / 2 * altura   # volta para pixels
    plt.figure(figsize=(8, 3))
    plt.plot(quadril_antes, ".", markersize=3, label="Antes (bruto)")
    plt.plot(quadril_depois, "-", label="Depois (tratado)")
    plt.gca().invert_yaxis()       # no vídeo, y cresce para BAIXO
    plt.xlabel("Frame")
    plt.ylabel("Altura do quadril (px)")
    plt.title(f"{stem}: efeito do tratamento")
    plt.legend()
    plt.tight_layout()
    plt.savefig(pasta / "grafico_tratamento.png", dpi=150)
    plt.close()

    print(f"{stem}: {detectados}/{total_frames} frames com detecção "
          f"({100 * detectados / total_frames:.0f}%)")


for pasta in sorted(Path("outputs").iterdir()):
    if (pasta / "keypoints.csv").exists():
        tratar_video(pasta)