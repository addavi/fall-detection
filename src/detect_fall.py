import pandas as pd
from pathlib import Path

# ---------- LIMIARES (escolhidos olhando a tabela da Etapa 5) ----------
LIM_VEL = 0.40        # velocidade vertical mínima do quadril (altura/s)
LIM_ANGULO = 45       # ângulo do tronco que indica corpo inclinado (graus)
LIM_DESCIDA = 0.15    # quanto o quadril precisa ter descido (fração da altura)
JANELA_APOS = 30      # frames observados depois do pico de velocidade
FRAMES_PERMANENCIA = 10  # quantos frames seguidos o corpo precisa ficar "caído"


def detectar_queda(feats: pd.DataFrame):
    """Retorna (houve_queda, frame_da_queda)."""
    # Candidatos: frames onde a velocidade passou do limiar
    candidatos = feats.index[feats["vel_vertical"] > LIM_VEL]

    for f in candidatos:
        # Olhamos os frames logo depois do movimento rápido
        depois = feats.loc[f : f + JANELA_APOS]

        # "Caído" = tronco inclinado E quadril baixo, ao mesmo tempo
        caido = (depois["angulo_tronco"] > LIM_ANGULO) & (
            depois["descida_quadril"] > LIM_DESCIDA
        )

        # Precisa ficar caído por frames seguidos (permanência)
        if caido.rolling(FRAMES_PERMANENCIA).sum().max() >= FRAMES_PERMANENCIA:
            return True, int(f)

    return False, None


resultados = []
for pasta in sorted(Path("outputs").iterdir()):
    arq = pasta / "features.csv"
    if not arq.exists():
        continue

    feats = pd.read_csv(arq, index_col="frame")
    real = pasta.name.startswith("fall")
    previsto, frame = detectar_queda(feats)

    if real and previsto:       tipo = "VP (acertou queda)"
    elif real and not previsto: tipo = "FN (perdeu queda)"
    elif not real and previsto: tipo = "FP (alarme falso)"
    else:                       tipo = "VN (acertou normal)"

    resultados.append({"video": pasta.name, "real": "queda" if real else "adl",
                       "previsto": "queda" if previsto else "adl",
                       "frame_detectado": frame, "resultado": tipo})

tabela = pd.DataFrame(resultados)
print(tabela.to_string(index=False))

vp = tabela["resultado"].str.startswith("VP").sum()
fn = tabela["resultado"].str.startswith("FN").sum()
fp = tabela["resultado"].str.startswith("FP").sum()
vn = tabela["resultado"].str.startswith("VN").sum()
print(f"\nVP={vp}  FN={fn}  FP={fp}  VN={vn}")
print(f"Recall:   {vp / (vp + fn):.2f}" if vp + fn else "Recall: n/a")
print(f"Precisão: {vp / (vp + fp):.2f}" if vp + fp else "Precisão: n/a")

Path("docs").mkdir(exist_ok=True)
tabela.to_csv("docs/resultado_regra.csv", index=False)