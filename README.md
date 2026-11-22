# Detecção de Quedas com YOLO11-Pose

Projeto de visão computacional que detecta quedas de pessoas usando
estimativa de pose (YOLO11-Pose) e uma regra baseada em keypoints.
Trabalho acadêmico em andamento.

## Status
- [x] Etapa 1: Ambiente e repositório
- [x] Etapa 2: Dataset (URFD, 5 quedas + 5 ADL)
- [x] Etapa 3: Extração de keypoints
- [x] Etapa 4: Tratamento dos dados
- [x] Etapa 5: Features (ângulo do tronco, descida e velocidade do quadril)
- [x] Etapa 6: Regra de decisão
- [ ] Etapa 7: Métricas completas e vídeos extras de ADL

## Pipeline
1. `src/extract_keypoints.py`: roda o YOLO11-Pose e salva os keypoints
2. `src/process_keypoints.py`: filtro de confiança, interpolação, suavização e normalização
3. `src/features.py`: calcula as features
4. `src/detect_fall.py`: aplica a regra e gera as métricas

## Dataset
UR Fall Detection Dataset (URFD), câmera 0. Vídeos não incluídos no
repositório; baixe no site oficial e coloque em `data/raw/`.

## Como rodar
​```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/extract_keypoints.py
python src/process_keypoints.py
python src/features.py
python src/detect_fall.py
​```

## Resultados preliminares (10 vídeos)
VP=3, FN=2, FP=0, VN=5 | Recall 0,60 | Precisão 1,00

Limitações: amostra pequena, limiares ajustados nos mesmos dados,
ADL sem "deitar no chão".