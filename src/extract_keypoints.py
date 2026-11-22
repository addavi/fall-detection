from ultralytics import YOLO
import cv2
import csv
from pathlib import Path

model = YOLO("yolo11n-pose.pt")  # carrega o modelo uma única vez


def processar_video(video_path: Path):
    """Processa um vídeo: salva o vídeo com esqueleto e o CSV de keypoints."""
    output_dir = Path("outputs") / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Não consegui abrir: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2  # só a metade RGB
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{video_path.name}: {largura}x{altura} a {fps:.1f} FPS")

    writer = cv2.VideoWriter(
        str(output_dir / "video_esqueleto.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), fps, (largura, altura),
    )
    csv_file = open(output_dir / "keypoints.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame"] + [f"{e}{i}" for i in range(17) for e in ("x", "y", "c")])

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = frame[:, largura:]                     # recorte: só o RGB
        result = model(frame, verbose=False)[0]
        writer.write(result.plot())

        if result.keypoints is not None and len(result.keypoints) > 0:
            xy = result.keypoints.xy[0].cpu().numpy()
            conf = result.keypoints.conf[0].cpu().numpy()
            linha = [frame_idx]
            for i in range(17):
                linha += [xy[i][0], xy[i][1], conf[i]]
            csv_writer.writerow(linha)
        frame_idx += 1

    cap.release()
    writer.release()
    csv_file.close()
    print(f"  -> {frame_idx} frames processados")


# Roda todos os vídeos da pasta data/raw (ordenados por nome)
for video in sorted(Path("data/raw").glob("*.mp4")):
    processar_video(video)