import os
import pandas as pd
from pydub import AudioSegment

AUDIO_DIR = "../data/extrAudio"
ANNOTATIONS_DIR = "../data/datasetAnnotation"
OUTPUT_DIR = "../data/audio_clips"

os.makedirs(OUTPUT_DIR, exist_ok=True)

utterance_count = 0

for annotation_file in os.listdir(ANNOTATIONS_DIR):
    if not annotation_file.endswith(".csv"):
        continue

    base_name = annotation_file.replace(".csv", ".wav")
    audio_path = os.path.join(AUDIO_DIR, base_name)
    annotation_path = os.path.join(ANNOTATIONS_DIR, annotation_file)

    if not os.path.exists(audio_path):
        print(f"Fișier audio lipsă pentru {annotation_file}, sărit.")
        continue

    audio = AudioSegment.from_wav(audio_path)
    df = pd.read_csv(annotation_path)

    # Parcurgem fiecare rostire
    for i, row in df.iterrows():
        speaker = str(row["Speaker"]).strip()

        # Ignorăm rostirile care nu aparțin subiecților (ex: TM)
        if speaker.upper() == "TM":
            continue

        start_ms = int(row["Start time"] * 1000)
        end_ms = int(row["Stop time"] * 1000)
        gender = row["Gender"]

        utterance = audio[start_ms:end_ms]

        # Ex: trial_lie_001_speaker03_utterance5.wav
        output_name = f"{base_name.replace('.wav','')}_speaker{speaker}_utt{i+1}.wav"
        output_path = os.path.join(OUTPUT_DIR, output_name)

        utterance.export(output_path, format="wav")
        utterance_count += 1

    print(f"Procesat: {base_name}")

print(f"\nTotal rostiri extrase : {utterance_count}")
