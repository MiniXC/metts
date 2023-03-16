import lco
import librosa
import soundfile as sf
from glob import glob
import os 
import shutil
from tqdm.contrib.concurrent import process_map
from pathlib import Path

path = os.environ["METTS_PATH"]

lco.init("config/config.yaml")

target_sr = lco["audio"]["sampling_rate"]

# loop recursivly through all wavs, resample them to 22050 and save in place
# for wav in glob(path + "/**/*.wav", recursive=True):
#     y, sr = librosa.load(wav, sr=target_sr, res_type="kaiser_fast")
#     sf.write(wav, y, sr, subtype="PCM_16")

# with multiprocessing
def resample(wav):
    if Path(wav).stat().st_size >= 10_000:
        old_sr = librosa.get_samplerate(wav)
        if old_sr != target_sr:
            y, sr = librosa.load(wav, sr=target_sr, res_type="kaiser_fast")
            sf.write(wav, y, sr, subtype="PCM_16")

def main():
    wav_list = list(glob(path + "/**/*.wav", recursive=True))
    process_map(resample, wav_list, max_workers=96, chunksize=1)

if __name__ == "__main__":
    main()