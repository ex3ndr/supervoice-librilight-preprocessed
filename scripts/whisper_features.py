import torch
import torchaudio
from transformers import pipeline
from pathlib import Path
import multiprocessing
from tqdm import tqdm
from _datasets import datasets

melscale_fbank_cache = {}
def melscale_fbanks(n_mels, n_fft, f_min, f_max, sample_rate, mel_norm, mel_scale, device):
    global melscale_fbank_cache
    key = str(n_mels) + "_" + str(n_fft) + "_" + str(f_min) + "_" + str(f_max) + "_" + str(sample_rate) + "_" + str(mel_norm) + "_" + str(mel_scale) + "_"+ str(device)
    if key in melscale_fbank_cache:
        return melscale_fbank_cache[key]
    else:
        res = torchaudio.functional.melscale_fbanks(
            n_freqs=int(n_fft // 2 + 1),
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            norm=mel_norm,
            mel_scale=mel_scale
        ).transpose(-1, -2).to(device)
        melscale_fbank_cache[key] = res
        return res

def process_features(args):
    """
    Extracting whisper mel-spectogram features using GPU(!) to speed up the process.
    """
    files, index = args
    file = files[index]
    process_id = multiprocessing.current_process()._identity[0]
    device = "cuda:" + str(process_id % torch.cuda.device_count())

    # Load file
    audio, sr = torchaudio.load(file)
    audio = audio.to(device)
    if sr != 16000:
        audio = torchaudio.transforms.Resample(sr, 16000, device)(file)

    # Spectogram
    window = torch.hann_window(400).to(audio.device)
    stft = torch.stft(audio, 400, 160, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = melscale_fbanks(80, 400, 0, 8000, 16000, "slaney", "slaney", device)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    # Save
    torch.save(log_spec.half(), file.replace(".flac", ".pt"))
    

def main():

    # Load files
    print("Loading files...")
    flac_files = []
    pt_files = []
    for d in datasets:
        flac_files += list(Path("./datasets/" + d + "/").rglob("*.flac"))
        pt_files += list(Path("./datasets/" + d + "/").rglob("*.codec.pt"))
    flac_files = [str(f) for f in flac_files]
    pt_files = [str(f) for f in pt_files]
    flac_files.sort()
    pt_files = set(pt_files)

    # Filter flac_files that have pt file next to it with the same name
    filtered_flac_files = [flac_file for flac_file in flac_files if flac_file.replace(".flac", ".pt") not in pt_files]
    # filtered_flac_files = flac_files
    print("Unprocessed files:", len(filtered_flac_files), "out of", len(flac_files))

    # Start processing
    workers_count = max(torch.cuda.device_count() * 4, 8)

    # Process the found files
    with multiprocessing.Manager() as manager:
        files = manager.list(filtered_flac_files)
        args_list = [(files, i) for i in range(len(files))]
        with multiprocessing.Pool(processes=workers_count) as pool:
            for result in tqdm(pool.imap_unordered(process_features, args_list, chunksize = 16), total=len(files)):
                pass

if __name__ == "__main__":
    main()