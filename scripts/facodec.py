import torch
import torchaudio
from transformers import pipeline
from pathlib import Path
import multiprocessing
from tqdm import tqdm

# 
# Facodec
#

facodec_instance = None
def get_facodec():
    global facodec_instance
    process_id = multiprocessing.current_process()._identity[0]
    device = "cuda:" + str(process_id % torch.cuda.device_count())
    if facodec_instance is None:
        facodec_instance = torch.hub.load(repo_or_dir='ex3ndr/facodec', model='facodec')
        facodec_instance.to(device)
    return facodec_instance, device

#
# Codec implementation
#

def process_features(args):
    """
    Extracting whisper mel-spectogram features using GPU(!) to speed up the process.
    """
    files, index = args
    file = files[index]
    base_file = file.replace(".flac", "")
    process_id = multiprocessing.current_process()._identity[0]
    facodec, device = get_facodec()

    # Load file
    audio, sr = torchaudio.load(file)
    if sr != 16000:
        audio = torchaudio.transforms.Resample(sr, 16000, device)(file)
    
    # Convert to mono
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # Convert to single dimension
    audio = audio[0]

    # Cut to 30s
    if audio.shape[0] > 16000 * 30:
        audio = audio[:16000 * 30]

    # Move to device
    audio = audio.to(device)

    # Run facodec
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        prosody_code, cotent_code, residual_code, spk_embs = facodec.encode(audio)
    
    # Concatenate
    codes = torch.cat([prosody_code, cotent_code, residual_code], dim=0)

    # Save
    torch.save(codes, base_file + ".codec.pt")
    torch.save(spk_embs, base_file + ".speaker.pt")
    

def main():

    # Load files
    print("Loading files...")
    directories = ["./datasets/librilight-medium/"]
    flac_files = []
    pt_files = []
    for d in directories:
        flac_files += list(Path(d).rglob("*.flac"))
        pt_files += list(Path(d).rglob("*.codec.pt"))
    flac_files = [str(f) for f in flac_files]
    pt_files = [str(f) for f in pt_files]
    flac_files.sort()
    pt_files = set(pt_files)

    # Filter flac_files that have pt file next to it with the same name
    filtered_flac_files = [flac_file for flac_file in flac_files if flac_file.replace(".flac", ".codec.pt") not in pt_files]
    # filtered_flac_files = flac_files
    print("Unprocessed files:", len(filtered_flac_files), "out of", len(flac_files))

    # Start processing
    workers_count = torch.cuda.device_count() * 4

    # Process the found files
    multiprocessing.set_start_method('spawn')
    torch.hub.load(repo_or_dir='ex3ndr/facodec', model='facodec', trust_repo = True) # Load to avoid race-conditions
    with multiprocessing.Manager() as manager:
        files = manager.list(filtered_flac_files)
        args_list = [(files, i) for i in range(len(files))]
        with multiprocessing.Pool(processes=workers_count) as pool:
            for result in tqdm(pool.imap_unordered(process_features, args_list, chunksize = 8), total=len(files)):
                pass

if __name__ == "__main__":
    main()