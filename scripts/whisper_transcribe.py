import torch
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
from pathlib import Path
import multiprocessing
from tqdm import tqdm
from _datasets import datasets

model = None
processor = None
prompt = None

def process_batch(args):

    # Prepare
    feature_files, index = args
    process_id = multiprocessing.current_process()._identity[0]
    device = "cuda:" + str(process_id % torch.cuda.device_count())
    file = feature_files[index]

    # Load model
    global model
    global processor
    global prompt
    if model is None:
        print("Loading model...")
        model_name = "distil-whisper/distil-large-v2"
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
        # prompt = processor.get_prompt_ids("Audiobook transcribing, correctly. From LibriVox, public domain. ")

    # Load spectogram
    input_features = torch.load(str(file), map_location = device).half()
    input_features = torch.nn.functional.pad(input_features, (0, 3000 - input_features.shape[2]), "constant", 0)

    # Predict
    predicted_ids = model.generate(input_features, prompt_ids = prompt)
    if prompt is not None:
        predicted_ids.squeeze_(0)
        predicted_ids = predicted_ids[len(prompt):]
        predicted_ids.unsqueeze_(0)

    # Decode
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    with open(str(file).replace(".whisper.pt", ".txt"), "w") as f:
        f.write(transcription[0].strip())

def main():

    # Load files
    print("Loading files...")
    text_files = []
    feature_files = []
    for d in datasets:
        flac_files += list(Path("./datasets/" + d + "/").rglob("*.flac"))
        pt_files += list(Path("./datasets/" + d + "/").rglob("*.whisper.pt"))
    feature_files = [str(f) for f in feature_files]
    feature_files.sort()
    text_files = [str(f) for f in text_files]
    text_files = set(text_files)

    # Unprocessed files
    feature_files_pending = [f for f in feature_files if f.replace(".whisper.pt", ".txt") not in text_files]
    # feature_files_pending = feature_files
    print("Unprocessed files:", len(feature_files_pending), "out of", len(feature_files))

    # Start processing
    workers_count = max(torch.cuda.device_count() * 8, 8) # 8x is the optimal number of workers per GPU for 4090, which ends up in 100% GPU utilization

    # Process the found files
    with multiprocessing.Manager() as manager:
            feature_files_pending = manager.list(feature_files_pending)
            args_list = [(feature_files_pending, i) for i in range(len(feature_files_pending))]
            with multiprocessing.Pool(processes=workers_count) as pool:
                for result in tqdm(pool.imap_unordered(process_batch, args_list, chunksize = 16), total=len(feature_files_pending)):
                    pass

if __name__ == "__main__":
    main()