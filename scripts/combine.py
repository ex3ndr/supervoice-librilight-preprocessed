from pathlib import Path
import multiprocessing
from tqdm import tqdm
import textgrid
import argparse

def copy_files(args):
    id, dataset = args

    # Soruce files
    src_flac = Path("./datasets/" + dataset).joinpath(id + ".flac")
    src_text = Path("./datasets/" + dataset).joinpath(id + ".txt")
    src_codec = Path("./datasets/" + dataset).joinpath(id + ".codec.pt")
    src_speaker = Path("./datasets/" + dataset).joinpath(id + ".speaker.pt")
    src_textgrid = Path("./datasets/" + dataset + "-aligned").joinpath(id + ".TextGrid")

    # Destination files
    Path("./datasets/" + dataset + "-processed").joinpath(Path(id).parent).mkdir(parents=True, exist_ok=True)

    dst_flac = Path("./datasets/" + dataset + "-processed").joinpath(id + ".flac")
    dst_text = Path("./datasets/" + dataset + "-processed").joinpath(id + ".txt")
    dst_codec = Path("./datasets/" + dataset + "-processed").joinpath(id + ".codec.pt")
    dst_speaker = Path("./datasets/" + dataset + "-processed").joinpath(id + ".speaker.pt")
    dst_textgrid = Path("./datasets/" + dataset + "-processed").joinpath(id + ".TextGrid")

    # Copy files
    dst_flac.write_bytes(src_flac.read_bytes())
    dst_text.write_bytes(src_text.read_bytes())
    dst_codec.write_bytes(src_codec.read_bytes())
    dst_speaker.write_bytes(src_speaker.read_bytes())
    dst_textgrid.write_bytes(src_textgrid.read_bytes())

def process_textgrid(args):
    id, dataset = args
    src_textgrid = Path("./datasets/" + dataset + "-aligned").joinpath(id + ".TextGrid")
    tg = textgrid.TextGrid.fromFile(str(src_textgrid))
    for i in range(len(tg)):
        for j in range(len(tg[i])):
            if tg[i][j].mark == "spn":
                return None
    return id

def main(dataset):

    print("Loading files...")
    flac_files = list(Path("./datasets/" + dataset).rglob("*.flac"))
    text_files = list(Path("./datasets/" + dataset).rglob("*.txt"))
    codec_files = list(Path("./datasets/" + dataset).rglob("*.codec.pt"))
    speaker_files = list(Path("./datasets/" + dataset).rglob("*.speaker.pt"))
    textgrid_files = list(Path("./datasets/" + dataset + "-aligned").rglob("*.TextGrid"))

    # Filter text files
    print("Filtering text files...")   
    textgrid_files = list([str(file.with_suffix("").relative_to("./datasets/" + dataset + "-aligned")) for file in textgrid_files])
    valid_textgrids = [] 
    with multiprocessing.Manager() as manager:
        args_list = [(textgrid_files[i], dataset) for i in range(len(textgrid_files))]
        with multiprocessing.Pool(processes=32) as pool:
            for result in tqdm(pool.imap_unordered(process_textgrid, args_list, chunksize = 16), total=len(args_list)):
                if result is not None:
                    valid_textgrids.append(result)
    textgrid_filtered_files = set(valid_textgrids)
    print("Valid textgrid files:", len(textgrid_filtered_files))

    print("Processing files...")
    # textgrid_files = set([str(file.with_suffix("").relative_to("./datasets/" + dataset + "-aligned")) for file in textgrid_files])
    flac_files = set([str(file.with_suffix("").relative_to("./datasets/" + dataset)) for file in flac_files])
    text_files = set([str(file.with_suffix("").relative_to("./datasets/" + dataset)) for file in text_files])
    codec_files = set([str(file.with_suffix("").with_suffix("").relative_to("./datasets/" + dataset)) for file in codec_files])
    speaker_files = set([str(file.with_suffix("").with_suffix("").relative_to("./datasets/" + dataset)) for file in speaker_files])
    all_files = flac_files.intersection(text_files, set(textgrid_files), codec_files, speaker_files)
    all_files = list(all_files)
    all_files.sort()
    valid_files = flac_files.intersection(text_files, textgrid_filtered_files, codec_files, speaker_files)
    valid_files = list(valid_files)
    valid_files.sort()

    print("Copying files...")
    Path("./datasets/" + dataset + "-processed").mkdir(parents=True, exist_ok=True)

    # Write file index
    with open("./datasets/" + dataset + "-processed/files_all.txt", "w") as f:
        for file in all_files:
            f.write(file + "\n")
    with open("./datasets/" + dataset + "-processed/files_valid.txt", "w") as f:
        for file in valid_files:
            f.write(file + "\n")

    # Write files
    with multiprocessing.Manager() as manager:
        args_list = [(all_files[i], dataset) for i in range(len(all_files))]
        with multiprocessing.Pool(processes=32) as pool:
            for result in tqdm(pool.imap_unordered(copy_files, args_list, chunksize = 16), total=len(args_list)):
                pass

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Dataset to process.", type=str, required=True)
    args = parser.parse_args()

    # Invoke main
    main(args.dataset)