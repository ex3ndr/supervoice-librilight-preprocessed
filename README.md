# Librilight Preprocessed

This is a preprocessed `librilight` dataset with ASR using Whisper Large model and then aligned with montreal forced aligner.

> [!WARNING]  
> This attempt of building a good large aligned dataset failed due to too low quality of whisper ASR for such task. Use libriheavy instead.

## Dataset Structure

Structure is similar to original dataset. Each file is represented in three formats: `.flac` with audio, `.txt` with text, `.TextGrid` with alignment. Top level folders are speakers, next one is a session and then the files split into up to `30 seconds` with rougtly `15 seconds` on average.

## Downloads

This dataset can easily be downloaded using my [datasets](https://github.com/ex3ndr/datasets) tool using identifiers: `librilight-processed`, `librilight-processed@medium` and `librilight-processed@large`.

Or it can be downloaded directly from [my server](https://shared.korshakov.com/datasets/supervoice/librilight/).

## Reproduction

To reproduce the dataset you need to execute the following steps:

```bash
datasets sync # Download source datasets (depends on your network)
./prepare_cut.sh # Cut the audio files (fast)
./prepare_transcribe.sh # Transcribe the audio files using Whisper, this could take days and GPUs are needed
./prepare_align.sh # Align the transcribed files using montreal forced aligner, this could take days
./prepare_final.sh # Prepare the final datasets
```

## License

MIT
