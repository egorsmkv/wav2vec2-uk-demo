"""
Code is borrowed here: https://github.com/requaos/ai-transcriber/blob/main/main.py#L183

Author: https://github.com/requaos/
License: MIT
"""

import argparse
import torch
import librosa
from pathlib import Path
from transformers import Wav2Vec2Processor, Wav2Vec2ProcessorWithLM, Wav2Vec2ForCTC

chunk_duration = 18  # sec
padding_duration = 2  # sec

sample_rate = 16_000
device = 'cpu'

def main(args):
    processor = Wav2Vec2ProcessorWithLM.from_pretrained(args.model_id)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_id)
    model.to(device)

    files = args.path_files.split(',')

    for path_file in files:
        print('File:', path_file)

        wav_file_path = str(Path(path_file).absolute())
        audio, _ = librosa.load(wav_file_path, sr=sample_rate)

        chunk_len = chunk_duration * sample_rate
        input_padding_len = int(padding_duration * sample_rate)
        output_padding_len = model._get_feat_extract_output_lengths(input_padding_len)

        logits_all = []
        for start in range(input_padding_len, len(audio) - input_padding_len, chunk_len):
            chunk = audio[start - input_padding_len:start + chunk_len + input_padding_len]

            input_values = processor(chunk, sampling_rate=sample_rate, return_tensors="pt").input_values
            with torch.no_grad():
                logits = model(input_values.to(device)).logits[0]
                x_logits = logits[output_padding_len:len(logits) - output_padding_len]
                logits_all.append(x_logits)

        x = torch.cat(logits_all)
        prediction = processor.decode(x.numpy())
        print(prediction.text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path_files", type=str, required=True, help="WAV files to transcribe, separated by a comma"
    )
    parser.add_argument(
        "--model_id", type=str, required=True, help="Model identifier. Should be loadable with 🤗 Transformers"
    )
    args = parser.parse_args()

    main(args)
