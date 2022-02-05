import argparse
import torch
import torchaudio
from pathlib import Path
from transformers import Wav2Vec2ProcessorWithLM, Wav2Vec2ForCTC


def main(args):
    processor = Wav2Vec2ProcessorWithLM.from_pretrained(args.model_id)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_id)
    model.to('cpu')

    print('File:', args.path_file)

    wav_file_path = str(Path(args.path_file).absolute())
    waveform, sample_rate = torchaudio.load(wav_file_path)

    if sample_rate != 16000:
        resample = torchaudio.transforms.Resample(
            sample_rate, 16000, resampling_method='sinc_interpolation')
        speech_array = resample(waveform)
        sp = speech_array.squeeze().numpy()
    else:
        sp = waveform.squeeze().numpy()

    # stride_length_s is a tuple of the left and right stride length.
    # With only 1 number, both sides get the same stride, by default
    # the stride_length on one side is 1/6th of the chunk_length_s
    input_values = processor(sp,
                             sample_rate=16000,
                             chunk_length_s=args.chunk_length_s,
                             stride_length_s=(args.stride_length_s_l, args.stride_length_s_r),
                             return_tensors="pt").input_values

    with torch.no_grad():
        logits = model(input_values).logits

    prediction = processor.batch_decode(logits.numpy()).text
    print(prediction[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path_file", type=str, required=True, help="WAV file to transcribe"
    )
    parser.add_argument(
        "--model_id", type=str, required=True, help="Model identifier. Should be loadable with 🤗 Transformers"
    )
    parser.add_argument(
        "--chunk_length_s", type=float, default=None, help="Chunk length in seconds. Defaults to 5 seconds."
    )
    parser.add_argument(
        "--stride_length_s_l", type=int, default=None, help="Stride of the audio chunks, left value."
    )
    parser.add_argument(
        "--stride_length_s_r", type=int, default=None, help="Stride of the audio chunks, right value."
    )
    parser.add_argument(
        "--log_outputs", action="store_true", help="If defined, write outputs to log file for analysis."
    )
    args = parser.parse_args()

    main(args)
