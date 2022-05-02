import gradio as gr
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM
import torch
import torchaudio

model_name = "Yehor/wav2vec2-xls-r-1b-uk-with-lm"

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name)
processor = Wav2Vec2ProcessorWithLM.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)
model.to("cpu")


# define function to read in sound file
def speech_file_to_array_fn(path, max_seconds=10):
    batch = {"file": path}
    speech_array, sampling_rate = torchaudio.load(batch["file"])
    if sampling_rate != 16000:
        transform = torchaudio.transforms.Resample(
            orig_freq=sampling_rate, new_freq=16000
        )
        speech_array = transform(speech_array)
    speech_array = speech_array[0]
    if max_seconds > 0:
        speech_array = speech_array[: max_seconds * 16000]
    batch["speech"] = speech_array.numpy()
    batch["sampling_rate"] = 16000
    return batch


# tokenize
def inference(audio):
    # read in sound file
    # load dummy dataset and read soundfiles
    sp = speech_file_to_array_fn(audio.name)

    sample_rate = 16000
    # stride_length_s is a tuple of the left and right stride length.
    # With only 1 number, both sides get the same stride, by default
    # the stride_length on one side is 1/6th of the chunk_length_s
    input_values = processor(
        sp["speech"],
        sample_rate=sample_rate,
        chunk_length_s=10,
        stride_length_s=(4, 2),
        return_tensors="pt",
    ).input_values

    with torch.no_grad():
        logits = model(input_values).logits

    pred_ids = torch.argmax(logits, axis=-1).cpu().tolist()
    prediction = tokenizer.decode(pred_ids[0], output_word_offsets=True)

    time_offset = 320 / sample_rate

    total_prediction = []
    words = []
    for item in prediction.word_offsets:
        r = item

        s = round(r['start_offset'] * time_offset, 2)
        e = round(r['end_offset'] * time_offset, 2)

        total_prediction.append(f"{s} - {e}: {r['word']}")
        words.append(r['word'])

    print(prediction[0])

    return "\n".join(total_prediction) + "\n\n" + ' '.join(words)


inputs = gr.inputs.Audio(label="Input Audio", type="file")
outputs = gr.outputs.Textbox(label="Output Text")
title = model_name
description = f"Gradio demo for a {model_name}. To use it, simply upload your audio, or click one of the examples to load them. Read more at the links below. Currently supports .wav 16_000hz files"
article = "<p style='text-align: center'><a href='https://github.com/egorsmkv/wav2vec2-uk-demo' target='_blank'> Github repo</a> | <a href='<HF Space link>' target='_blank'>Pretrained model</a> | Made with help from <a href='https://github.com/robinhad' target='_blank'>@robinhad</a></p>"
examples = [
    ["long_1.wav"],
    ["mer_lviv_interview.wav"],
    ["short_1.wav"],
    ["tsn_2.wav"],
    ["tsn.wav"],
]
gr.Interface(
    inference,
    inputs,
    outputs,
    title=title,
    description=description,
    article=article,
    examples=examples,
).launch()
