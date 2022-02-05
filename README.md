# Demo of Ukrainian wav2vec2 model

The model is hosted here: https://huggingface.co/Yehor/wav2vec2-xls-r-1b-uk-with-lm

Follow our community in Telegram: https://t.me/speech_recognition_uk

---

Install deps:

```bash
pip install https://github.com/huggingface/transformers/archive/refs/tags/v4.16.2.zip
pip install https://github.com/kpu/kenlm/archive/master.zip

pip install torch==1.9.1 torchaudio==0.9.1 pyctcdecode==0.3.0
```

Run inference:

```bash
python inference.py --model_id Yehor/wav2vec2-xls-r-1b-uk-with-lm --path_file short_1.wav

# with chunking
python inference.py --model_id Yehor/wav2vec2-xls-r-1b-uk-with-lm --path_file short_1.wav --chunk_length_s 10 --stride_length_s_l 4 --stride_length_s_r 2
python inference.py --model_id Yehor/wav2vec2-xls-r-1b-uk-with-lm --path_file long_1.wav --chunk_length_s 10 --stride_length_s_l 4 --stride_length_s_r 2
```

NOTE: Do the inference process for long files with chunking.

---

short_1.wav:

```
пана сполучені штати над важливий стратегічний партнер однак є різницяштати мають спеціальний закон який передбачає якщо китай напади на тайвань американський військові мають його захищати у гри
```

long_1.wav:

```
серце чи дивовижни порятунок мільйони людей фактично в прямому ефірі вже три доби спостерігають за спробамиамероканських рятувальникив дісттисколодя за пятирічне хлопя досі не зрозуміло чи вдастядістати його з тридцяти метрового провал живим про надзвичайно складну операцію що триває в цю мить я на есарчуккулояз який провалився пятирічнийраян ледь помітна діра в землі менше тридцяти сантиметріву діаметрі але в глиб вона тягнеться на тридцять два метро батьки шукали сина кілька один перед тим як зрозуміле він під землею коли він зник я молилися богупросила аби алагзбиріг мосина і його дістали з колодязь живим господихай йому та менше болить в тій ділі я так сподіваючиь що у рятувальники все вийде його неможливо витягти просто так розуміють рятувальники занадто вуськоа розширяти діру не можна вона просто завалитья тому вони три до бою розкопують амундалік і поки працює техніки
```
