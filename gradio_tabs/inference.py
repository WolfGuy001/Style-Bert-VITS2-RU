import datetime
import json
from pathlib import Path
from typing import Optional

import gradio as gr

from style_bert_vits2.constants import (
    DEFAULT_ASSIST_TEXT_WEIGHT,
    DEFAULT_LENGTH,
    DEFAULT_LINE_SPLIT,
    DEFAULT_NOISE,
    DEFAULT_NOISEW,
    DEFAULT_SDP_RATIO,
    DEFAULT_SPLIT_INTERVAL,
    DEFAULT_STYLE,
    DEFAULT_STYLE_WEIGHT,
    GRADIO_THEME,
    Languages,
)
from style_bert_vits2.logging import logger
from style_bert_vits2.nlp import InvalidToneError
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker as pyopenjtalk
from style_bert_vits2.nlp.japanese.g2p_utils import g2kata_tone, kata_tone2phone_tone
from style_bert_vits2.nlp.japanese.normalizer import normalize_text
from style_bert_vits2.tts_model import NullModelParam, TTSModelHolder
from style_bert_vits2.utils import torch_device_to_onnx_providers


# pyopenjtalk_worker を起動
## pyopenjtalk_worker は TCP ソケットサーバーのため、ここで起動する
pyopenjtalk.initialize_worker()

# Web UI での学習時の無駄な GPU VRAM 消費を避けるため、あえてここでは BERT モデルの事前ロードを行わない
# データセットの BERT 特徴量は事前に bert_gen.py により抽出されているため、学習時に BERT モデルをロードしておく必要はない
# BERT モデルの事前ロードは「ロード」ボタン押下時に実行される TTSModelHolder.get_model_for_gradio() 内で行われる
# Web UI での学習時、音声合成タブの「ロード」ボタンを押さなければ、BERT モデルが VRAM にロードされていない状態で学習を開始できる

languages = [lang.value for lang in Languages]

initial_text = "Привет! Рад познакомиться. Как тебя зовут?"

examples = [
    [initial_text, "RU"],
    [
        """Я так рада, что ты это сказал, мне очень приятно.
Я так зла, что ты это сказал, я просто в ярости!
Я так удивлена, что ты это сказал, это невероятно!
Мне так грустно от твоих слов, на душе очень тяжело...""",
        "RU",
    ],
    [
        """Слушай, я уже давно за тобой наблюдаю. Твоя улыбка, твоя доброта и сила... они меня просто очаровали.
Пока мы дружили, я поняла, что ты становишься для меня кем-то очень особенным.
Эм, в общем... ты мне очень нравишься! Если ты не против, может... давай будем встречаться?""",
        "RU",
    ],
    [
        """Синтез речи — это технология воссоздания человеческого голоса из текста с помощью машинного обучения.
Эти системы анализируют структуру языка и генерируют соответствующий звук.
Используя современные исследования, можно создавать очень естественные и выразительные голоса.""",
        "RU",
    ],
    [
        "Speech synthesis is the artificial production of human speech. A computer system used for this purpose is called a speech synthesizer, and can be implemented in software or hardware products.",
        "EN",
    ],
    [
        "语音合成是人工制造人类语音。用于此目的的计算机系统称为语音合成器，可以通过软件或硬件产品实现。",
        "ZH",
    ],
]

initial_md = """
- Модели по умолчанию ( Ami и Amitaro) были обучены на основе корпусов и записей стримов Amitaro. Пожалуйста, обязательно ознакомьтесь с **условиями использования** перед их применением.

- Чтобы скачать эти модели после обновления, запустите `Initialize.bat` или скачайте их вручную и поместите в директорию `model_assets`.

- Для реального использования (например, чтения текстов) удобнее использовать **версию с редактором**. Запустите её через `Editor.bat`.
"""

terms_of_use_md = """
## Лицензии и правила

Актуальные правила использования Style-Bert-VITS2 можно найти [здесь](https://github.com/litagin02/Style-Bert-VITS2/blob/master/docs/TERMS_OF_USE.md).

### Пожалуйста, НЕ используйте это для:
- Нарушения закона
- Политических целей
- Причинения вреда другим людям
- Создания дипфейков или подмены личности

### Соблюдайте правила:
При использовании Style-Bert-VITS2, пожалуйста, соблюдайте лицензии конкретных моделей, которые вы используете.
Если вы используете этот исходный код, соблюдайте [лицензию репозитория](https://github.com/litagin02/Style-Bert-VITS2#license).
"""

how_to_md = """
Поместите файлы модели в директорию `model_assets` следующим образом:
```
model_assets
├── your_model
│   ├── config.json
│   ├── your_model_file1.safetensors
│   ├── your_model_file2.safetensors
│   ├── ...
│   └── style_vectors.npy
└── another_model
    ├── ...
```
Для каждой модели необходимы:
- `config.json` (конфигурация обучения)
- `*.safetensors` (файл(ы) весов модели)
- `style_vectors.npy` (файл векторов стиля)

Первые два файла создаются автоматически при обучении через `Train.bat`. Файл `style_vectors.npy` можно создать с помощью `StyleVectors.bat`.
"""

style_md = f"""
- Вы можете управлять голосом, эмоциями и стилем речи с помощью пресетов или эталонных аудиофайлов.
- Стандартный стиль {DEFAULT_STYLE} обычно хорошо справляется с эмоциями на основе текста. Управление стилем позволяет усилить эти эффекты.
- Слишком большая интенсивность стиля может привести к искажению звука.
- Эффективность использования эталонного аудио зависит от того, насколько голос в нем похож на целевую модель.
"""
voice_keys = ["dec"]
voice_pitch_keys = ["flow"]
speech_style_keys = ["enc_p"]
tempo_keys = ["sdp", "dp"]


def make_interactive():
    return gr.update(interactive=True, value="Синтезировать голос")


def make_non_interactive():
    return gr.update(interactive=False, value="Синтезировать голос (сначала загрузите модель)")


def gr_util(item):
    if item == "プリセットから選ぶ":
        return (gr.update(visible=True), gr.Audio(visible=False, value=None))
    else:
        return (gr.update(visible=False), gr.update(visible=True))


null_models_frame = 0


def change_null_model_row(
    null_model_index: int,
    null_model_name: str,
    null_model_path: str,
    null_voice_weights: float,
    null_voice_pitch_weights: float,
    null_speech_style_weights: float,
    null_tempo_weights: float,
    null_models: dict[int, NullModelParam],
):
    null_models[null_model_index] = NullModelParam(
        name=null_model_name,
        path=Path(null_model_path),
        weight=null_voice_weights,
        pitch=null_voice_pitch_weights,
        style=null_speech_style_weights,
        tempo=null_tempo_weights,
    )
    if len(null_models) > null_models_frame:
        keys_to_keep = list(range(null_models_frame))
        result = {k: null_models[k] for k in keys_to_keep}
    else:
        result = null_models
    return result, True


def create_inference_app(model_holder: TTSModelHolder) -> gr.Blocks:
    def tts_fn(
        model_name,
        model_path,
        text,
        language,
        reference_audio_path,
        sdp_ratio,
        noise_scale,
        noise_scale_w,
        length_scale,
        line_split,
        split_interval,
        assist_text,
        assist_text_weight,
        use_assist_text,
        style,
        style_weight,
        kata_tone_json_str,
        use_tone,
        speaker,
        pitch_scale,
        intonation_scale,
        null_models: dict[int, NullModelParam],
        force_reload_model: bool,
    ):
        model_holder.get_model(model_name, model_path)
        assert model_holder.current_model is not None
        logger.debug(f"Null models setting: {null_models}")

        wrong_tone_message = ""
        kata_tone: Optional[list[tuple[str, int]]] = None
        if use_tone and kata_tone_json_str != "":
            if language != "JP":
                logger.warning("Only Japanese is supported for tone generation.")
                wrong_tone_message = "Настройка акцентов сейчас поддерживается только для японского языка."
            if line_split:
                logger.warning("Tone generation is not supported for line split.")
                wrong_tone_message = (
                    "Настройка акцентов поддерживается только при выключенном разделении по строкам."
                )
            try:
                kata_tone = []
                json_data = json.loads(kata_tone_json_str)
                # tupleを使うように変換
                for kana, tone in json_data:
                    assert isinstance(kana, str) and tone in (0, 1), f"{kana}, {tone}"
                    kata_tone.append((kana, tone))
            except Exception as e:
                logger.warning(f"Error occurred when parsing kana_tone_json: {e}")
                wrong_tone_message = f"Некорректная настройка акцента: {e}"
                kata_tone = None

        # toneは実際に音声合成に代入される際のみnot Noneになる
        tone: Optional[list[int]] = None
        if kata_tone is not None:
            phone_tone = kata_tone2phone_tone(kata_tone)
            tone = [t for _, t in phone_tone]

        speaker_id = model_holder.current_model.spk2id[speaker]

        start_time = datetime.datetime.now()

        try:
            sr, audio = model_holder.current_model.infer(
                text=text,
                language=language,
                reference_audio_path=reference_audio_path,
                sdp_ratio=sdp_ratio,
                noise=noise_scale,
                noise_w=noise_scale_w,
                length=length_scale,
                line_split=line_split,
                split_interval=split_interval,
                assist_text=assist_text,
                assist_text_weight=assist_text_weight,
                use_assist_text=use_assist_text,
                style=style,
                style_weight=style_weight,
                given_tone=tone,
                speaker_id=speaker_id,
                pitch_scale=pitch_scale,
                intonation_scale=intonation_scale,
                null_model_params=null_models,
                force_reload_model=force_reload_model,
            )
        except InvalidToneError as e:
            logger.error(f"Tone error: {e}")
            return f"Ошибка: Некорректная настройка акцента:\n{e}", None, kata_tone_json_str
        except ValueError as e:
            logger.error(f"Value error: {e}")
            return f"Ошибка: {e}", None, kata_tone_json_str

        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()

        if tone is None and language == "JP":
            # アクセント指定に使えるようにアクセント情報を返す
            norm_text = normalize_text(text)
            kata_tone = g2kata_tone(norm_text)
            kata_tone_json_str = json.dumps(kata_tone, ensure_ascii=False)
        elif tone is None:
            kata_tone_json_str = ""
        message = f"Успех, время: {duration:.2f} сек."
        if wrong_tone_message != "":
            message = wrong_tone_message + "\n" + message
        return message, (sr, audio), kata_tone_json_str, False

    def get_model_files(model_name: str):
        return [str(f) for f in model_holder.model_files_dict[model_name]]

    model_names = model_holder.model_names
    if len(model_names) == 0:
        logger.error(
            f"Модели не найдены. Пожалуйста, поместите модели в {model_holder.root_dir}."
        )
        with gr.Blocks() as app:
            gr.Markdown(
                f"Ошибка: Модели не найдены. Пожалуйста, поместите модели в {model_holder.root_dir}."
            )
        return app
    initial_id = 0
    initial_pth_files = get_model_files(model_names[initial_id])

    with gr.Blocks(theme=GRADIO_THEME) as app:
        gr.Markdown(initial_md)
        gr.Markdown(terms_of_use_md)
        null_models = gr.State({})
        force_reload_model = gr.State(False)
        with gr.Accordion(label="Инструкция", open=False):
            gr.Markdown(how_to_md)
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column(scale=3):
                        model_name = gr.Dropdown(
                            label="Список моделей",
                            choices=model_names,
                            value=model_names[initial_id],
                        )
                        model_path = gr.Dropdown(
                            label="Файл модели",
                            choices=initial_pth_files,
                            value=initial_pth_files[0],
                        )
                    refresh_button = gr.Button("Обновить", scale=1, visible=True)
                    load_button = gr.Button("Загрузить", scale=1, variant="primary")
                text_input = gr.TextArea(label="Текст", value=initial_text)
                pitch_scale = gr.Slider(
                    minimum=0.8,
                    maximum=1.5,
                    value=1,
                    step=0.05,
                    label="Высота тона (лучше оставить 1)",
                )
                intonation_scale = gr.Slider(
                    minimum=0,
                    maximum=2,
                    value=1,
                    step=0.1,
                    label="Интонация (лучше оставить 1)",
                )

                line_split = gr.Checkbox(
                    label="Разбивать по строкам (улучшает выразительность)",
                    value=DEFAULT_LINE_SPLIT,
                )
                split_interval = gr.Slider(
                    minimum=0.0,
                    maximum=2,
                    value=DEFAULT_SPLIT_INTERVAL,
                    step=0.1,
                    label="Длительность паузы между строками (сек)",
                )
                line_split.change(
                    lambda x: (gr.Slider(visible=x)),
                    inputs=[line_split],
                    outputs=[split_interval],
                )
                tone = gr.Textbox(
                    label="Настройка акцентов (0=низкий, 1=высокий)",
                    info="Только для японского языка. Не идеально.",
                )
                use_tone = gr.Checkbox(label="Использовать настройку акцентов", value=False)
                use_tone.change(
                    lambda x: (gr.Checkbox(value=False) if x else gr.Checkbox()),
                    inputs=[use_tone],
                    outputs=[line_split],
                )
                language = gr.Dropdown(choices=languages, value="RU", label="Язык")
                speaker = gr.Dropdown(label="Диктор")
                with gr.Accordion(label="Дополнительные настройки", open=False):
                    sdp_ratio = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=DEFAULT_SDP_RATIO,
                        step=0.1,
                        label="SDP Ratio",
                    )
                    noise_scale = gr.Slider(
                        minimum=0.1,
                        maximum=2,
                        value=DEFAULT_NOISE,
                        step=0.1,
                        label="Noise",
                    )
                    noise_scale_w = gr.Slider(
                        minimum=0.1,
                        maximum=2,
                        value=DEFAULT_NOISEW,
                        step=0.1,
                        label="Noise_W",
                    )
                    length_scale = gr.Slider(
                        minimum=0.1,
                        maximum=2,
                        value=DEFAULT_LENGTH,
                        step=0.1,
                        label="Length",
                    )
                    use_assist_text = gr.Checkbox(
                        label="Использовать Assist text", value=False
                    )
                    assist_text = gr.Textbox(
                        label="Assist text (текст-помощник)",
                        placeholder="Введите текст с нужной интонацией...",
                        info="Помогает задать манеру речи и эмоции, но может снизить четкость.",
                        visible=False,
                    )
                    assist_text_weight = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=DEFAULT_ASSIST_TEXT_WEIGHT,
                        step=0.1,
                        label="Сила Assist text",
                        visible=False,
                    )
                    use_assist_text.change(
                        lambda x: (gr.Textbox(visible=x), gr.Slider(visible=x)),
                        inputs=[use_assist_text],
                        outputs=[assist_text, assist_text_weight],
                    )
                with gr.Accordion(label="Дополнительные модели (Style Mixing)", open=False):
                    with gr.Row():
                        null_models_count = gr.Number(
                            label="Количество моделей для смешивания", value=0, step=1
                        )
                    with gr.Column(variant="panel"):

                        @gr.render(inputs=[null_models_count])
                        def render_null_models(
                            null_models_count: int,
                        ):
                            global null_models_frame
                            null_models_frame = null_models_count
                            for i in range(null_models_count):
                                with gr.Row():
                                    null_model_index = gr.Number(
                                        value=i,
                                        key=f"null_model_index_{i}",
                                        visible=False,
                                    )
                                    null_model_name = gr.Dropdown(
                                        label="Модель",
                                        choices=model_names,
                                        key=f"null_model_name_{i}",
                                        value=model_names[initial_id],
                                    )
                                    null_model_path = gr.Dropdown(
                                        label="Файл модели",
                                        key=f"null_model_path_{i}",
                                        allow_custom_value=True,
                                    )
                                    null_voice_weights = gr.Slider(
                                        minimum=0,
                                        maximum=1,
                                        value=1,
                                        step=0.1,
                                        key=f"null_voice_weights_{i}",
                                        label="Вес голоса",
                                    )
                                    null_voice_pitch_weights = gr.Slider(
                                        minimum=0,
                                        maximum=1,
                                        value=1,
                                        step=0.1,
                                        key=f"null_voice_pitch_weights_{i}",
                                        label="Вес высоты тона",
                                    )
                                    null_speech_style_weights = gr.Slider(
                                        minimum=0,
                                        maximum=1,
                                        value=1,
                                        step=0.1,
                                        key=f"null_speech_style_weights_{i}",
                                        label="Вес манеры речи",
                                    )
                                    null_tempo_weights = gr.Slider(
                                        minimum=0,
                                        maximum=1,
                                        value=1,
                                        step=0.1,
                                        key=f"null_tempo_weights_{i}",
                                        label="Вес темпа",
                                    )

                                    null_model_name.change(
                                        model_holder.update_model_files_for_gradio,
                                        inputs=[null_model_name],
                                        outputs=[null_model_path],
                                    )
                                    null_model_path.change(
                                        make_non_interactive, outputs=[tts_button]
                                    )
                                    # 愚直すぎるのでもう少しなんとかしたい
                                    null_model_path.change(
                                        change_null_model_row,
                                        inputs=[
                                            null_model_index,
                                            null_model_name,
                                            null_model_path,
                                            null_voice_weights,
                                            null_voice_pitch_weights,
                                            null_speech_style_weights,
                                            null_tempo_weights,
                                            null_models,
                                        ],
                                        outputs=[null_models, force_reload_model],
                                    )
                                    null_voice_weights.change(
                                        change_null_model_row,
                                        inputs=[
                                            null_model_index,
                                            null_model_name,
                                            null_model_path,
                                            null_voice_weights,
                                            null_voice_pitch_weights,
                                            null_speech_style_weights,
                                            null_tempo_weights,
                                            null_models,
                                        ],
                                        outputs=[null_models, force_reload_model],
                                    )
                                    null_voice_pitch_weights.change(
                                        change_null_model_row,
                                        inputs=[
                                            null_model_index,
                                            null_model_name,
                                            null_model_path,
                                            null_voice_weights,
                                            null_voice_pitch_weights,
                                            null_speech_style_weights,
                                            null_tempo_weights,
                                            null_models,
                                        ],
                                        outputs=[null_models, force_reload_model],
                                    )
                                    null_speech_style_weights.change(
                                        change_null_model_row,
                                        inputs=[
                                            null_model_index,
                                            null_model_name,
                                            null_model_path,
                                            null_voice_weights,
                                            null_voice_pitch_weights,
                                            null_speech_style_weights,
                                            null_tempo_weights,
                                            null_models,
                                        ],
                                        outputs=[null_models, force_reload_model],
                                    )
                                    null_tempo_weights.change(
                                        change_null_model_row,
                                        inputs=[
                                            null_model_index,
                                            null_model_name,
                                            null_model_path,
                                            null_voice_weights,
                                            null_voice_pitch_weights,
                                            null_speech_style_weights,
                                            null_tempo_weights,
                                            null_models,
                                        ],
                                        outputs=[null_models, force_reload_model],
                                    )

                    add_btn = gr.Button("ヌルモデルを増やす")
                    del_btn = gr.Button("ヌルモデルを減らす")
                    add_btn.click(
                        lambda x: x + 1,
                        inputs=[null_models_count],
                        outputs=[null_models_count],
                    )
                    del_btn.click(
                        lambda x: x - 1 if x > 0 else 0,
                        inputs=[null_models_count],
                        outputs=[null_models_count],
                    )

            with gr.Column():
                with gr.Accordion("Подробнее о стилях", open=False):
                    gr.Markdown(style_md)
                style_mode = gr.Radio(
                    ["Выбрать пресет", "Использовать аудиофайл"],
                    label="Способ выбора стиля",
                    value="Выбрать пресет",
                )
                style = gr.Dropdown(
                    label=f"Стиль ({DEFAULT_STYLE} - стандартный)",
                    choices=["Загрузите модель"],
                    value="Загрузите модель",
                )
                style_weight = gr.Slider(
                    minimum=0,
                    maximum=20,
                    value=DEFAULT_STYLE_WEIGHT,
                    step=0.1,
                    label="Интенсивность стиля (уменьшите, если голос искажается)",
                )
                ref_audio_path = gr.Audio(
                    label="Референсное аудио", type="filepath", visible=False
                )
                tts_button = gr.Button(
                    "Синтезировать голос (сначала загрузите модель)",
                    variant="primary",
                    interactive=False,
                )
                text_output = gr.Textbox(label="Статус")
                audio_output = gr.Audio(label="Результат")
                with gr.Accordion("Примеры текста", open=False):
                    gr.Examples(examples, inputs=[text_input, language])

        tts_button.click(
            tts_fn,
            inputs=[
                model_name,
                model_path,
                text_input,
                language,
                ref_audio_path,
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                line_split,
                split_interval,
                assist_text,
                assist_text_weight,
                use_assist_text,
                style,
                style_weight,
                tone,
                use_tone,
                speaker,
                pitch_scale,
                intonation_scale,
                null_models,
                force_reload_model,
            ],
            outputs=[text_output, audio_output, tone, force_reload_model],
        )

        model_name.change(
            model_holder.update_model_files_for_gradio,
            inputs=[model_name],
            outputs=[model_path],
        )

        model_path.change(make_non_interactive, outputs=[tts_button])

        refresh_button.click(
            model_holder.update_model_names_for_gradio,
            outputs=[model_name, model_path, tts_button],
        )

        load_button.click(
            model_holder.get_model_for_gradio,
            inputs=[model_name, model_path],
            outputs=[style, tts_button, speaker],
        )

        style_mode.change(
            gr_util,
            inputs=[style_mode],
            outputs=[style, ref_audio_path],
        )

    return app


if __name__ == "__main__":
    import torch

    from config import get_path_config

    path_config = get_path_config()
    assets_root = path_config.assets_root
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_holder = TTSModelHolder(
        assets_root, device, torch_device_to_onnx_providers(device)
    )
    app = create_inference_app(model_holder)
    app.launch(inbrowser=True)
