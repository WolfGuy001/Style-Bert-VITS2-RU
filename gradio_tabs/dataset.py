import gradio as gr

from style_bert_vits2.constants import GRADIO_THEME
from style_bert_vits2.logging import logger
from style_bert_vits2.utils.subprocess import run_script_with_log


def do_slice(
    model_name: str,
    min_sec: float,
    max_sec: float,
    min_silence_dur_ms: int,
    time_suffix: bool,
    input_dir: str,
):
    if model_name == "":
        return "Ошибка: Введите имя модели."
    logger.info("Start slicing...")
    cmd = [
        "slice.py",
        "--model_name",
        model_name,
        "--min_sec",
        str(min_sec),
        "--max_sec",
        str(max_sec),
        "--min_silence_dur_ms",
        str(min_silence_dur_ms),
    ]
    if time_suffix:
        cmd.append("--time_suffix")
    if input_dir != "":
        cmd += ["--input_dir", input_dir]
    # Игнорируем предупреждения ONNX
    success, message = run_script_with_log(cmd, ignore_warning=True)
    if not success:
        return f"Ошибка: {message}"
    return "Нарезка аудио завершена."


def do_transcribe(
    model_name,
    whisper_model,
    compute_type,
    language,
    initial_prompt,
    use_hf_whisper,
    batch_size,
    num_beams,
    hf_repo_id,
):
    if model_name == "":
        return "Ошибка: Введите имя модели."
    if hf_repo_id == "litagin/anime-whisper":
        logger.info(
            "Так как litagin/anime-whisper не поддерживает начальный промпт, он будет проигнорирован."
        )
        initial_prompt = ""

    cmd = [
        "transcribe.py",
        "--model_name",
        model_name,
        "--model",
        whisper_model,
        "--compute_type",
        compute_type,
        "--language",
        language,
        "--initial_prompt",
        f'"{initial_prompt}"',
        "--num_beams",
        str(num_beams),
    ]
    if use_hf_whisper:
        cmd.append("--use_hf_whisper")
        cmd.extend(["--batch_size", str(batch_size)])
        if hf_repo_id != "openai/whisper":
            cmd.extend(["--hf_repo_id", hf_repo_id])
    success, message = run_script_with_log(cmd, ignore_warning=True)
    if not success:
        return f"Ошибка: {message}. Если сообщение об ошибке пустое, возможно, проблем нет — проверьте файл транскрипции."
    return "Транскрибация аудио завершена."


how_to_md = """
Инструмент для создания датасетов для обучения Style-Bert-VITS2. Состоит из двух этапов:

- Нарезка длинных аудио на фрагменты подходящей длины (Slice)
- Транскрибация (превращение голоса в текст)

Вы можете использовать оба этапа или только второй, если нарезка не требуется. **Если ваши аудиофайлы уже имеют подходящую длину (2-12 сек), нарезка не нужна.**

## Что требуется

Несколько аудиофайлов для обучения (wav, mp3 и др.). 
Желательно иметь хотя бы 10 минут записи (хотя сообщалось об успехах и с меньшим объемом). Можно использовать один длинный файл или несколько коротких.

## Как использовать нарезку (Slice)
1. Поместите все аудиофайлы в папку `inputs` (если хотите разделить по стилям, создайте подпапки для каждого стиля).
2. Введите `Имя модели`, настройте параметры при необходимости и нажмите кнопку `Запустить нарезку`.
3. Готовые фрагменты будут сохранены в `Data/{Имя модели}/raw`.

## Как использовать транскрибацию
1. Убедитесь, что аудиофайлы находятся в `Data/{Имя модели}/raw`.
2. Настройте параметры и нажмите кнопку `Запустить транскрибацию`.
3. Файл транскрипции будет сохранен в `Data/{Имя модели}/esd.list`.

## Примечание

- Фрагменты длиннее 12-15 секунд могут не использоваться при обучении (хотя в версии 2.5 это ограничение можно снять). Рекомендуется нарезать аудио на фрагменты умеренной длины для стабильности и меньшего потребления VRAM.
- Качество транскрибации может потребовать ручной проверки/правки в зависимости от чистоты записи.
"""


def create_dataset_app() -> gr.Blocks:
    with gr.Blocks(theme=GRADIO_THEME) as app:
        gr.Markdown(
            "**Если у вас уже есть набор файлов по 2-12 секунд и данные транскрипции, эту вкладку использовать не нужно.**"
        )
        with gr.Accordion("Инструкция", open=False):
            gr.Markdown(how_to_md)
        model_name = gr.Textbox(
            label="Введите имя модели (используется как имя диктора)."
        )
        with gr.Accordion("Нарезка аудио (Slice)"):
            gr.Markdown(
                "**Если аудио уже нарезано, просто положите его в Data/{Имя модели}/raw, и этот шаг не потребуется.**"
            )
            with gr.Row():
                with gr.Column():
                    input_dir = gr.Textbox(
                        label="Путь к исходной папке",
                        value="inputs",
                        info="Положите файлы wav/mp3 в указанную папку.",
                    )
                    min_sec = gr.Slider(
                        minimum=0,
                        maximum=10,
                        value=2,
                        step=0.5,
                        label="Минимальная длительность (удалить всё, что короче)",
                    )
                    max_sec = gr.Slider(
                        minimum=0,
                        maximum=15,
                        value=12,
                        step=0.5,
                        label="Максимальная длительность (разрезать длиннее этого)",
                    )
                    min_silence_dur_ms = gr.Slider(
                        minimum=0,
                        maximum=2000,
                        value=700,
                        step=100,
                        label="Мин. тишина для разделения (мс)",
                    )
                    time_suffix = gr.Checkbox(
                        value=False,
                        label="Добавить временной диапазон в имя WAV файла",
                    )
                    slice_button = gr.Button("Запустить нарезку")
                result1 = gr.Textbox(label="Результат")
        with gr.Row():
            with gr.Column():
                use_hf_whisper = gr.Checkbox(
                    label="Использовать HuggingFace Whisper (быстрее, но требует больше VRAM)",
                    value=False,
                )
                whisper_model = gr.Dropdown(
                    [
                        "large",
                        "large-v2",
                        "large-v3",
                    ],
                    label="Модель Whisper",
                    value="large-v3",
                    visible=True,
                )
                hf_repo_id = gr.Dropdown(
                    [
                        "openai/whisper-large-v3-turbo",
                        "openai/whisper-large-v3",
                        "openai/whisper-large-v2",
                        "kotoba-tech/kotoba-whisper-v2.1",
                        "litagin/anime-whisper",
                    ],
                    label="HuggingFace Whisper repo_id",
                    value="openai/whisper-large-v3-turbo",
                    visible=False,
                )
                compute_type = gr.Dropdown(
                    [
                        "int8",
                        "int8_float32",
                        "int8_float16",
                        "int8_bfloat16",
                        "int16",
                        "float16",
                        "bfloat16",
                        "float32",
                    ],
                    label="Точность вычислений",
                    value="bfloat16",
                    visible=True,
                )
                batch_size = gr.Slider(
                    minimum=1,
                    maximum=128,
                    value=16,
                    step=1,
                    label="Размер батча",
                    info="Ускоряет работу, но требует больше VRAM",
                    visible=False,
                )
                language = gr.Dropdown(["ja", "en", "zh", "ru"], value="ru", label="Язык")
                initial_prompt = gr.Textbox(
                    label="Начальный промпт",
                    value="Здравствуйте. Как ваши дела? Хм, я... я в порядке!",
                    info="Пример того, как должен выглядеть текст (пунктуация, стиль и т.д.)",
                )
                num_beams = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=1,
                    step=1,
                    label="Количество лучей (num_beams)",
                    info="Меньше = быстрее (рекомендуется 1)",
                )
            transcribe_button = gr.Button("Запустить транскрибацию")
            result2 = gr.Textbox(label="Результат")
        slice_button.click(
            do_slice,
            inputs=[
                model_name,
                min_sec,
                max_sec,
                min_silence_dur_ms,
                time_suffix,
                input_dir,
            ],
            outputs=[result1],
        )
        transcribe_button.click(
            do_transcribe,
            inputs=[
                model_name,
                whisper_model,
                compute_type,
                language,
                initial_prompt,
                use_hf_whisper,
                batch_size,
                num_beams,
                hf_repo_id,
            ],
            outputs=[result2],
        )
        use_hf_whisper.change(
            lambda x: (
                gr.update(visible=not x),
                gr.update(visible=x),
                gr.update(visible=x),
                gr.update(visible=not x),
            ),
            inputs=[use_hf_whisper],
            outputs=[whisper_model, hf_repo_id, batch_size, compute_type],
        )

    return app


if __name__ == "__main__":
    app = create_dataset_app()
    app.launch(inbrowser=True)
