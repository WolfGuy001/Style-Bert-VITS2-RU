import json
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path

import gradio as gr
import yaml

from config import get_path_config
from style_bert_vits2.constants import GRADIO_THEME
from style_bert_vits2.logging import logger
from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT
from style_bert_vits2.utils.subprocess import run_script_with_log, second_elem_of


logger_handler = None
tensorboard_executed = False

path_config = get_path_config()
dataset_root = path_config.dataset_root


@dataclass
class PathsForPreprocess:
    dataset_path: Path
    esd_path: Path
    train_path: Path
    val_path: Path
    config_path: Path


def get_path(model_name: str) -> PathsForPreprocess:
    assert model_name != "", "Имя модели не может быть пустым"
    dataset_path = dataset_root / model_name
    esd_path = dataset_path / "esd.list"
    train_path = dataset_path / "train.list"
    val_path = dataset_path / "val.list"
    config_path = dataset_path / "config.json"
    return PathsForPreprocess(dataset_path, esd_path, train_path, val_path, config_path)


def initialize(
    model_name: str,
    batch_size: int,
    epochs: int,
    save_every_steps: int,
    freeze_EN_bert: bool,
    freeze_JP_bert: bool,
    freeze_ZH_bert: bool,
    freeze_style: bool,
    freeze_decoder: bool,
    use_jp_extra: bool,
    log_interval: int,
):
    global logger_handler
    paths = get_path(model_name)

    # 前処理のログをファイルに保存する
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"preprocess_{timestamp}.log"
    if logger_handler is not None:
        logger.remove(logger_handler)
    logger_handler = logger.add(paths.dataset_path / file_name)

    logger.info(
        f"Step 1: start initialization...\nmodel_name: {model_name}, batch_size: {batch_size}, epochs: {epochs}, save_every_steps: {save_every_steps}, freeze_ZH_bert: {freeze_ZH_bert}, freeze_JP_bert: {freeze_JP_bert}, freeze_EN_bert: {freeze_EN_bert}, freeze_style: {freeze_style}, freeze_decoder: {freeze_decoder}, use_jp_extra: {use_jp_extra}"
    )

    default_config_path = (
        "configs/config.json" if not use_jp_extra else "configs/config_jp_extra.json"
    )

    with open(default_config_path, encoding="utf-8") as f:
        config = json.load(f)
    config["model_name"] = model_name
    config["data"]["training_files"] = str(paths.train_path)
    config["data"]["validation_files"] = str(paths.val_path)
    config["train"]["batch_size"] = batch_size
    config["train"]["epochs"] = epochs
    config["train"]["eval_interval"] = save_every_steps
    config["train"]["log_interval"] = log_interval

    config["train"]["freeze_EN_bert"] = freeze_EN_bert
    config["train"]["freeze_JP_bert"] = freeze_JP_bert
    config["train"]["freeze_ZH_bert"] = freeze_ZH_bert
    config["train"]["freeze_style"] = freeze_style
    config["train"]["freeze_decoder"] = freeze_decoder

    config["train"]["bf16_run"] = False  # デフォルトでFalseのはずだが念のため

    # 今はデフォルトであるが、以前は非JP-Extra版になくバグの原因になるので念のため
    config["data"]["use_jp_extra"] = use_jp_extra

    model_path = paths.dataset_path / "models"
    if model_path.exists():
        logger.warning(
            f"Step 1: {model_path} already exists, so copy it to backup to {model_path}_backup"
        )
        shutil.copytree(
            src=model_path,
            dst=paths.dataset_path / "models_backup",
            dirs_exist_ok=True,
        )
        shutil.rmtree(model_path)
    pretrained_dir = Path("pretrained" if not use_jp_extra else "pretrained_jp_extra")
    try:
        shutil.copytree(
            src=pretrained_dir,
            dst=model_path,
        )
    except FileNotFoundError:
        logger.error(f"Step 1: {pretrained_dir} folder not found.")
        return False, f"Step 1, Error: {pretrained_dir}フォルダが見つかりません。"

    with open(paths.config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    if not Path("config.yml").exists():
        shutil.copy(src="default_config.yml", dst="config.yml")
    with open("config.yml", encoding="utf-8") as f:
        yml_data = yaml.safe_load(f)
    yml_data["model_name"] = model_name
    yml_data["dataset_path"] = str(paths.dataset_path)
    with open("config.yml", "w", encoding="utf-8") as f:
        yaml.dump(yml_data, f, allow_unicode=True)
    logger.success("Step 1: initialization finished.")
    return True, "Шаг 1, Успех: Начальная настройка завершена"


def resample(model_name: str, normalize: bool, trim: bool, num_processes: int):
    logger.info("Step 2: start resampling...")
    dataset_path = get_path(model_name).dataset_path
    input_dir = dataset_path / "raw"
    output_dir = dataset_path / "wavs"
    cmd = [
        "resample.py",
        "-i",
        str(input_dir),
        "-o",
        str(output_dir),
        "--num_processes",
        str(num_processes),
        "--sr",
        "44100",
    ]
    if normalize:
        cmd.append("--normalize")
    if trim:
        cmd.append("--trim")
    success, message = run_script_with_log(cmd)
    if not success:
        logger.error("Step 2: resampling failed.")
        return False, f"Step 2, Error: 音声ファイルの前処理に失敗しました:\n{message}"
    elif message:
        logger.warning("Step 2: resampling finished with stderr.")
        return True, f"Step 2, Success: 音声ファイルの前処理が完了しました:\n{message}"
    logger.success("Step 2: resampling finished.")
    return True, "Шаг 2, Успех: Предобработка аудио завершена"


def preprocess_text(
    model_name: str, use_jp_extra: bool, val_per_lang: int, yomi_error: str
):
    logger.info("Step 3: start preprocessing text...")
    paths = get_path(model_name)
    if not paths.esd_path.exists():
        logger.error(f"Step 3: {paths.esd_path} not found.")
        return (
            False,
            f"Step 3, Error: 書き起こしファイル {paths.esd_path} が見つかりません。",
        )

    cmd = [
        "preprocess_text.py",
        "--config-path",
        str(paths.config_path),
        "--transcription-path",
        str(paths.esd_path),
        "--train-path",
        str(paths.train_path),
        "--val-path",
        str(paths.val_path),
        "--val-per-lang",
        str(val_per_lang),
        "--yomi_error",
        yomi_error,
        "--correct_path",  # 音声ファイルのパスを正しいパスに修正する
    ]
    if use_jp_extra:
        cmd.append("--use_jp_extra")
    success, message = run_script_with_log(cmd)
    if not success:
        logger.error("Step 3: preprocessing text failed.")
        return (
            False,
            f"Step 3, Error: 書き起こしファイルの前処理に失敗しました:\n{message}",
        )
    elif message:
        logger.warning("Step 3: preprocessing text finished with stderr.")
        return (
            True,
            f"Step 3, Success: 書き起こしファイルの前処理が完了しました:\n{message}",
        )
    logger.success("Step 3: preprocessing text finished.")
    return True, "Шаг 3, Успех: Предобработка текста завершена"


def bert_gen(model_name: str):
    logger.info("Step 4: start bert_gen...")
    config_path = get_path(model_name).config_path
    success, message = run_script_with_log(
        ["bert_gen.py", "--config", str(config_path)]
    )
    if not success:
        logger.error("Step 4: bert_gen failed.")
        return False, f"Step 4, Error: BERT特徴ファイルの生成に失敗しました:\n{message}"
    elif message:
        logger.warning("Step 4: bert_gen finished with stderr.")
        return (
            True,
            f"Step 4, Success: BERT特徴ファイルの生成が完了しました:\n{message}",
        )
    logger.success("Step 4: bert_gen finished.")
    return True, "Шаг 4, Успех: Генерация BERT-признаков завершена"


def style_gen(model_name: str, num_processes: int):
    logger.info("Step 5: start style_gen...")
    config_path = get_path(model_name).config_path
    success, message = run_script_with_log(
        [
            "style_gen.py",
            "--config",
            str(config_path),
            "--num_processes",
            str(num_processes),
        ]
    )
    if not success:
        logger.error("Step 5: style_gen failed.")
        return (
            False,
            f"Step 5, Error: スタイル特徴ファイルの生成に失敗しました:\n{message}",
        )
    elif message:
        logger.warning("Step 5: style_gen finished with stderr.")
        return (
            True,
            f"Step 5, Success: スタイル特徴ファイルの生成が完了しました:\n{message}",
        )
    logger.success("Step 5: style_gen finished.")
    return True, "Шаг 5, Успех: Генерация стилевых векторов завершена"


def preprocess_all(
    model_name: str,
    batch_size: int,
    epochs: int,
    save_every_steps: int,
    num_processes: int,
    normalize: bool,
    trim: bool,
    freeze_EN_bert: bool,
    freeze_JP_bert: bool,
    freeze_ZH_bert: bool,
    freeze_style: bool,
    freeze_decoder: bool,
    use_jp_extra: bool,
    val_per_lang: int,
    log_interval: int,
    yomi_error: str,
):
    if model_name == "":
        return False, "Ошибка: Введите имя модели"
    success, message = initialize(
        model_name=model_name,
        batch_size=batch_size,
        epochs=epochs,
        save_every_steps=save_every_steps,
        freeze_EN_bert=freeze_EN_bert,
        freeze_JP_bert=freeze_JP_bert,
        freeze_ZH_bert=freeze_ZH_bert,
        freeze_style=freeze_style,
        freeze_decoder=freeze_decoder,
        use_jp_extra=use_jp_extra,
        log_interval=log_interval,
    )
    if not success:
        return False, message
    success, message = resample(
        model_name=model_name,
        normalize=normalize,
        trim=trim,
        num_processes=num_processes,
    )
    if not success:
        return False, message

    success, message = preprocess_text(
        model_name=model_name,
        use_jp_extra=use_jp_extra,
        val_per_lang=val_per_lang,
        yomi_error=yomi_error,
    )
    if not success:
        return False, message
    success, message = bert_gen(
        model_name=model_name
    )  # bert_genは重いのでプロセス数いじらない
    if not success:
        return False, message
    success, message = style_gen(model_name=model_name, num_processes=num_processes)
    if not success:
        return False, message
    logger.success("Success: All preprocess finished!")
    return (
        True,
        "Успех: Все этапы предобработки завершены! Рекомендуется проверить консоль на наличие ошибок.",
    )


def train(
    model_name: str,
    skip_style: bool = False,
    use_jp_extra: bool = True,
    speedup: bool = False,
    not_use_custom_batch_sampler: bool = False,
):
    paths = get_path(model_name)
    # 学習再開の場合を考えて念のためconfig.ymlの名前等を更新
    with open("config.yml", encoding="utf-8") as f:
        yml_data = yaml.safe_load(f)
    yml_data["model_name"] = model_name
    yml_data["dataset_path"] = str(paths.dataset_path)
    with open("config.yml", "w", encoding="utf-8") as f:
        yaml.dump(yml_data, f, allow_unicode=True)

    train_py = "train_ms.py" if not use_jp_extra else "train_ms_jp_extra.py"
    cmd = [
        train_py,
        "--config",
        str(paths.config_path),
        "--model",
        str(paths.dataset_path),
    ]
    if skip_style:
        cmd.append("--skip_default_style")
    if speedup:
        cmd.append("--speedup")
    if not_use_custom_batch_sampler:
        cmd.append("--not_use_custom_batch_sampler")
    success, message = run_script_with_log(cmd, ignore_warning=True)
    if not success:
        logger.error("Train failed.")
        return False, f"Error: 学習に失敗しました:\n{message}"
    elif message:
        logger.warning("Train finished with stderr.")
        return True, f"Success: 学習が完了しました:\n{message}"
    logger.success("Train finished.")
    return True, "Успех: Обучение завершено"


def wait_for_tensorboard(port: int = 6006, timeout: float = 10):
    start_time = time.time()
    while True:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return True  # ポートが開いている場合
        except OSError:
            pass  # ポートがまだ開いていない場合

        if time.time() - start_time > timeout:
            return False  # タイムアウト

        time.sleep(0.1)


def run_tensorboard(model_name: str):
    global tensorboard_executed
    if not tensorboard_executed:
        python = sys.executable
        tensorboard_cmd = [
            python,
            "-m",
            "tensorboard.main",
            "--logdir",
            f"Data/{model_name}/models",
        ]
        subprocess.Popen(
            tensorboard_cmd,
            stdout=SAFE_STDOUT,  # type: ignore
            stderr=SAFE_STDOUT,  # type: ignore
        )
        yield gr.Button("起動中…")
        if wait_for_tensorboard():
            tensorboard_executed = True
        else:
            logger.error("Tensorboard did not start in the expected time.")
    webbrowser.open("http://localhost:6006")
    yield gr.Button("Открыть Tensorboard")


change_log_md = """
**Изменения с версии 2.5**

- Теперь стили создаются автоматически, если вы разделяете аудио по подпапкам внутри `raw/`. Подробности см. ниже в разделе «Подготовка данных».
- Раньше аудиофайлы длиннее 14 секунд не использовались. Теперь можно включить «Отключить кастомный батч-сэмплер», чтобы учиться на длинных файлах. Однако:
    - Обучение на длинных файлах может быть менее эффективным.
    - Это значительно увеличивает потребление VRAM. Если обучение падает, уменьшите размер батча или разбейте аудио на более короткие фрагменты.
"""

how_to_md = """
## Как использовать

- Подготовьте данные, введите имя модели, настройте параметры и нажмите «Запустить автоматическую предобработку». Прогресс будет виден в консоли.
- Если нужно запускать этапы по одному, используйте раздел «Ручная предобработка».
- После завершения предобработки нажмите «Начать обучение».
- Чтобы продолжить обучение с чекпоинта, просто введите то же имя модели и нажмите «Начать обучение».

## О версии JP-Extra

Можно использовать архитектуру [Bert-VITS2 Japanese-Extra](https://github.com/fishaudio/Bert-VITS2/releases/tag/JP-Exta).
Она улучшает японские акценты и интонации, но **отключает поддержку английского, китайского и русского языков**. Для русского обучения используйте СТАНДАРТНУЮ версию (снимите галочку JP-Extra).
"""

prepare_md = """
Сначала подготовьте аудиоданные и текст транскрипции.

Разместите их следующим образом:
```
├── Data/
│   ├── {Имя_модели}
│   │   ├── esd.list
│   │   ├── raw/
│   │   │   ├── foo.wav
│   │   │   ├── bar.mp3
│   │   │   ├── style1/
│   │   │   │   ├── baz.wav
│   │   │   │   ├── qux.wav
│   │   │   ├── style2/
│   │   │   │   ├── corge.wav
│   │   │   │   ├── grault.wav
...
```

### Как размещать файлы
- Если вы разместите файлы, как показано выше, то на основе папок `style1/` и `style2/` будут автоматически созданы стили с соответствующими именами (в дополнение к стандартному стилю).
- Если вам не нужно разделение по стилям, просто положите все файлы прямо в папку `raw/`. В этом случае будет создан только один стандартный стиль.
- Поддерживаются многие форматы, не только wav (например, mp3).

### Файл транскрипции `esd.list`

В файле `Data/{Имя_модели}/esd.list` опишите информацию о каждом аудиофайле в следующем формате:

```
относительный/путь/к/аудио.wav(даже если это mp3)|{Имя_диктора}|{Язык: RU, JP, ZH или EN}|{Текст}
```

- Путь `относительный/путь/к/аудио.wav` указывается относительно папки `raw/`. Например, для `raw/foo.wav` это будет `foo.wav`, для `raw/style1/bar.wav` это будет `style1/bar.wav`.
- Даже если расширение файла не wav, в `esd.list` пишите `wav` (например, для `raw/bar.mp3` пишите `bar.wav`).

Пример:
```
foo.wav|ann|RU|Привет, как дела?
bar.wav|ivan|RU|Да, я слышу тебя. Что-то случилось?
style1/baz.wav|ann|RU|Сегодня отличная погода.
style1/qux.wav|ivan|RU|Да, это точно.
...
english_teacher.wav|Mary|EN|How are you? I'm fine, thank you, and you?
...
```
Конечно, можно использовать датасет только с русским языком.
"""


def create_train_app():
    with gr.Blocks(theme=GRADIO_THEME).queue() as app:
        gr.Markdown(change_log_md)
        with gr.Accordion("Инструкция", open=False):
            gr.Markdown(how_to_md)
            with gr.Accordion(label="Подготовка данных", open=False):
                gr.Markdown(prepare_md)
        model_name = gr.Textbox(label="Имя модели")
        gr.Markdown("### 自動前処理")
        with gr.Row(variant="panel"):
            with gr.Column():
                use_jp_extra = gr.Checkbox(
                    label="Использовать JP-Extra (Улучшает японский, но отключает RU/EN/ZH. Для русского ОБУЧЕНИЯ — СНИМИТЕ ГАЛОЧКУ)",
                    value=False,
                )
                batch_size = gr.Slider(
                    label="Размер батча",
                    info="Уменьшите, если не хватает VRAM. Увеличьте, если памяти много. Примерный расход VRAM для JP-Extra: 1: 6GB, 2: 8GB, 4: 12GB",
                    value=2,
                    minimum=1,
                    maximum=64,
                    step=1,
                )
                epochs = gr.Slider(
                    label="Количество эпох",
                    info="100 обычно достаточно, но больше может улучшить качество",
                    value=100,
                    minimum=10,
                    maximum=1000,
                    step=10,
                )
                save_every_steps = gr.Slider(
                    label="Интервал сохранения (шагов)",
                    info="Не путайте с эпохами",
                    value=1000,
                    minimum=100,
                    maximum=10000,
                    step=100,
                )
                normalize = gr.Checkbox(
                    label="Нормализовать громкость (если аудио слишком разное по громкости)",
                    value=False,
                )
                trim = gr.Checkbox(
                    label="Удалить тишину в начале и конце аудио",
                    value=False,
                )
                yomi_error = gr.Radio(
                    label="Действие при ошибке чтения текста",
                    choices=[
                        ("Остановить препроцессинг при ошибке", "raise"),
                        ("Пропустить файл и продолжить", "skip"),
                        ("Пытаться прочитать и использовать", "use"),
                    ],
                    value="skip",
                )
                with gr.Accordion("Дополнительные настройки", open=False):
                    num_processes = gr.Slider(
                        label="Количество процессов",
                        info="Количество потоков для предобработки. Уменьшите, если система зависает.",
                        value=cpu_count() // 2,
                        minimum=1,
                        maximum=cpu_count(),
                        step=1,
                    )
                    val_per_lang = gr.Slider(
                        label="Количество валидационных данных",
                        info="Данные для сравнения в Tensorboard, не используются при обучении",
                        value=0,
                        minimum=0,
                        maximum=100,
                        step=1,
                    )
                    log_interval = gr.Slider(
                        label="Интервал логов Tensorboard",
                        info="Частота вывода логов в Tensorboard",
                        value=200,
                        minimum=10,
                        maximum=1000,
                        step=10,
                    )
                    gr.Markdown("Заморозка весов при обучении")
                    freeze_EN_bert = gr.Checkbox(
                        label="Заморозить EN BERT",
                        value=False,
                    )
                    freeze_JP_bert = gr.Checkbox(
                        label="Заморозить JP BERT",
                        value=False,
                    )
                    freeze_ZH_bert = gr.Checkbox(
                        label="Заморозить ZH BERT",
                        value=False,
                    )
                    freeze_style = gr.Checkbox(
                        label="Заморозить стиль",
                        value=False,
                    )
                    freeze_decoder = gr.Checkbox(
                        label="Заморозить декодер",
                        value=False,
                    )

            with gr.Column():
                preprocess_button = gr.Button(
                    value="Запустить автоматическую предобработку", variant="primary"
                )
                info_all = gr.Textbox(label="Статус")
        with gr.Accordion(open=False, label="Ручная предобработка"):
            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(value="#### Шаг 1: Генерация конфигурации")
                    use_jp_extra_manual = gr.Checkbox(
                        label="Использовать JP-Extra",
                        value=False,
                    )
                    batch_size_manual = gr.Slider(
                        label="Размер батча",
                        value=2,
                        minimum=1,
                        maximum=64,
                        step=1,
                    )
                    epochs_manual = gr.Slider(
                        label="Количество эпох",
                        value=100,
                        minimum=1,
                        maximum=1000,
                        step=1,
                    )
                    save_every_steps_manual = gr.Slider(
                        label="Интервал сохранения (шагов)",
                        value=1000,
                        minimum=100,
                        maximum=10000,
                        step=100,
                    )
                    log_interval_manual = gr.Slider(
                        label="Интервал логов Tensorboard",
                        value=200,
                        minimum=10,
                        maximum=1000,
                        step=10,
                    )
                    freeze_EN_bert_manual = gr.Checkbox(
                        label="Заморозить EN BERT",
                        value=False,
                    )
                    freeze_JP_bert_manual = gr.Checkbox(
                        label="Заморозить JP BERT",
                        value=False,
                    )
                    freeze_ZH_bert_manual = gr.Checkbox(
                        label="Заморозить ZH BERT",
                        value=False,
                    )
                    freeze_style_manual = gr.Checkbox(
                        label="Заморозить стиль",
                        value=False,
                    )
                    freeze_decoder_manual = gr.Checkbox(
                        label="Заморозить декодер",
                        value=False,
                    )
                with gr.Column():
                    generate_config_btn = gr.Button(value="Запустить (Шаг 1)", variant="primary")
                    info_init = gr.Textbox(label="Статус")
            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(value="#### Шаг 2: Ресемплинг/Предобработка аудио")
                    num_processes_resample = gr.Slider(
                        label="Количество процессов",
                        value=cpu_count() // 2,
                        minimum=1,
                        maximum=cpu_count(),
                        step=1,
                    )
                    normalize_resample = gr.Checkbox(
                        label="Нормализовать громкость",
                        value=False,
                    )
                    trim_resample = gr.Checkbox(
                        label="Удалить тишину",
                        value=False,
                    )
                with gr.Column():
                    resample_btn = gr.Button(value="Запустить (Шаг 2)", variant="primary")
                    info_resample = gr.Textbox(label="Статус")
            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(value="#### Шаг 3: Предобработка текста")
                    val_per_lang_manual = gr.Slider(
                        label="Количество валидационных данных",
                        value=0,
                        minimum=0,
                        maximum=100,
                        step=1,
                    )
                    yomi_error_manual = gr.Radio(
                        label="Действие при ошибке чтения текста",
                        choices=[
                            ("Остановить препроцессинг при ошибке", "raise"),
                            ("Пропустить файл и продолжить", "skip"),
                            ("Пытаться прочитать и использовать", "use"),
                        ],
                        value="raise",
                    )
                with gr.Column():
                    preprocess_text_btn = gr.Button(value="Запустить (Шаг 3)", variant="primary")
                    info_preprocess_text = gr.Textbox(label="Статус")
            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(value="#### Шаг 4: Генерация BERT-признаков")
                with gr.Column():
                    bert_gen_btn = gr.Button(value="Запустить (Шаг 4)", variant="primary")
                    info_bert = gr.Textbox(label="Статус")
            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(value="#### Шаг 5: Генерация стилевых векторов")
                    num_processes_style = gr.Slider(
                        label="Количество процессов",
                        value=cpu_count() // 2,
                        minimum=1,
                        maximum=cpu_count(),
                        step=1,
                    )
                with gr.Column():
                    style_gen_btn = gr.Button(value="Запустить (Шаг 5)", variant="primary")
                    info_style = gr.Textbox(label="Статус")
        gr.Markdown("## Обучение")
        with gr.Row():
            skip_style = gr.Checkbox(
                label="Пропустить генерацию стилей",
                info="Отметьте, если продолжаете обучение",
                value=False,
            )
            use_jp_extra_train = gr.Checkbox(
                label="Использовать JP-Extra",
                value=False,
            )
            not_use_custom_batch_sampler = gr.Checkbox(
                label="Отключить кастомный батч-сэмплер",
                info="Если VRAM позволяет, позволяет использовать длинные аудио",
                value=False,
            )
            speedup = gr.Checkbox(
                label="Ускорить обучение (пропускать логи)",
                value=False,
                visible=False,  # Experimental
            )
            train_btn = gr.Button(value="Начать обучение", variant="primary")
            tensorboard_btn = gr.Button(value="Открыть Tensorboard")
        gr.Markdown(
            "Следите за прогрессом в терминале. Результаты сохраняются автоматически. Обучение можно остановить и продолжить позже. Чтобы завершить работу, просто закройте окно терминала."
        )
        info_train = gr.Textbox(label="Статус")

        preprocess_button.click(
            second_elem_of(preprocess_all),
            inputs=[
                model_name,
                batch_size,
                epochs,
                save_every_steps,
                num_processes,
                normalize,
                trim,
                freeze_EN_bert,
                freeze_JP_bert,
                freeze_ZH_bert,
                freeze_style,
                freeze_decoder,
                use_jp_extra,
                val_per_lang,
                log_interval,
                yomi_error,
            ],
            outputs=[info_all],
        )

        # Manual preprocess
        generate_config_btn.click(
            second_elem_of(initialize),
            inputs=[
                model_name,
                batch_size_manual,
                epochs_manual,
                save_every_steps_manual,
                freeze_EN_bert_manual,
                freeze_JP_bert_manual,
                freeze_ZH_bert_manual,
                freeze_style_manual,
                freeze_decoder_manual,
                use_jp_extra_manual,
                log_interval_manual,
            ],
            outputs=[info_init],
        )
        resample_btn.click(
            second_elem_of(resample),
            inputs=[
                model_name,
                normalize_resample,
                trim_resample,
                num_processes_resample,
            ],
            outputs=[info_resample],
        )
        preprocess_text_btn.click(
            second_elem_of(preprocess_text),
            inputs=[
                model_name,
                use_jp_extra_manual,
                val_per_lang_manual,
                yomi_error_manual,
            ],
            outputs=[info_preprocess_text],
        )
        bert_gen_btn.click(
            second_elem_of(bert_gen),
            inputs=[model_name],
            outputs=[info_bert],
        )
        style_gen_btn.click(
            second_elem_of(style_gen),
            inputs=[model_name, num_processes_style],
            outputs=[info_style],
        )

        # Train
        train_btn.click(
            second_elem_of(train),
            inputs=[
                model_name,
                skip_style,
                use_jp_extra_train,
                speedup,
                not_use_custom_batch_sampler,
            ],
            outputs=[info_train],
        )
        tensorboard_btn.click(
            run_tensorboard, inputs=[model_name], outputs=[tensorboard_btn]
        )

        use_jp_extra.change(
            lambda x: gr.Checkbox(value=x),
            inputs=[use_jp_extra],
            outputs=[use_jp_extra_train],
        )
        use_jp_extra_manual.change(
            lambda x: gr.Checkbox(value=x),
            inputs=[use_jp_extra_manual],
            outputs=[use_jp_extra_train],
        )

    return app


if __name__ == "__main__":
    app = create_train_app()
    app.launch(inbrowser=True)
