from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import onnxruntime
from numpy.typing import NDArray

from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import bert_models, onnx_bert_models
from style_bert_vits2.utils import get_onnx_device_options


if TYPE_CHECKING:
    import torch


def extract_bert_feature(
    text: str,
    word2ph: list[int],
    device: str,
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
) -> torch.Tensor:
    """
    Русский текст: извлечение признаков BERT (PyTorch)

    Args:
        text (str): Русский текст
        word2ph (list[int]): Список, показывающий, сколько фонем соответствует каждому токену
        device (str): Устройство для инференса (cuda/cpu)
        assist_text (Optional[str], optional): Вспомогательный текст для стиля (по умолчанию: None)
        assist_text_weight (float, optional): Вес вспомогательного текста (по умолчанию: 0.7)

    Returns:
        torch.Tensor: Признаки BERT
    """

    import torch

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # ИЗМЕНЕНИЕ: Загружаем русскую модель (Languages.RU)
    model = bert_models.load_model(Languages.RU, device_map=device)
    bert_models.transfer_model(Languages.RU, device)

    style_res_mean = None
    with torch.no_grad():
        # ИЗМЕНЕНИЕ: Загружаем русский токенизатор
        tokenizer = bert_models.load_tokenizer(Languages.RU)
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)  # type: ignore

        # Получаем выход модели вместе со скрытыми состояниями
        res = model(**inputs, output_hidden_states=True)

        # Берем предпоследний слой (стандартная практика для VITS2)
        # res["hidden_states"][-3:-2] берет слой перед последним
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()

        if assist_text:
            style_inputs = tokenizer(assist_text, return_tensors="pt")
            for i in style_inputs:
                style_inputs[i] = style_inputs[i].to(device)  # type: ignore
            style_res = model(**style_inputs, output_hidden_states=True)
            style_res = torch.cat(style_res["hidden_states"][-3:-2], -1)[0].cpu()
            style_res_mean = style_res.mean(0)

    # Проверка совпадения длины токенов и word2ph
    assert len(word2ph) == res.shape[0], (text, res.shape[0], len(word2ph))

    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        if assist_text:
            assert style_res_mean is not None
            repeat_feature = (
                res[i].repeat(word2phone[i], 1) * (1 - assist_text_weight)
                + style_res_mean.repeat(word2phone[i], 1) * assist_text_weight
            )
        else:
            repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    return phone_level_feature.T


def extract_bert_feature_onnx(
    text: str,
    word2ph: list[int],
    onnx_providers: Sequence[Union[str, tuple[str, dict[str, Any]]]],
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
) -> NDArray[Any]:
    """
    Русский текст: извлечение признаков BERT (ONNX)
    Примечание: Для работы этого метода необходимо конвертировать модель RuBERT в ONNX
    и прописать путь в constants.py
    """

    # ИЗМЕНЕНИЕ: Загрузка ONNX модели и токенизатора для RU
    tokenizer = onnx_bert_models.load_tokenizer(Languages.RU)
    session = onnx_bert_models.load_model(
        language=Languages.RU,
        onnx_providers=onnx_providers,
    )
    input_names = [input.name for input in session.get_inputs()]
    output_name = session.get_outputs()[0].name

    device_type, device_id, run_options = get_onnx_device_options(session, onnx_providers)  # fmt: skip

    inputs = tokenizer(text, return_tensors="np")
    input_tensor = [
        inputs["input_ids"].astype(np.int64),  # type: ignore
        inputs["attention_mask"].astype(np.int64),  # type: ignore
    ]

    # Привязка ввода/вывода (IO Binding) для оптимизации
    io_binding = session.io_binding()
    for name, value in zip(input_names, input_tensor):
        gpu_tensor = onnxruntime.OrtValue.ortvalue_from_numpy(
            value, device_type, device_id
        )
        io_binding.bind_ortvalue_input(name, gpu_tensor)

    io_binding.bind_output(output_name, device_type)
    session.run_with_iobinding(io_binding, run_options=run_options)
    res = io_binding.get_outputs()[0].numpy()

    style_res_mean = None
    if assist_text:
        style_inputs = tokenizer(assist_text, return_tensors="np")
        style_input_tensor = [
            style_inputs["input_ids"].astype(np.int64),  # type: ignore
            style_inputs["attention_mask"].astype(np.int64),  # type: ignore
        ]

        io_binding = session.io_binding()
        for name, value in zip(input_names, style_input_tensor):
            gpu_tensor = onnxruntime.OrtValue.ortvalue_from_numpy(
                value, device_type, device_id
            )
            io_binding.bind_ortvalue_input(name, gpu_tensor)

        io_binding.bind_output(output_name, device_type)
        session.run_with_iobinding(io_binding, run_options=run_options)
        style_res = io_binding.get_outputs()[0].numpy()
        style_res_mean = np.mean(style_res, axis=0)

    assert len(word2ph) == res.shape[0], (text, res.shape[0], len(word2ph))
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        if assist_text:
            assert style_res_mean is not None
            repeat_feature = (
                np.tile(res[i], (word2phone[i], 1)) * (1 - assist_text_weight)
                + np.tile(style_res_mean, (word2phone[i], 1)) * assist_text_weight
            )
        else:
            repeat_feature = np.tile(res[i], (word2phone[i], 1))
        phone_level_feature.append(repeat_feature)

    phone_level_feature = np.concatenate(phone_level_feature, axis=0)

    return phone_level_feature.T
