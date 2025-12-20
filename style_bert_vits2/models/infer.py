import torch
import numpy as np
from style_bert_vits2.models import utils
from style_bert_vits2.models.models import SynthesizerTrn
from style_bert_vits2.nlp import extract_bert_feature
from style_bert_vits2.constants import Languages

def get_net_g(model_path: str, version: str, device: str, hps):
    if version.endswith("JP-Extra"):
        from style_bert_vits2.models.models_jp_extra import SynthesizerTrn
    else:
        from style_bert_vits2.models.models import SynthesizerTrn

    net_g = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to(device)
    _ = net_g.eval()

    # Загрузка весов. strict=False важно, если вы добавляете новый слой в уже обученную базу
    utils.checkpoints.load_checkpoint(model_path, net_g, None, skip_optimizer=True)
    return net_g

def infer(
    text,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    sid,
    language,
    hps,
    net_g,
    device,
    assist_text=None,
    assist_text_weight=0.7,
    style_vec=None,
    given_phone=None,
    given_tone=None,
):
    # Подготовка BERT фичей
    bert, ja_bert, en_bert, ru_bert = None, None, None, None

    # Получаем word2ph (выравнивание)
    # Здесь предполагается, что phonemes/tokens уже обработаны снаружи или внутри extract_bert_feature
    # В Style-Bert-VITS2 обычно phone/tone/word2ph генерируются до вызова infer в tts_model.py,
    # но extract_bert_feature требует word2ph.

    # Примечание: В tts_model.py обычно текст уже нормализован и разбит на фонемы.
    # Но для extract_bert_feature нам нужно передать оригинальный текст и word2ph.
    # Поскольку infer вызывается уже с готовыми phones (внутри net_g.infer они не нужны явно, кроме как для длины),
    # нам нужно пересчитать word2ph, если они не переданы.

    # В текущей архитектуре infer принимает `text` (строку).
    # tts_model.py делает clean_text и получает phones, tones, word2ph.
    # Нам нужно достать word2ph.

    # ВАЖНО: tts_model.py в методе infer (PyTorch) вызывает clean_text и передает данные сюда?
    # Нет, tts_model.py (в вашем файле) вызывает models.infer.infer напрямую.
    # Но models.infer.infer НЕ принимает word2ph как аргумент. Это архитектурный нюанс.
    # Обычно extract_bert_feature вызывается ВНУТРИ tts_model.py, но здесь он вызывается ВНУТРИ models/infer.py?

    # Давайте посмотрим, как это сделано в оригинале.
    # В оригинале models/infer.py делает clean_text заново.

    from style_bert_vits2.nlp import clean_text

    if given_phone is None:
        phones, tones, lang_ids, word2ph = clean_text(text, language)
    else:
        # Если фонемы даны вручную (редкий кейс)
        phones = given_phone
        tones = given_tone
        if given_tone is None:
            tones = [0] * len(phones)
        lang_ids = [0] * len(phones) # Заглушка
        # Пытаемся угадать word2ph (1 к 1)
        word2ph = [1] * len(phones)

    phone_tensor = torch.LongTensor(phones).to(device).unsqueeze(0)
    tone_tensor = torch.LongTensor(tones).to(device).unsqueeze(0)
    language_tensor = torch.LongTensor(lang_ids).to(device).unsqueeze(0)

    # Инициализируем нулями
    bert = torch.zeros(1024, len(phones)).to(device)
    ja_bert = torch.zeros(1024, len(phones)).to(device)
    en_bert = torch.zeros(1024, len(phones)).to(device)
    ru_bert = torch.zeros(768, len(phones)).to(device) # RU имеет 768!

    # Извлечение BERT фичей в зависимости от языка
    if language == Languages.ZH:
        bert = extract_bert_feature(
            text, word2ph, device, assist_text, assist_text_weight
        ).to(device)
    elif language == Languages.JP:
        ja_bert = extract_bert_feature(
            text, word2ph, device, assist_text, assist_text_weight
        ).to(device)
    elif language == Languages.EN:
        en_bert = extract_bert_feature(
            text, word2ph, device, assist_text, assist_text_weight
        ).to(device)
    elif language == Languages.RU:
        # --- RU BLOCK ---
        ru_bert = extract_bert_feature(
            text, word2ph, device, assist_text, assist_text_weight
        ).to(device)
        # ----------------

    bert_tensor = bert.unsqueeze(0)
    ja_bert_tensor = ja_bert.unsqueeze(0)
    en_bert_tensor = en_bert.unsqueeze(0)
    ru_bert_tensor = ru_bert.unsqueeze(0)

    x_lengths = torch.LongTensor([phone_tensor.size(1)]).to(device)

    if style_vec is None:
        # Если стиль не передан, берем средний (обычно он есть в модели, но тут создадим нулевой или случайный для безопасности)
        # В продакшене tts_model.py всегда передает style_vec
        style_vec = torch.zeros(1, 256).to(device)
    else:
        style_vec = torch.from_numpy(style_vec).to(device).unsqueeze(0)

    audio = (
        net_g.infer(
            phone_tensor,
            x_lengths,
            sid,
            tone_tensor,
            language_tensor,
            bert_tensor,
            ja_bert_tensor,
            en_bert_tensor,
            ru_bert_tensor, # <-- ПЕРЕДАЕМ RU BERT
            style_vec,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=length_scale,
            sdp_ratio=sdp_ratio,
        )[0][0, 0]
        .data.cpu()
        .float()
        .numpy()
    )

    return audio
