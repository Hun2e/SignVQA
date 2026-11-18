"""PaddleOCR-VL-1.6 Spotting 엔진.

실험 A(노트북)에서 검증된 공식 전처리/생성 설정을 그대로 함수로 뽑아둔 것입니다.
OCR 캐시 생성과 이후 실험이 같은 코드를 쓰도록 여기 한 곳에만 둡니다.

공식 근거: https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6
  - task="spotting" -> prompt "Spotting:"
  - 두 변이 모두 1500 미만이면 2배 확대(LANCZOS)
  - max_pixels = 2048*28*28 (spotting 전용 값)
"""
from pathlib import Path

import torch
from PIL import Image

OCR_MODEL_DIR = Path("downloads/models/PaddleOCR-VL-1.6")
OCR_MAX_PIXELS = 2048 * 28 * 28   # 공식 spotting 값 = 1,605,632
SPOTTING_UPSCALE_THRESHOLD = 1500
SPOTTING_PROMPT = "Spotting:"


def spotting_image(image_path):
    """공식 spotting 전처리. 공통 960px 축소 규칙을 적용하지 않습니다."""
    with Image.open(image_path) as im:
        im = im.convert("RGB")
    if im.width < SPOTTING_UPSCALE_THRESHOLD and im.height < SPOTTING_UPSCALE_THRESHOLD:
        im = im.resize((im.width * 2, im.height * 2), Image.Resampling.LANCZOS)
    return im


def _read_pixel_bounds(ip):
    """transformers 5.x 내장 클래스는 size가 SizeDict(shortest_edge/longest_edge = 픽셀 수),
    모델 폴더의 remote code 클래스는 min_pixels/max_pixels 속성을 씁니다."""
    size = getattr(ip, "size", None)

    def pick(*keys):
        for key in keys:
            value = size.get(key) if isinstance(size, dict) else getattr(size, key, None)
            if value:
                return int(value)
        return None

    low = pick("shortest_edge", "min_pixels") or getattr(ip, "min_pixels", None)
    high = pick("longest_edge", "max_pixels") or getattr(ip, "max_pixels", None)
    if not low or not high:
        raise ValueError(f"OCR 픽셀 설정을 찾지 못했습니다: {type(ip).__name__}, size={size}")
    return int(low), int(high)


def _probe_pixels(proc, edge=2048):
    """전처리를 실제로 한 번 돌려 결과 픽셀 수를 측정합니다."""
    grid = proc.image_processor(images=Image.new("RGB", (edge, edge)),
                               return_tensors="pt")["image_grid_thw"]
    patches = int(grid[0][1]) * int(grid[0][2])
    return patches * proc.image_processor.patch_size ** 2


def configure_ocr_pixels(proc, max_pixels=OCR_MAX_PIXELS, verbose=True):
    """픽셀 상한을 spotting 값으로 올리고, 실제로 적용됐는지 실측으로 검증합니다.

    apply_chat_template(images_kwargs=...)는 transformers 5.x에서 무시되므로
    프로세서 객체에 직접 설정합니다.
    """
    ip = proc.image_processor
    min_pixels, default_max = _read_pixel_bounds(ip)
    if not 0 < min_pixels <= max_pixels:
        raise ValueError(f"잘못된 OCR 픽셀 범위: {min_pixels}..{max_pixels}")
    before = _probe_pixels(proc)
    size = getattr(ip, "size", None)
    if isinstance(size, dict):
        updated = dict(size)
        keys = ("shortest_edge", "longest_edge") if size.get("shortest_edge") else ("min_pixels", "max_pixels")
        updated[keys[0]], updated[keys[1]] = min_pixels, max_pixels
        ip.size = updated
    elif size is not None and getattr(size, "shortest_edge", None):
        size.shortest_edge, size.longest_edge = min_pixels, max_pixels
    if hasattr(ip, "min_pixels"):
        ip.min_pixels = min_pixels
    if hasattr(ip, "max_pixels"):
        ip.max_pixels = max_pixels
    after = _probe_pixels(proc)
    if not before < after <= max_pixels:
        raise RuntimeError(
            f"OCR 픽셀 상한이 적용되지 않았습니다: 2048px 입력이 {before} -> {after} 픽셀 "
            f"(기대: {default_max} 초과, {max_pixels} 이하). {type(ip).__name__}")
    if verbose:
        print(f"OCR 픽셀 범위 {min_pixels}~{max_pixels} 적용 확인 "
              f"(2048px 입력: {before} -> {after} 픽셀) / {type(ip).__name__}")
    return min_pixels


def load_ocr(model_dir=OCR_MODEL_DIR, device="cuda:0", dtype=None):
    from transformers import AutoProcessor, AutoModelForImageTextToText

    model_dir = Path(model_dir)
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"로컬 모델을 준비하세요: {model_dir}")
    if dtype is None:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    proc = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_dir), local_files_only=True, dtype=dtype).to(device).eval()
    configure_ocr_pixels(proc)
    return model, proc


@torch.inference_mode()
def run_spotting(model, proc, image, max_new_tokens):
    """이미지 한 장의 Spotting 결과와 길이 제한 도달 여부를 반환합니다."""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": SPOTTING_PROMPT}]}]
    inputs = proc.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to(model.device)
    prefix_len = inputs["input_ids"].shape[-1]
    # 이 모델의 generation_config는 use_cache=False라서 매 스텝 전체를 다시 계산합니다.
    # greedy이므로 캐시를 켜도 생성 토큰은 동일하고 속도만 회복됩니다.
    output = model.generate(**inputs, max_new_tokens=max_new_tokens,
                            do_sample=False, use_cache=True)
    new = output[0, prefix_len:]
    eos = model.generation_config.eos_token_id
    eos_ids = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
    truncated = len(new) >= max_new_tokens and (not len(new) or int(new[-1]) not in eos_ids)
    text = proc.decode(new, skip_special_tokens=True).strip()
    return text, bool(truncated), int(len(new))


@torch.inference_mode()
def run_spotting_batch(model, proc, images, max_new_tokens):
    """Return one (text, truncated, tokens) tuple per image, in input order."""
    import inspect
    if not images:
        return []
    proc.tokenizer.padding_side = "left"
    messages = [[{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": SPOTTING_PROMPT}]}] for image in images]
    padding = ({"processor_kwargs": {"padding": True}}
               if "processor_kwargs" in inspect.signature(proc.apply_chat_template).parameters
               else {"padding": True})
    inputs = proc.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt", **padding).to(model.device)
    prefix_len = inputs["input_ids"].shape[-1]
    output = model.generate(**inputs, max_new_tokens=max_new_tokens,
                            do_sample=False, use_cache=True)
    eos = model.generation_config.eos_token_id
    eos_ids = set([] if eos is None else ([eos] if isinstance(eos, int) else eos))
    results = []
    for sequence in output[:, prefix_len:].tolist():
        end = next((i + 1 for i, token in enumerate(sequence) if token in eos_ids), None)
        tokens = sequence if end is None else sequence[:end]
        results.append((proc.decode(tokens, skip_special_tokens=True).strip(),
                        end is None and len(tokens) >= max_new_tokens, len(tokens)))
    if len(results) != len(images):
        raise RuntimeError("배치 입력/출력 개수가 다릅니다.")
    return results
