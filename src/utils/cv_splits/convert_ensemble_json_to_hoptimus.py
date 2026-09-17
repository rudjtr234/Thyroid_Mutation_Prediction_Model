# -*- coding: utf-8 -*-
"""
UNI2-H 앙상블 fold 분할 JSON(ensemble_5models_cv.json, test_set.json)을
H-optimus-0 임베딩 경로로 변환하는 1회성 스크립트.

WSI positive/negative 구성과 fold 분할(파일명 리스트)은 그대로 유지하고,
uni2_embeddings 디렉토리 경로 문자열만 DIR_MAP을 통해 h-optimus-0 경로로 치환한다.
uni2_embeddings와 h_optimus_embeddings는 meta/non_meta 명명 규칙이 다르므로
(예: nonmeta vs non_meta, 버전 번호 상이) "uni2_embeddings" -> "h_optimus_embeddings"
같은 부분 문자열 치환은 쓰지 않고, 알려진 dir 전체 값 단위로만 매핑한다.
JSON 구조(어느 필드에 dir이 있는지)에 의존하지 않도록 전체를 재귀 순회하며 값
단위로 매핑하므로, 최상위 요약 필드나 레거시 필드에 있어도 놓치지 않는다.

사용법:
    python convert_ensemble_json_to_hoptimus.py \
        --ensemble_json /path/to/dataset/braf_ensemble_meta_data/ensemble_5models_cv.json \
        --test_json /path/to/dataset/braf_ensemble_meta_data/test_set.json \
        --out_dir /path/to/dataset/braf_ensemble_meta_data/

DIR_MAP의 키(uni2 쪽 실제 dir 값)는 처음 실행 시 실제 JSON을 읽어 확인 후 채운다.
매핑에 없는 dir을 만나면 즉시 에러를 던지므로, 에러 메시지에 출력된 실제 경로를
DIR_MAP에 추가하면 된다.
"""

import argparse
import json
import sys
from pathlib import Path

# uni2 dir -> h-optimus-0 dir 명시적 매핑.
# 처음 실행 시 UnmappedDirError로 실제 uni2 dir 값이 출력되면 여기에 채워 넣는다.
DIR_MAP = {
    "/path/to/embeddings/meta/npy":
        "/path/to/embeddings/meta/npy",
    "/path/to/embeddings/nonmeta/npy":
        "/path/to/embeddings/nonmeta/npy",
}


class UnmappedDirError(RuntimeError):
    pass


def map_dir(uni2_dir: str) -> str:
    if uni2_dir not in DIR_MAP:
        raise UnmappedDirError(
            f"DIR_MAP에 없는 uni2 경로: {uni2_dir!r}\n"
            f"이 경로를 h-optimus-0 대응 경로와 함께 DIR_MAP에 추가하세요."
        )
    return DIR_MAP[uni2_dir]


def convert_paths_recursive(obj):
    """
    JSON 전체를 재귀적으로 순회하며, uni2_embeddings 경로 문자열을 전부 찾아 치환한다.
    (알려진 필드 경로(models[i].train.positive.dir 등)만 짚어가는 방식은 최상위 요약
    필드(meta_dir/nonmeta_dir 등)나 레거시 필드를 놓칠 수 있어, 구조에 의존하지 않는
    전수 순회 + 개별 문자열 매핑 방식으로 변경했다. 파일명 리스트(files)의 각 항목은
    디렉토리 경로가 아니라 "TC_04_xxxx.npy" 형태라 DIR_MAP에 없어 그대로 통과한다.)
    """
    if isinstance(obj, dict):
        return {k: convert_paths_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_paths_recursive(v) for v in obj]
    if isinstance(obj, str) and "uni2_embeddings" in obj:
        return map_dir(obj)
    return obj


def assert_no_uni2_leftover(data: dict, label: str):
    dumped = json.dumps(data)
    if "uni2_embeddings" in dumped:
        raise RuntimeError(f"{label}: 변환 후에도 uni2_embeddings 경로가 남아있음 — 치환 누락")


def assert_same_structure_except_mapped_strings(original, converted, label: str):
    """
    구조(dict 키 집합, list 길이/순서)가 완전히 동일하고, 값이 달라진 곳은
    전부 uni2_embeddings 문자열이 DIR_MAP을 통해 h-optimus 경로로 바뀐 경우여야 한다.
    파일명 리스트 등 다른 값이 하나라도 달라지면 실패 — WSI 구성이 바뀌었다는 뜻이다.
    """
    def walk(o, c, path):
        if type(o) is not type(c):
            raise RuntimeError(f"{label}: {path} 타입이 원본과 다름 ({type(o)} -> {type(c)})")

        if isinstance(o, dict):
            if o.keys() != c.keys():
                raise RuntimeError(f"{label}: {path} 키 집합이 원본과 다름")
            for k in o:
                walk(o[k], c[k], f"{path}.{k}")
        elif isinstance(o, list):
            if len(o) != len(c):
                raise RuntimeError(f"{label}: {path} 리스트 길이가 원본과 다름")
            for i, (ov, cv) in enumerate(zip(o, c)):
                walk(ov, cv, f"{path}[{i}]")
        elif o != c:
            if not (isinstance(o, str) and "uni2_embeddings" in o and c == DIR_MAP.get(o)):
                raise RuntimeError(
                    f"{label}: {path} 값이 uni2->h-optimus 매핑이 아닌 방식으로 변경됨 "
                    f"({o!r} -> {c!r})"
                )

    walk(original, converted, "$")


def main():
    parser = argparse.ArgumentParser(description="UNI2-H ensemble split JSON -> H-optimus-0 경로 변환")
    parser.add_argument("--ensemble_json", type=str, required=True)
    parser.add_argument("--test_json", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    with open(args.ensemble_json) as f:
        ensemble_orig = json.load(f)
    with open(args.test_json) as f:
        test_orig = json.load(f)

    ensemble_converted = convert_paths_recursive(ensemble_orig)
    test_converted = convert_paths_recursive(test_orig)

    assert_same_structure_except_mapped_strings(ensemble_orig, ensemble_converted, "ensemble_5models_cv.json")
    assert_same_structure_except_mapped_strings(test_orig, test_converted, "test_set.json")
    assert_no_uni2_leftover(ensemble_converted, "ensemble_5models_cv.json")
    assert_no_uni2_leftover(test_converted, "test_set.json")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ensemble_out = out_dir / "ensemble_5models_cv_hoptimus0.json"
    test_out = out_dir / "test_set_hoptimus0.json"

    with open(ensemble_out, "w") as f:
        json.dump(ensemble_converted, f, indent=2)
    with open(test_out, "w") as f:
        json.dump(test_converted, f, indent=2)

    print(f"[✓] 변환 완료: {ensemble_out}")
    print(f"[✓] 변환 완료: {test_out}")


if __name__ == "__main__":
    try:
        main()
    except UnmappedDirError as e:
        print(f"[!] {e}", file=sys.stderr)
        sys.exit(1)
