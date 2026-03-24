import json
from collections import Counter
import numpy as np

def get_label_from_name(filename):
    """경로 문자열에서 meta / nonmeta 라벨 판별"""
    lower = filename.lower()
    if "/nonmeta/" in lower:
        return 0  # Negative
    elif "/meta/" in lower:
        return 1  # Positive
    else:
        raise ValueError(f"Unknown label in filename: {filename}")


def validate_cv_splits(json_path):
    """CV splits 검증: 중복, 라벨 분포, Stratified 체크"""
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    print("\n" + "="*80)
    print("📊 CV Splits Validation Report")
    print("="*80)
    
    # =========================
    # 1. Fold별 파일 수 체크
    # =========================
    print("\n1️⃣  Fold별 파일 수 체크")
    print("-"*80)
    
    for fold in data["folds"]:
        fold_id = fold["fold"]
        train_files = fold["train_wsis"]
        val_files = fold["val_wsis"]
        test_files = fold["test_wsis"]
        
        # Fold 내부 중복 체크
        all_in_fold = train_files + val_files + test_files
        counts = Counter(all_in_fold)
        dupes = [f for f, c in counts.items() if c > 1]
        
        if dupes:
            print(f"❌ Fold {fold_id}: 내부 중복 {len(dupes)}개 발견!")
            for dup in dupes[:3]:
                print(f"     - {dup}")
        else:
            print(f"✅ Fold {fold_id}: 내부 중복 없음")
        
        print(f"   Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}, "
              f"Total={len(all_in_fold)}")
    
    # =========================
    # 2. ✅ Test Set 간 중복 체크 (가장 중요!)
    # =========================
    print("\n2️⃣  Test Set 간 중복 체크 (핵심!)")
    print("-"*80)
    
    test_sets = []
    for fold in data["folds"]:
        test_set = set(fold["test_wsis"])
        test_sets.append(test_set)
    
    test_overlap_found = False
    for i in range(len(test_sets)):
        for j in range(i + 1, len(test_sets)):
            overlap = test_sets[i] & test_sets[j]
            if overlap:
                print(f"❌ Fold {i+1} ↔ Fold {j+1} Test overlap: {len(overlap)}개")
                test_overlap_found = True
    
    if not test_overlap_found:
        print("✅ 모든 Fold의 Test set이 완전히 독립적!")
    
    # Test set 커버리지 확인
    all_test_files = set()
    for test_set in test_sets:
        all_test_files.update(test_set)
    
    print(f"\n모든 Test set 합: {len(all_test_files)}개")
    print(f"각 Fold Test: {len(test_sets[0])}개 × {len(test_sets)}개 = {len(test_sets[0]) * len(test_sets)}개")
    
    if len(all_test_files) == len(test_sets[0]) * len(test_sets):
        print("✅ 모든 Test 파일이 중복 없이 정확히 사용됨!")
    else:
        print(f"⚠️ Test 파일 중복 또는 누락 ({len(all_test_files)}개 != {len(test_sets[0]) * len(test_sets)}개)")
    
    # =========================
    # 3. Train/Val/Test 간 Overlap 체크 (각 Fold 내부)
    # =========================
    print("\n3️⃣  각 Fold 내 Train/Val/Test Overlap 체크")
    print("-"*80)
    
    for fold in data["folds"]:
        fold_id = fold["fold"]
        train_set = set(fold["train_wsis"])
        val_set = set(fold["val_wsis"])
        test_set = set(fold["test_wsis"])
        
        train_val_overlap = train_set & val_set
        train_test_overlap = train_set & test_set
        val_test_overlap = val_set & test_set
        
        if train_val_overlap or train_test_overlap or val_test_overlap:
            print(f"❌ Fold {fold_id}:")
            if train_val_overlap:
                print(f"     Train-Val overlap: {len(train_val_overlap)}개")
            if train_test_overlap:
                print(f"     Train-Test overlap: {len(train_test_overlap)}개")
            if val_test_overlap:
                print(f"     Val-Test overlap: {len(val_test_overlap)}개")
        else:
            print(f"✅ Fold {fold_id}: Train/Val/Test 간 overlap 없음")
    
    # =========================
    # 4. Label Distribution (Stratified 확인)
    # =========================
    print("\n4️⃣  Label Distribution (Stratified 확인)")
    print("-"*80)
    
    for fold in data["folds"]:
        fold_id = fold["fold"]
        
        print(f"\nFold {fold_id}:")
        
        for split_name in ['train', 'val', 'test']:
            files = fold[f"{split_name}_wsis"]
            labels = [get_label_from_name(f) for f in files]
            
            pos_count = sum(labels)
            neg_count = len(labels) - pos_count
            pos_ratio = pos_count / len(labels) * 100 if len(labels) > 0 else 0
            
            print(f"  {split_name.capitalize():5s}: "
                  f"Pos={pos_count:3d} ({pos_ratio:5.1f}%), "
                  f"Neg={neg_count:3d} ({100-pos_ratio:5.1f}%), "
                  f"Total={len(labels):3d}")
    
    # =========================
    # 5. 전체 통계
    # =========================
    print("\n5️⃣  전체 통계")
    print("-"*80)
    
    # 모든 unique 파일 수집
    all_unique_files = set()
    for fold in data["folds"]:
        all_unique_files.update(fold["train_wsis"])
        all_unique_files.update(fold["val_wsis"])
        all_unique_files.update(fold["test_wsis"])
    
    total_unique = len(all_unique_files)
    all_labels = [get_label_from_name(f) for f in all_unique_files]
    total_pos = sum(all_labels)
    total_neg = len(all_labels) - total_pos
    
    print(f"Total unique files: {total_unique}")
    print(f"  Positive (meta):    {total_pos} ({total_pos/total_unique*100:.1f}%)")
    print(f"  Negative (nonmeta): {total_neg} ({total_neg/total_unique*100:.1f}%)")
    
    # =========================
    # 6. Stratified 균형 체크
    # =========================
    print("\n6️⃣  Stratified 균형 검증 (Val/Test 비율 편차)")
    print("-"*80)
    
    val_ratios = []
    test_ratios = []
    
    for fold in data["folds"]:
        val_files = fold["val_wsis"]
        test_files = fold["test_wsis"]
        
        val_labels = [get_label_from_name(f) for f in val_files]
        test_labels = [get_label_from_name(f) for f in test_files]
        
        val_pos_ratio = sum(val_labels) / len(val_labels) * 100
        test_pos_ratio = sum(test_labels) / len(test_labels) * 100
        
        val_ratios.append(val_pos_ratio)
        test_ratios.append(test_pos_ratio)
    
    val_mean = np.mean(val_ratios)
    val_std = np.std(val_ratios)
    test_mean = np.mean(test_ratios)
    test_std = np.std(test_ratios)
    
    print(f"Val  Positive ratio: {val_mean:.1f}% ± {val_std:.1f}%")
    print(f"Test Positive ratio: {test_mean:.1f}% ± {test_std:.1f}%")
    
    if val_std < 5.0 and test_std < 5.0:
        print("✅ 매우 균형잡힌 Stratified split (편차 < 5%)")
    elif val_std < 10.0 and test_std < 10.0:
        print("✅ 균형잡힌 Stratified split (편차 < 10%)")
    else:
        print("⚠️  Stratified가 제대로 안 됨 (편차 > 10%)")
    
    # =========================
    # 7. 최종 판정
    # =========================
    print("\n" + "="*80)
    print("🏁 최종 판정")
    print("="*80)
    
    issues = []
    warnings = []
    
    # ✅ Test set 중복 체크 (가장 중요!)
    if test_overlap_found:
        issues.append("❌ Test set 간 중복 존재 (치명적!)")
    
    # Stratified 체크
    if val_std > 10.0 or test_std > 10.0:
        issues.append("❌ Stratified split 불균형 (편차 > 10%)")
    
    # Train/Val/Test Overlap 체크
    has_internal_overlap = False
    for fold in data["folds"]:
        train_set = set(fold["train_wsis"])
        val_set = set(fold["val_wsis"])
        test_set = set(fold["test_wsis"])
        if (train_set & val_set) or (train_set & test_set) or (val_set & test_set):
            has_internal_overlap = True
            break
    
    if has_internal_overlap:
        issues.append("❌ Train/Val/Test 간 overlap 존재")
    
    # Train+Val 영역 겹침은 경고만 (정상일 수 있음)
    all_train_val = set()
    for fold in data["folds"]:
        all_train_val.update(fold["train_wsis"])
        all_train_val.update(fold["val_wsis"])
    
    if len(all_train_val) < total_unique:
        warnings.append("ℹ️  Train+Val 영역이 fold 간 겹침 (K-Fold에서는 정상)")
    
    # 결과 출력
    if not issues:
        print("✅✅✅ 모든 검증 통과! CV splits이 올바르게 생성됨")
        if warnings:
            print("\n참고사항:")
            for warning in warnings:
                print(f"  {warning}")
    else:
        print("발견된 문제:")
        for issue in issues:
            print(f"  {issue}")
        if warnings:
            print("\n참고사항:")
            for warning in warnings:
                print(f"  {warning}")
        print("\n⚠️  CV splits을 재생성해야 합니다!")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    import os
    
    # 기존 파일 검증
    if os.path.exists("cv_splits_k5_seed42.json"):
        print("\n[검증] cv_splits_k5_seed42.json")
        validate_cv_splits("cv_splits_k5_seed42.json")
    
    # 새로 만든 파일 검증
    new_file = "cv_splits/cv_splits_k5_seed42_stratified_8_1_1.json"
    if os.path.exists(new_file):
        print("\n[검증] cv_splits_k5_seed42_stratified_8_1_1.json")
        validate_cv_splits(new_file)
