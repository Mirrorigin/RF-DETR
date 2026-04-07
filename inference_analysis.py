# fixme: This is a bad script, just for quick inference analysis

import pandas as pd

# gt_csv_path = "/home/jingmliang/Projects/Assets/9mm fast walk annotations - Frame Analysis 9mm_fast_walk.mp4.csv"
gt_csv_path = "/home/jingmliang/Projects/Assets/Lucas Backyard Annotations - Sheet1.csv"
pred_csv_path = "./backyard_result.csv"

print("Running analysis (Analyze Mode)...")

gt_df = pd.read_csv(gt_csv_path)
pred_df = pd.read_csv(pred_csv_path)

# Merge and keep all frames in gt and pred files
merged_df = pd.merge(
    gt_df[['Frame_id', 'Timestamp(seconds)', 'Ground Truth Label']],
    pred_df[['Frame_id', 'Pred_Labels', 'Pred_Scores']],
    on='Frame_id',
    how='outer'
)

# Fill
merged_df['Ground Truth Label'] = merged_df['Ground Truth Label'].fillna('--')
merged_df['Pred_Labels'] = merged_df['Pred_Labels'].fillna('--')

# Sort
merged_df = merged_df.sort_values(by='Frame_id').reset_index(drop=True)

CLASSES_TO_ANALYZE = {
        "Standing_Person": {
            "GT_Labels": ["person", "person_standing"],
            "PRED_Labels": ["person_standing"]
        },
        "Fallen_Person": {
            "GT_Labels": ["person_on_ground"],
            "PRED_Labels": ["person_fallen"]
        },
        "Sitting_Person": {
            "GT_Labels": ["person_sitting"],
            "PRED_Labels": ["person_sitting"]
        },
        "Weapon": {
            "GT_Labels": ["weapon_risk"],
            "PRED_Labels": ["handgun"]
        }
}

stats = {target_concept: {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0} for target_concept in CLASSES_TO_ANALYZE}
analysis_result = {target_concept: [] for target_concept in CLASSES_TO_ANALYZE}

print(f"Analyzing for target class!")

for _, row in merged_df.iterrows():
    gt_labels_str = str(row['Ground Truth Label'])
    gt_labels_list = [label.strip() for label in gt_labels_str.split(',')]

    pred_labels_str = str(row['Pred_Labels'])
    pred_labels_list = [label.strip() for label in pred_labels_str.split(',')]

    for target_concept, mapping in CLASSES_TO_ANALYZE.items():

        gt_labels_for_concept = mapping['GT_Labels']
        pred_labels_for_concept = mapping['PRED_Labels']

        gt_contains_target = any(gt_l in gt_labels_for_concept for gt_l in gt_labels_list)
        pred_contains_target = any(p_label in pred_labels_for_concept for p_label in pred_labels_list)

        frame_result = ""
        if gt_contains_target and pred_contains_target:
            stats[target_concept]['TP'] += 1
            frame_result = "TP"
        elif gt_contains_target and not pred_contains_target:
            stats[target_concept]['FN'] += 1
            frame_result = "FN"
        elif not gt_contains_target and pred_contains_target:
            stats[target_concept]['FP'] += 1
            frame_result = "FP"
        elif not gt_contains_target and not pred_contains_target:
            stats[target_concept]['TN'] += 1
            frame_result = "TN"

        analysis_result[target_concept].append(frame_result)

print("Adding analysis columns to DataFrame...")
for target_concept, results_list in analysis_result.items():
    merged_df[f'Analysis_Result ({target_concept})'] = results_list

print("\n" + "=" * 48)
print(f"--- Analysis Summary (Concept-Based) ---")

for target_concept, counts in stats.items():
    tp = counts['TP']
    fp = counts['FP']
    fn = counts['FN']
    tn = counts['TN']
    print(f"\nConcept: '{target_concept}'")
    print(f"  - GT Labels:   {CLASSES_TO_ANALYZE[target_concept]['GT_Labels']}")
    print(f"  - PRED Labels: {CLASSES_TO_ANALYZE[target_concept]['PRED_Labels']}")
    print(f"  True Positives (TP):  {tp}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  True Negatives (TN):  {tn}")

    if (tp + fp) > 0:
        precision = tp / (tp + fp)
        print(f"  Precision: {precision:.4f}")
    else:
        print("  Precision: N/A (TP+FP=0)")

    if (tp + fn) > 0:
        recall = tp / (tp + fn)
        print(f"  Recall:    {recall:.4f}")
    else:
        print("  Recall:    N/A (TP+FN=0)")

print("=" * 48)

merged_df.to_csv("./inference_analysis.csv", index=False)
print("Successfully saved detailed comparison to inference_analysis.csv!")