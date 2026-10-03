# Scope cleanup

Preserved: six classical model folders, CNN, progress, final briefs, assigned dataset and required deployment files.

Removed obsolete mixed-model tables:
- CNN/results/model_comparison_with_cnn.csv (mixed CNN with classical models and stale metrics)
- parts/gradient_boosting/outputs/model_comparison_6_members.csv (superseded by outputs/six_model_comparison.csv)

Replaced rather than duplicated: conflicting main training runner; six partial-stage notebooks; main orchestration notebook; CNN notebook; stale README/analytics instructions; repeated app feature functions. The app now imports training preprocessing directly.

Preserved legacy raw source photos: byte-hash audit found distinct original files in all four short-name folders, including larger originals. They are excluded from training and not treated as disposable duplicates. See cleanup_audit.json. No progress artifact or unique source image was removed.
