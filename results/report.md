# Traffic sign robustness: SVM results (HOG + HSV + Hu features)

All numbers are accuracy unless stated. Corruption grid ran in `fast` mode (subset of the test set per corrupted condition, full test set for clean). Severity 1 = mild, 5 = strongest. Retention = accuracy under corruption / clean accuracy on the same samples.

## 1. Headline

| dataset | train_acc | test_acc | test_macro_f1 | mean_acc_sev5 | mean_f1_sev5 | retention_sev5 | n_test |
|---|---|---|---|---|---|---|---|
| GTSRB | 0.994 | 0.943 | 0.921 | 0.602 | 0.559 | 0.637 | 12630 |
| BTSD | 0.999 | 0.876 | 0.655 | 0.656 | 0.407 | 0.749 | 1425 |
| CTSD | 1.000 | 1.000 | 1.000 | 0.642 | 0.496 | 0.642 | 1204 |


## 2. Accuracy per corruption and severity

### GTSRB (clean accuracy 0.943)

| corruption | family | sev1 | sev2 | sev3 | sev4 | sev5 | mean |
|---|---|---|---|---|---|---|---|
| defocus_blur | blur | 0.950 | 0.939 | 0.937 | 0.909 | 0.868 | 0.920 |
| motion_blur | blur | 0.935 | 0.914 | 0.851 | 0.703 | 0.500 | 0.781 |
| zoom_blur | blur | 0.952 | 0.949 | 0.939 | 0.907 | 0.797 | 0.909 |
| brightness | digital | 0.945 | 0.923 | 0.891 | 0.861 | 0.822 | 0.888 |
| contrast | digital | 0.946 | 0.941 | 0.935 | 0.918 | 0.896 | 0.927 |
| elastic_transform | digital | 0.940 | 0.932 | 0.921 | 0.888 | 0.841 | 0.904 |
| jpeg_compression | digital | 0.919 | 0.887 | 0.823 | 0.692 | 0.493 | 0.763 |
| pixelate | digital | 0.915 | 0.890 | 0.886 | 0.812 | 0.658 | 0.832 |
| gaussian_noise | noise | 0.863 | 0.650 | 0.535 | 0.388 | 0.286 | 0.545 |
| impulse_noise | noise | 0.944 | 0.947 | 0.941 | 0.931 | 0.911 | 0.935 |
| shot_noise | noise | 0.929 | 0.870 | 0.752 | 0.553 | 0.336 | 0.688 |
| fog | weather | 0.907 | 0.843 | 0.760 | 0.684 | 0.598 | 0.758 |
| frost | weather | 0.693 | 0.539 | 0.388 | 0.265 | 0.160 | 0.409 |
| snow | weather | 0.786 | 0.575 | 0.427 | 0.319 | 0.267 | 0.475 |


### BTSD (clean accuracy 0.876)

| corruption | family | sev1 | sev2 | sev3 | sev4 | sev5 | mean |
|---|---|---|---|---|---|---|---|
| defocus_blur | blur | 0.875 | 0.874 | 0.872 | 0.855 | 0.820 | 0.859 |
| motion_blur | blur | 0.866 | 0.853 | 0.818 | 0.701 | 0.550 | 0.758 |
| zoom_blur | blur | 0.873 | 0.867 | 0.854 | 0.816 | 0.720 | 0.826 |
| brightness | digital | 0.872 | 0.866 | 0.865 | 0.860 | 0.848 | 0.862 |
| contrast | digital | 0.873 | 0.867 | 0.866 | 0.862 | 0.851 | 0.864 |
| elastic_transform | digital | 0.866 | 0.855 | 0.843 | 0.829 | 0.791 | 0.837 |
| jpeg_compression | digital | 0.854 | 0.844 | 0.815 | 0.781 | 0.685 | 0.796 |
| pixelate | digital | 0.850 | 0.841 | 0.836 | 0.794 | 0.701 | 0.804 |
| gaussian_noise | noise | 0.844 | 0.786 | 0.710 | 0.604 | 0.437 | 0.676 |
| impulse_noise | noise | 0.873 | 0.868 | 0.867 | 0.846 | 0.837 | 0.858 |
| shot_noise | noise | 0.849 | 0.828 | 0.778 | 0.701 | 0.524 | 0.736 |
| fog | weather | 0.866 | 0.850 | 0.839 | 0.799 | 0.754 | 0.822 |
| frost | weather | 0.783 | 0.706 | 0.587 | 0.416 | 0.251 | 0.549 |
| snow | weather | 0.810 | 0.716 | 0.636 | 0.513 | 0.418 | 0.619 |


### CTSD (clean accuracy 1.000)

| corruption | family | sev1 | sev2 | sev3 | sev4 | sev5 | mean |
|---|---|---|---|---|---|---|---|
| defocus_blur | blur | 0.999 | 0.999 | 0.998 | 0.985 | 0.944 | 0.985 |
| motion_blur | blur | 0.998 | 0.986 | 0.948 | 0.681 | 0.443 | 0.811 |
| zoom_blur | blur | 0.999 | 0.994 | 0.987 | 0.864 | 0.625 | 0.894 |
| brightness | digital | 0.998 | 0.999 | 0.995 | 0.983 | 0.958 | 0.987 |
| contrast | digital | 1.000 | 0.999 | 0.999 | 0.997 | 0.994 | 0.998 |
| elastic_transform | digital | 0.999 | 0.998 | 0.988 | 0.953 | 0.880 | 0.964 |
| jpeg_compression | digital | 0.993 | 0.981 | 0.970 | 0.911 | 0.708 | 0.913 |
| pixelate | digital | 0.978 | 0.972 | 0.946 | 0.844 | 0.543 | 0.856 |
| gaussian_noise | noise | 0.979 | 0.907 | 0.748 | 0.502 | 0.277 | 0.683 |
| impulse_noise | noise | 0.998 | 0.998 | 0.999 | 0.995 | 0.980 | 0.994 |
| shot_noise | noise | 0.991 | 0.971 | 0.917 | 0.716 | 0.394 | 0.798 |
| fog | weather | 0.997 | 0.989 | 0.976 | 0.948 | 0.895 | 0.961 |
| frost | weather | 0.938 | 0.775 | 0.524 | 0.265 | 0.134 | 0.527 |
| snow | weather | 0.929 | 0.709 | 0.487 | 0.307 | 0.213 | 0.529 |


## 3. Macro-F1 per corruption and severity

### GTSRB

| corruption | sev1 | sev2 | sev3 | sev4 | sev5 | mean |
|---|---|---|---|---|---|---|
| brightness | 0.929 | 0.909 | 0.877 | 0.853 | 0.813 | 0.876 |
| contrast | 0.932 | 0.928 | 0.920 | 0.898 | 0.877 | 0.911 |
| defocus_blur | 0.937 | 0.923 | 0.920 | 0.878 | 0.816 | 0.895 |
| elastic_transform | 0.920 | 0.907 | 0.897 | 0.855 | 0.786 | 0.873 |
| fog | 0.899 | 0.829 | 0.752 | 0.676 | 0.576 | 0.746 |
| frost | 0.673 | 0.494 | 0.338 | 0.194 | 0.093 | 0.358 |
| gaussian_noise | 0.848 | 0.631 | 0.493 | 0.336 | 0.237 | 0.509 |
| impulse_noise | 0.931 | 0.930 | 0.926 | 0.907 | 0.890 | 0.917 |
| jpeg_compression | 0.902 | 0.867 | 0.800 | 0.673 | 0.450 | 0.738 |
| motion_blur | 0.921 | 0.894 | 0.811 | 0.648 | 0.422 | 0.739 |
| pixelate | 0.892 | 0.855 | 0.847 | 0.762 | 0.577 | 0.787 |
| shot_noise | 0.918 | 0.859 | 0.717 | 0.500 | 0.276 | 0.654 |
| snow | 0.753 | 0.537 | 0.388 | 0.280 | 0.215 | 0.435 |
| zoom_blur | 0.940 | 0.938 | 0.926 | 0.893 | 0.799 | 0.899 |


### BTSD

| corruption | sev1 | sev2 | sev3 | sev4 | sev5 | mean |
|---|---|---|---|---|---|---|
| brightness | 0.659 | 0.646 | 0.649 | 0.632 | 0.615 | 0.640 |
| contrast | 0.664 | 0.658 | 0.660 | 0.650 | 0.637 | 0.654 |
| defocus_blur | 0.681 | 0.677 | 0.679 | 0.633 | 0.562 | 0.646 |
| elastic_transform | 0.657 | 0.629 | 0.607 | 0.561 | 0.507 | 0.592 |
| fog | 0.643 | 0.637 | 0.615 | 0.572 | 0.503 | 0.594 |
| frost | 0.523 | 0.418 | 0.317 | 0.192 | 0.083 | 0.307 |
| gaussian_noise | 0.617 | 0.532 | 0.446 | 0.321 | 0.211 | 0.425 |
| impulse_noise | 0.667 | 0.647 | 0.641 | 0.595 | 0.588 | 0.628 |
| jpeg_compression | 0.616 | 0.613 | 0.557 | 0.504 | 0.413 | 0.540 |
| motion_blur | 0.658 | 0.644 | 0.553 | 0.423 | 0.275 | 0.510 |
| pixelate | 0.623 | 0.584 | 0.579 | 0.512 | 0.389 | 0.538 |
| shot_noise | 0.627 | 0.574 | 0.511 | 0.406 | 0.256 | 0.475 |
| snow | 0.560 | 0.439 | 0.370 | 0.274 | 0.198 | 0.368 |
| zoom_blur | 0.670 | 0.665 | 0.651 | 0.578 | 0.461 | 0.605 |


### CTSD

| corruption | sev1 | sev2 | sev3 | sev4 | sev5 | mean |
|---|---|---|---|---|---|---|
| brightness | 0.995 | 0.983 | 0.951 | 0.899 | 0.836 | 0.933 |
| contrast | 1.000 | 0.982 | 0.982 | 0.960 | 0.933 | 0.972 |
| defocus_blur | 0.983 | 0.983 | 0.965 | 0.885 | 0.783 | 0.919 |
| elastic_transform | 0.998 | 0.995 | 0.913 | 0.816 | 0.690 | 0.883 |
| fog | 0.980 | 0.931 | 0.908 | 0.832 | 0.770 | 0.884 |
| frost | 0.805 | 0.570 | 0.366 | 0.146 | 0.035 | 0.384 |
| gaussian_noise | 0.896 | 0.748 | 0.537 | 0.363 | 0.146 | 0.538 |
| impulse_noise | 0.995 | 0.995 | 0.998 | 0.943 | 0.887 | 0.964 |
| jpeg_compression | 0.934 | 0.884 | 0.843 | 0.706 | 0.479 | 0.769 |
| motion_blur | 0.965 | 0.902 | 0.797 | 0.475 | 0.238 | 0.676 |
| pixelate | 0.852 | 0.811 | 0.815 | 0.625 | 0.320 | 0.685 |
| shot_noise | 0.928 | 0.860 | 0.738 | 0.530 | 0.243 | 0.660 |
| snow | 0.803 | 0.581 | 0.359 | 0.189 | 0.122 | 0.411 |
| zoom_blur | 0.982 | 0.941 | 0.858 | 0.689 | 0.455 | 0.785 |


## 4. Retention per corruption family

| dataset | family | acc_sev5 | retention_sev5 | acc_mean_sev1to5 | retention_sev1to5 |
|---|---|---|---|---|---|
| GTSRB | blur | 0.722 | 0.763 | 0.870 | 0.919 |
| GTSRB | digital | 0.742 | 0.784 | 0.863 | 0.912 |
| GTSRB | noise | 0.511 | 0.540 | 0.722 | 0.764 |
| GTSRB | weather | 0.342 | 0.361 | 0.547 | 0.579 |
| BTSD | blur | 0.697 | 0.795 | 0.814 | 0.930 |
| BTSD | digital | 0.775 | 0.885 | 0.833 | 0.951 |
| BTSD | noise | 0.600 | 0.685 | 0.757 | 0.864 |
| BTSD | weather | 0.474 | 0.542 | 0.663 | 0.757 |
| CTSD | blur | 0.671 | 0.671 | 0.897 | 0.897 |
| CTSD | digital | 0.817 | 0.817 | 0.943 | 0.943 |
| CTSD | noise | 0.550 | 0.550 | 0.825 | 0.825 |
| CTSD | weather | 0.414 | 0.414 | 0.672 | 0.672 |


## 5. Best and worst corruption (severity 5)

| dataset | best | best_acc | worst | worst_acc |
|---|---|---|---|---|
| GTSRB | impulse_noise | 0.911 | frost | 0.160 |
| BTSD | contrast | 0.851 | frost | 0.251 |
| CTSD | contrast | 0.994 | frost | 0.134 |


## 6. Feature-group ablation (SVM trained on one feature group)

Mean accuracy over the 14 corruptions at each severity; clean = full clean test set.

| dataset | features | clean | sev3 | sev5 |
|---|---|---|---|---|
| BTSD | SVM[HOG+HSV] | 0.876 | 0.799 | 0.655 |
| BTSD | SVM[HOG] | 0.869 | 0.787 | 0.639 |
| BTSD | SVM[HSV] | 0.461 | 0.318 | 0.229 |
| BTSD | SVM[Hu] | 0.193 | 0.162 | 0.143 |
| BTSD | SVM[full model] | 0.876 | 0.799 | 0.656 |
| CTSD | SVM[HOG+HSV] | 1.000 | 0.891 | 0.641 |
| CTSD | SVM[HOG] | 1.000 | 0.886 | 0.637 |
| CTSD | SVM[HSV] | 0.947 | 0.508 | 0.343 |
| CTSD | SVM[Hu] | 0.421 | 0.175 | 0.136 |
| CTSD | SVM[full model] | 1.000 | 0.891 | 0.642 |
| GTSRB | SVM[HOG+HSV] | 0.943 | 0.786 | 0.603 |
| GTSRB | SVM[HOG] | 0.943 | 0.787 | 0.603 |
| GTSRB | SVM[HSV] | 0.264 | 0.185 | 0.148 |
| GTSRB | SVM[Hu] | 0.073 | 0.065 | 0.062 |
| GTSRB | SVM[full model] | 0.943 | 0.785 | 0.602 |


## 7. Preprocessing ablation (GTSRB only)

| setting | clean | sev1 | sev3 | sev5 |
|---|---|---|---|---|
| SVM[median=off,clahe=on] | 0.943 | 0.860 | 0.694 | 0.490 |
| SVM[median=on,clahe=off] | 0.940 | 0.889 | 0.757 | 0.544 |
| SVM[median=on,clahe=on] | 0.943 | 0.902 | 0.785 | 0.602 |


## 8. Model tuning on GTSRB validation data (Phase 2)

Each row is a candidate change compared with the reference at that step. `adopted` = kept.

| step | config | val_accuracy | val_macro_f1 | feature_dim | fit_s | adopted | note |
|---|---|---|---|---|---|---|---|
| 0 baseline | baseline (roi, 400/class, median, HOG8/9, HSV16, Hu, SVC C=10) | 0.8778 | 0.8524 | 379 | 27 | yes | start |
| a ROI crop | full frame | 0.7716 | 0.7260 | 379 | 34 | no | gain -10.62 pt |
| b train size | 1000/class | 0.8741 | 0.8500 | 379 | 67 | no | gain -0.38 pt |
| b train size | all | 0.8756 | 0.8506 | 379 | 72 | no | gain -0.23 pt |
| c CLAHE | CLAHE on L | 0.8938 | 0.8718 | 379 | 24 | yes | gain +1.59 pt |
| d median blur | no median blur | 0.8769 | 0.8719 | 379 | 34 | no | gain -1.69 pt |
| e (resume) | re-evaluated state at resume | 0.8938 | 0.8718 | 379 | 25 | yes | run restarted here after fixing the combination rule |
| e HOG | COMBINED cell 4x4+resolution 48+orient 12 | 0.9492 | 0.9327 | 5863 | 488 | no (tested only as a combination) | combination of passing changes |
| e HOG | cell 4x4 | 0.9485 | 0.9461 | 1819 | 107 | yes | gain +5.47 pt |
| e HOG | orient 12 | 0.8994 | 0.8789 | 487 | 33 | no | gain +0.57 pt |
| e HOG | resolution 48 | 0.9388 | 0.9317 | 955 | 51 | no | gain +4.51 pt |
| e HOG | multi-scale 8+16 | 0.8972 | 0.8732 | 415 | 27 | no | gain +0.34 pt |


Final frozen setup: ROI crop, 400 images/class, median blur on, CLAHE on, HOG 4x4 cells / 9 orientations / 32 px, HSV 16 bins, Hu moments, SVC C=10 gamma=scale. Steps for HSV, Hu, C/gamma grid, RF and speed variants were skipped to save time.

## 9. Most frequent confusions (top 5 per dataset: clean, and the worst corruption at severity 5)

### GTSRB

**clean**

| true_class | predicted_class | count | share_of_true_class |
|---|---|---|---|
| 24 | 26 | 32 | 0.356 |
| 26 | 25 | 31 | 0.172 |
| 18 | 25 | 30 | 0.077 |
| 6 | 5 | 27 | 0.180 |
| 5 | 1 | 26 | 0.041 |


**frost_sev5**

| true_class | predicted_class | count | share_of_true_class |
|---|---|---|---|
| 10 | 25 | 43 | 0.410 |
| 18 | 25 | 38 | 0.613 |
| 5 | 12 | 37 | 0.370 |
| 38 | 25 | 36 | 0.330 |
| 4 | 25 | 35 | 0.333 |


### BTSD

**clean**

| true_class | predicted_class | count | share_of_true_class |
|---|---|---|---|
| 46 | 47 | 6 | 0.545 |
| 58 | 47 | 5 | 0.833 |
| 46 | 45 | 5 | 0.455 |
| 28 | 22 | 5 | 0.139 |
| 21 | 22 | 5 | 0.278 |


**frost_sev5**

| true_class | predicted_class | count | share_of_true_class |
|---|---|---|---|
| 38 | 22 | 55 | 0.567 |
| 32 | 38 | 47 | 0.313 |
| 61 | 22 | 44 | 0.557 |
| 40 | 22 | 43 | 0.729 |
| 32 | 22 | 32 | 0.213 |


### CTSD

**clean**

| true_class | predicted_class | count | share_of_true_class |
|---|---|---|---|
| 57 | 57 | 0 | 0.000 |
| 57 | 56 | 0 | 0.000 |
| 57 | 55 | 0 | 0.000 |
| 57 | 54 | 0 | 0.000 |
| 57 | 53 | 0 | 0.000 |


**frost_sev5**

| true_class | predicted_class | count | share_of_true_class |
|---|---|---|---|
| 28 | 54 | 88 | 0.863 |
| 3 | 54 | 41 | 0.661 |
| 5 | 54 | 41 | 0.804 |
| 17 | 54 | 40 | 0.952 |
| 11 | 54 | 40 | 0.909 |

