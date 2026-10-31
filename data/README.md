# Data folder

Place the datasets here (or point the environment variable `SPARSEATTNNET_DATA` to another folder with the
same layout). Files inside each class folder are read in sorted file-name order, which is the acquisition order
used to build the cross-validation folds.

```
data/
  WBC_Train/                 public, see below
    Basophil/  Eosinophil/  Lymphocyte/  Monocyte/  Neutrophil/        (.jpg)
  Tcells/                    internal, available from the corresponding author on reasonable request
    event_new/SW480/  event_new/SW620/                                 (.png, event camera)
    holos/ACT/  holos/ACT7d/  holos/Tcells/                            (.png, raw holograms)
    opd_mat/ACT/  opd_mat/ACT7d/  opd_mat/Tcells/                      (.mat with variable 'opd_crop')
```

**WBC (public).** M. Nickparvar, "White Blood Cells Dataset", Kaggle, 2022,
https://www.kaggle.com/datasets/masoudnickparvar/white-blood-cells-dataset .
Copy its `Train` folder to `data/WBC_Train` (10,170 images: 211 basophils, 743 eosinophils, 2,426 lymphocytes,
560 monocytes, 6,230 neutrophils).

**Event camera, raw holograms, OPD (internal).** These datasets were acquired in our lab and cannot be shared
publicly. They are available from the corresponding author on reasonable request, subject to institutional
data-sharing restrictions.

On first use each dataset is resized once and stored in `cache/<dataset>_<size>.npz`.
