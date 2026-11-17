> **CẢNH BÁO:** Số liệu trong tài liệu này thuộc gói study 16/11 12:20, trước v2; không dùng cho báo cáo.\r\n\r\n# KiÃ¡Â»Æ’m tra web official v2 Ã¢â‚¬â€ 16/11/2026

**TrÃ¡ÂºÂ¡ng thÃƒÂ¡i nghiÃ¡Â»â€¡m thu cuÃ¡Â»â€˜i:** adapter gÃƒÂ³i thÃ¡Â»Â­ Ã„â€˜ÃƒÂ£ truyÃ¡Â»Ân method tÃ†Â°Ã¡Â»Âng minh vÃƒÂ o decide; 15 test hÃ¡Â»â€œi quy Ã„â€˜Ã¡ÂºÂ¡t khÃƒÂ´ng skip, cÃ¡ÂºÂ£ gÃƒÂ³i 13/11 vÃƒÂ  official v2 Ã„â€˜Ã¡ÂºÂ¡t 20/20 qua adapter, 10 ca predict Ã„â€˜Ã¡ÂºÂ§y Ã„â€˜Ã¡Â»Â§ Ã„â€˜Ã¡ÂºÂ¡t hÃ¡Â»Â£p Ã„â€˜Ã¡Â»â€œng. BÃ¡Â»â„¢ Ã¡ÂºÂ£nh Ã„â€˜ÃƒÂ£ Ã„â€˜Ã¡Â»Â§ vÃƒÂ  Word/PDF Ã„â€˜Ã†Â°Ã¡Â»Â£c cÃ¡ÂºÂ­p nhÃ¡ÂºÂ­t theo kÃ¡ÂºÂ¿t quÃ¡ÂºÂ£ nÃƒÂ y. BÃ¡ÂºÂ£n sao bÃ¡ÂºÂ±ng chÃ¡Â»Â©ng Ã„â€˜Ã¡Â»Æ’ lÃ†Â°u Git nÃ¡ÂºÂ±m trong `docs/evidence/study_1611_pre_v2/`. CÃƒÂ¡c ghi chÃƒÂº chÃ¡Â»Â test, chÃ¡Â»Â Ã¡ÂºÂ£nh hoÃ¡ÂºÂ·c thiÃ¡ÂºÂ¿u tham chiÃ¡ÂºÂ¿u bÃƒÂªn dÃ†Â°Ã¡Â»â€ºi chÃ¡Â»â€° mÃƒÂ´ tÃ¡ÂºÂ£ lÃ¡Â»â€¹ch sÃ¡Â»Â­ trÃ†Â°Ã¡Â»â€ºc nghiÃ¡Â»â€¡m thu cuÃ¡Â»â€˜i.

## KiÃ¡Â»Æ’m thÃ¡Â»Â­ gÃƒÂ³i thÃ¡Â»Â­ 13/11

- GÃƒÂ³i thÃ¡Â»Â­ cÃƒÂ³ checkpoint SHA-256 `d90bca482b4367579a70b3158ad62e331644cbc9731229a39b4c15233bfbe052`; cÃƒÂ¡c file `val.npz` vÃƒÂ  `validation_scores.csv` trong hai thÃ†Â° mÃ¡Â»Â¥c `fbf86953` vÃƒÂ  `d90bca48` cÃƒÂ³ cÃƒÂ¹ng SHA-256.
- GÃƒÂ³i khÃƒÂ´ng cÃƒÂ³ `locked_selection.json` Ã„â€˜Ã†Â°Ã¡Â»Â£c nÃ¡ÂºÂ¡p qua adapter vÃ¡Â»â€ºi `method="KNN"` vÃƒÂ  `exif=False`; 20/20 Ã¡ÂºÂ£nh Ã„â€˜Ã¡ÂºÂ¡t logits cÃƒÂ¹ng Ã„â€˜Ã¡Â»Â§ nÃ„Æ’m Ã„â€˜iÃ¡Â»Æ’m OOD.
- BÃ¡ÂºÂ±ng chÃ¡Â»Â©ng: [CSV](../outputs/trial_d90bca48/trial_d90bca48_20_exif_false.csv) vÃƒÂ  [tÃƒÂ³m tÃ¡ÂºÂ¯t](../outputs/trial_d90bca48/trial_d90bca48_20_exif_false.summary.json). Ã„ÂÃƒÂ¢y lÃƒÂ  kiÃ¡Â»Æ’m thÃ¡Â»Â­ adapter, khÃƒÂ´ng dÃƒÂ¹ng sÃ¡Â»â€˜ liÃ¡Â»â€¡u gÃƒÂ³i thÃ¡Â»Â­ cho bÃƒÂ¡o cÃƒÂ¡o official v2.

**CÃ¡ÂºÂ­p nhÃ¡ÂºÂ­t nghiÃ¡Â»â€¡m thu core vÃƒÂ  gÃƒÂ³i 13/11:** Ã„â€˜ÃƒÂ£ cÃƒÂ³ merge d649df4; Ã„â€˜ÃƒÂ£ chÃ¡ÂºÂ¡y bÃ¡Â»â„¢ test core bÃ¡ÂºÂ±ng nguÃ¡Â»â€œn tham chiÃ¡ÂºÂ¿u cÃ¡Â»Â¥c bÃ¡Â»â„¢: 11 passed, khÃƒÂ´ng skip; kiÃ¡Â»Æ’m khÃ¡Â»â€ºp gÃƒÂ³i 13/11 Ã„â€˜Ã¡ÂºÂ¡t 20/20 trÃƒÂªn thanhkhoi. Xem [biÃƒÂªn bÃ¡ÂºÂ£n vÃƒÂ  Ã„â€˜Ã†Â°Ã¡Â»Âng dÃ¡ÂºÂ«n log](ket_qua_nghiem_thu_core_1311.md). CÃƒÂ¡c ghi chÃƒÂº thiÃ¡ÂºÂ¿u file/gÃƒÂ³i 13/11 Ã¡Â»Å¸ dÃ†Â°Ã¡Â»â€ºi lÃƒÂ  lÃ¡Â»â€¹ch sÃ¡Â»Â­ trÃ†Â°Ã¡Â»â€ºc khi ngÃ†Â°Ã¡Â»Âi dÃƒÂ¹ng cung cÃ¡ÂºÂ¥p hai thÃ†Â° mÃ¡Â»Â¥c theo SHA. ChÃ†Â°a cÃ¡ÂºÂ­p nhÃ¡ÂºÂ­t hÃ¡Â»â€œi tÃ¡Â»â€˜ bÃƒÂ¡o cÃƒÂ¡o Word/PDF cÃ…Â©.

**CÃ¡ÂºÂ­p nhÃ¡ÂºÂ­t sau xÃƒÂ¡c minh Ã¡ÂºÂ£nh:** Ã„â€˜ÃƒÂ£ xem Ã„â€˜Ã¡Â»Â§ 10 Ã¡ÂºÂ£nh mÃƒÂ n hÃƒÂ¬nh cÃ¡Â»Â§a 5 ca trong `outputs/study_1611_pre_v2/screenshots/module_review/`, tÃ¡ÂºÂ¥t cÃ¡ÂºÂ£ Ã„â€˜Ã¡ÂºÂ¡t nÃ¡Â»â„¢i dung cÃ¡ÂºÂ§n chÃ¡Â»Â¥p. Xem [biÃƒÂªn bÃ¡ÂºÂ£n kiÃ¡Â»Æ’m Ã¡ÂºÂ£nh vÃƒÂ  viÃ¡Â»â€¡c cÃƒÂ²n lÃ¡ÂºÂ¡i](kiem_tra_anh_minh_chung.md). BÃ¡Â»â„¢ Ã¡ÂºÂ£nh Ã„â€˜ÃƒÂ£ Ã„â€˜Ã¡Â»Â§ Ã„â€˜Ã¡Â»Æ’ viÃ¡ÂºÂ¿t ChÃ†Â°Ã†Â¡ng 5; cÃƒÂ¡c ghi chÃƒÂº chÃ¡Â»Â Ã¡ÂºÂ£nh vÃƒÂ  cÃ¡Â»Â `chapter5_ready=false` cÃ¡Â»Â§a phiÃƒÂªn Ã„â€˜o bÃƒÂªn dÃ†Â°Ã¡Â»â€ºi lÃƒÂ  trÃ¡ÂºÂ¡ng thÃƒÂ¡i lÃ¡Â»â€¹ch sÃ¡Â»Â­, khÃƒÂ´ng phÃ¡ÂºÂ£i yÃƒÂªu cÃ¡ÂºÂ§u chÃ¡Â»Â¥p thÃƒÂªm. BÃƒÂ¡o cÃƒÂ¡o cuÃ¡Â»â€˜i vÃ¡ÂºÂ«n cÃ¡ÂºÂ§n hoÃƒÂ n thiÃ¡Â»â€¡n vÃƒÂ  kiÃ¡Â»Æ’m bÃ¡ÂºÂ£n xuÃ¡ÂºÂ¥t.

## GÃƒÂ³i triÃ¡Â»Æ’n khai

- NguÃ¡Â»â€œn gÃƒÂ³i mÃ¡Â»â€ºi: `C:\Users\vyx\Downloads\EfficientNet-B0`; bÃ¡ÂºÂ£n sao chÃ¡ÂºÂ¡y trong `model_package/`.
- EfficientNet-B0, frozen, seed 42, epoch 9; run `20261116_060136_efficientnet_b0_s42_b8c9dba0`.
- SHA-256 checkpoint: `985e40d4036ae06845503687f77e976cb734dea63801c928ca970081b081d660`. Checkpoint, thresholds, calibration vÃƒÂ  distance parameters khÃ¡Â»â€ºp khÃƒÂ³a trong `locked_selection.json`.
- KNN Ã¡Â»Å¸ T=1, nhÃ¡ÂºÂ­n nÃ¡ÂºÂ¿u Ã„â€˜iÃ¡Â»Æ’m Ã¢â€°Â¥ Ãâ€ž = -1.0221926071848497. Top-5 dÃƒÂ¹ng nhiÃ¡Â»â€¡t Ã„â€˜Ã¡Â»â„¢ 0.6049425684624492. `top_label` bÃ¡ÂºÂ±ng null khi tÃ¡Â»Â« chÃ¡Â»â€˜i.

## KiÃ¡Â»Æ’m khÃ¡Â»â€ºp trÃƒÂªn mÃƒÂ¡y thanhkhoi

MÃƒÂ¡y `megalab-HP-Z8-G4-Workstation`, tÃƒÂ i khoÃ¡ÂºÂ£n `thanhkhoi`, NVIDIA RTX A6000; Linux 6.8, Xeon Gold 6138, RAM 67,098,542,080 byte, Python 3.13.18, torch 2.15.0+cu132, torchvision 0.29.2+cu132. ChÃ¡ÂºÂ¡y CUDA float32, tÃ¡ÂºÂ¯t TF32. DÃ¡Â»Â¯ liÃ¡Â»â€¡u vÃƒÂ  mÃƒÂ£ Ã„â€˜Ã†Â°Ã¡Â»Â£c Ã„â€˜Ã¡Â»â€˜i chiÃ¡ÂºÂ¿u SHA-256 trÃ†Â°Ã¡Â»â€ºc khi chÃ¡ÂºÂ¡y trong thÃ†Â° mÃ¡Â»Â¥c riÃƒÂªng `/mnt/thanhkhoi/catbreed-webapp-20261116-8295e721`.

- PhiÃƒÂªn tÃƒÂ­ch hÃ¡Â»Â£p `src/catbreed_core`: 20 Ã¡ÂºÂ£nh bÃ¡ÂºÂ¯t buÃ¡Â»â„¢c **20/20 Ã„â€˜Ã¡ÂºÂ¡t**. [CSV](../outputs/study_1611_pre_v2/thanhkhoi/module_review/mandatory_20.csv) vÃƒÂ  [tÃƒÂ³m tÃ¡ÂºÂ¯t](../outputs/study_1611_pre_v2/thanhkhoi/module_review/mandatory_20.summary.json).
- ToÃƒÂ n bÃ¡Â»â„¢ 238 Ã¡ÂºÂ£nh cÃ¡Â»Â§a phiÃƒÂªn tÃƒÂ­ch hÃ¡Â»Â£p: logits vÃƒÂ  KNN Ã„â€˜Ã¡ÂºÂ¡t 238/238; cÃ¡ÂºÂ£ 5 Ã„â€˜iÃ¡Â»Æ’m Ã„â€˜Ã¡ÂºÂ¡t **236/238**. `Bengal_143` vÃƒÂ  `Siamese_167` vÃ†Â°Ã¡Â»Â£t dung sai Mahalanobis; giÃ¡Â»Â¯ nguyÃƒÂªn trong [CSV](../outputs/study_1611_pre_v2/thanhkhoi/module_review/all_validation.csv) vÃƒÂ  [tÃƒÂ³m tÃ¡ÂºÂ¯t](../outputs/study_1611_pre_v2/thanhkhoi/module_review/all_validation.summary.json).
- TiÃƒÂªu chÃƒÂ­: `|ÃŽâ€logits| < 1e-4`, `|ÃŽâ€score| < 1e-4`; Mahalanobis cho phÃƒÂ©p thÃƒÂªm `relative < 1e-6` theo phÃƒÂ©p OR. Ã„ÂÃƒÂ¢y lÃƒÂ  kiÃ¡Â»Æ’m khÃ¡Â»â€ºp sÃ¡Â»â€˜ hÃ¡Â»Âc, khÃƒÂ´ng phÃ¡ÂºÂ£i Ã„â€˜Ã¡Â»â„¢ chÃƒÂ­nh xÃƒÂ¡c phÃƒÂ¢n loÃ¡ÂºÂ¡i.
- KiÃ¡Â»Æ’m khÃ¡Â»â€ºp tÃ¡ÂºÂ¯t xoay EXIF nhÃ†Â° notebook; web bÃ¡ÂºÂ­t xoay EXIF. Alpha ghÃƒÂ©p nÃ¡Â»Ân xÃƒÂ¡m 128.

## NÃ„Æ’m ca minh chÃ¡Â»Â©ng

[Manifest vÃƒÂ  kÃ¡ÂºÂ¿t quÃ¡ÂºÂ£ HTTP cÃ¡Â»Â§a phiÃƒÂªn Ã„â€˜ÃƒÂ£ kiÃ¡Â»Æ’m lÃ¡ÂºÂ¡i](../outputs/study_1611_pre_v2/thanhkhoi/module_review/review_summary.json) lÃ†Â°u Ã¡ÂºÂ£nh, nguÃ¡Â»â€œn, SHA-256, nhÃƒÂ£n thÃ¡ÂºÂ­t vÃƒÂ  quyÃ¡ÂºÂ¿t Ã„â€˜Ã¡Â»â€¹nh. API Ã„â€˜ÃƒÂ£ Ã„â€˜Ã†Â°Ã¡Â»Â£c khÃ¡Â»Å¸i Ã„â€˜Ã¡Â»â„¢ng lÃ¡ÂºÂ¡i Ã„â€˜Ã¡Â»Æ’ nÃ¡ÂºÂ¡p code tÃƒÂ­ch hÃ¡Â»Â£p mÃ¡Â»â€ºi. Ã¡ÂºÂ¢nh Ã„â€˜Ã¡ÂºÂ§u vÃƒÂ o Ã¡Â»Å¸ `outputs/study_1611_pre_v2/gpu/cases/`:

- `id_correct.jpg`: Oxford validation `Abyssinian_179`, nhÃ¡ÂºÂ­n Ã„â€˜ÃƒÂºng Abyssinian.
- `dog.jpg`: Oxford chÃƒÂ³ `american_bulldog_138`, bÃ¡Â»â€¹ tÃ¡Â»Â« chÃ¡Â»â€˜i.
- `dtd_val_sample.jpg`: ngÃ†Â°Ã¡Â»Âi dÃƒÂ¹ng bÃƒÂ¡o tÃ¡ÂºÂ¡o bÃ¡ÂºÂ±ng `torchvision.datasets.DTD(split="val", partition=1)[0]`; annotation val1 chÃƒÂ­nh thÃ¡Â»Â©c ghi phÃ¡ÂºÂ§n tÃ¡Â»Â­ Ã„â€˜Ã¡ÂºÂ§u `banded/banded_0006.jpg`. BÃ¡Â»â€¹ tÃ¡Â»Â« chÃ¡Â»â€˜i. ChÃ†Â°a so byte Ã¡ÂºÂ£nh vÃ¡Â»â€ºi bÃ¡ÂºÂ£n gÃ¡Â»â€˜c DTD.
- `felibreed.jpg`: ngÃ†Â°Ã¡Â»Âi dÃƒÂ¹ng xÃƒÂ¡c nhÃ¡ÂºÂ­n lÃ¡ÂºÂ¥y tÃ¡Â»Â« thÃ†Â° mÃ¡Â»Â¥c `Local` cÃ¡Â»Â§a FeliBreed. Ã¡ÂºÂ¢nh Ã„â€˜Ã†Â°Ã¡Â»Â£c **nhÃ¡ÂºÂ­n lÃƒÂ  Ragdoll**; giÃ¡Â»Â¯ nguyÃƒÂªn ca nÃƒÂ y. ChÃ†Â°a cÃƒÂ³ tÃƒÂªn file gÃ¡Â»â€˜c Ã„â€˜Ã¡Â»Æ’ Ã„â€˜Ã¡Â»â€˜i chiÃ¡ÂºÂ¿u trÃ¡Â»Â±c tiÃ¡ÂºÂ¿p.
- `id_error.jpg`: Oxford validation `Abyssinian_121`, nhÃƒÂ£n thÃ¡ÂºÂ­t Abyssinian, app nhÃ¡ÂºÂ­n sai Maine Coon.

KhÃƒÂ´ng chÃ¡Â»â€°nh ngÃ†Â°Ã¡Â»Â¡ng hay thay Ã¡ÂºÂ£nh Ã„â€˜Ã¡Â»Æ’ ÃƒÂ©p tÃ¡Â»Â« chÃ¡Â»â€˜i. Ã¡ÂºÂ¢nh chÃ¡Â»Â¥p trÃƒÂ¬nh duyÃ¡Â»â€¡t cho nÃ„Æ’m ca cÃƒÂ²n chÃ¡Â»Â ngÃ†Â°Ã¡Â»Âi dÃƒÂ¹ng. [NguÃ¡Â»â€œn ngoÃ¡ÂºÂ¡i lai vÃƒÂ  mÃ¡Â»Â©c xÃƒÂ¡c minh](../outputs/study_1611_pre_v2/provenance/external_sources.json).

## Ã„ÂÃ¡Â»â„¢ trÃ¡Â»â€¦ vÃƒÂ  dung lÃ†Â°Ã¡Â»Â£ng trÃƒÂªn thanhkhoi

Ã„ÂÃƒÂºng **50 lÃ†Â°Ã¡Â»Â£t tÃ¡Â»â€¢ng**, bÃ¡Â»Â 5 lÃ†Â°Ã¡Â»Â£t Ã„â€˜Ã¡ÂºÂ§u lÃƒÂ m nÃƒÂ³ng, thÃ¡Â»â€˜ng kÃƒÂª **45 lÃ†Â°Ã¡Â»Â£t** cÃƒÂ²n lÃ¡ÂºÂ¡i. Ã¡ÂºÂ¢nh chÃ¡Â»Ân Ã„â€˜Ã¡Â»Âu tÃ¡Â»Â« validation. Ã„Âo `predict` sau giÃ¡ÂºÂ£i mÃƒÂ£ Ã¡ÂºÂ£nh; khÃƒÂ´ng gÃ¡Â»â€œm Ã„â€˜Ã¡Â»Âc file, upload, mÃƒÂ£ hÃƒÂ³a JSON hay render trÃƒÂ¬nh duyÃ¡Â»â€¡t. GPU Ã„â€˜Ã†Â°Ã¡Â»Â£c Ã„â€˜Ã¡Â»â€œng bÃ¡Â»â„¢ tÃ¡ÂºÂ¡i mÃ¡Â»â€˜c Ã„â€˜o. PNG/Base64 thuÃ¡Â»â„¢c hÃ¡ÂºÂ­u xÃ¡Â»Â­ lÃƒÂ½. [50 lÃ†Â°Ã¡Â»Â£t gÃ¡Â»â€˜c](../outputs/study_1611_pre_v2/thanhkhoi/module_review/latency_raw.csv), [45 lÃ†Â°Ã¡Â»Â£t Ã„â€˜o](../outputs/study_1611_pre_v2/thanhkhoi/module_review/latency_measured.csv), [tÃƒÂ³m tÃ¡ÂºÂ¯t](../outputs/study_1611_pre_v2/thanhkhoi/module_review/latency_summary.json).

| Giai Ã„â€˜oÃ¡ÂºÂ¡n | Median (ms) | p95 (ms) |
| --- | ---: | ---: |
| TiÃ¡Â»Ân xÃ¡Â»Â­ lÃƒÂ½ | 61.78 | 89.51 |
| Suy luÃ¡ÂºÂ­n | 11.08 | 13.38 |
| Grad-CAM | 2.91 | 3.88 |
| Ã„ÂiÃ¡Â»Æ’m tÃ¡Â»Â« chÃ¡Â»â€˜i | 5.86 | 6.43 |
| HÃ¡ÂºÂ­u xÃ¡Â»Â­ lÃƒÂ½/PNG/Base64 | 92.13 | 122.14 |
| **TÃ¡Â»â€¢ng** | **169.75** | **226.83** |

KhÃƒÂ´ng cÃ¡Â»â„¢ng median tÃ¡Â»Â«ng phÃ¡ÂºÂ§n Ã„â€˜Ã¡Â»Æ’ suy ra median tÃ¡Â»â€¢ng. `checkpoint.pt` 16,390,147 byte; `distance_parameters.npz` 21,994,386 byte; tÃ¡Â»â€¢ng 38,384,533 byte (~38.38 MB dung lÃ†Â°Ã¡Â»Â£ng file).
[CSV 50 lÃ†Â°Ã¡Â»Â£t](../outputs/study_1611_pre_v2/thanhkhoi/module_review/latency_raw.csv), [45 lÃ†Â°Ã¡Â»Â£t Ã„â€˜o](../outputs/study_1611_pre_v2/thanhkhoi/module_review/latency_measured.csv), [tÃƒÂ³m tÃ¡ÂºÂ¯t](../outputs/study_1611_pre_v2/thanhkhoi/module_review/latency_summary.json).

BÃ¡Â»â„¢ `gpu/module_integration` vÃƒÂ  `thanhkhoi/results` Ã„â€˜Ã†Â°Ã¡Â»Â£c giÃ¡Â»Â¯ lÃƒÂ m lÃ¡Â»â€¹ch sÃ¡Â»Â­. **BÃ¡Â»â„¢ kÃ¡ÂºÂ¿t quÃ¡ÂºÂ£ cuÃ¡Â»â€˜i sau sÃ¡Â»Â­a phÃƒÂ©p Ã„â€˜o nÃ¡ÂºÂ±m trong `outputs/study_1611_pre_v2/thanhkhoi/module_review/`**: mandatory_20.csv (20/20), all_validation.csv (236/238), latency_raw.csv (50), latency_measured.csv (45), latency_summary.json vÃƒÂ  review_summary.json. BÃ¡ÂºÂ£ng trÃƒÂªn dÃƒÂ¹ng bÃ¡Â»â„¢ nÃƒÂ y. Ã„ÂÃƒÂ£ Ã„â€˜Ã¡Â»â€œng bÃ¡Â»â„¢ CUDA trÃ†Â°Ã¡Â»â€ºc/sau suy luÃ¡ÂºÂ­n; tÃ¡Â»â€¢ng Ã„â€˜o trÃ¡Â»Â±c tiÃ¡ÂºÂ¿p tÃ¡Â»Â« Ã„â€˜Ã¡ÂºÂ§u hÃƒÂ m Ã„â€˜Ã¡ÂºÂ¿n hÃ¡ÂºÂ¿t mÃƒÂ£ hÃƒÂ³a Base64, bao gÃ¡Â»â€œm quyÃ¡ÂºÂ¿t Ã„â€˜Ã¡Â»â€¹nh vÃƒÂ  tÃ¡ÂºÂ¡o Ã¡ÂºÂ£nh giÃ¡ÂºÂ£i thÃƒÂ­ch. KhÃƒÂ´ng dÃƒÂ¹ng sÃ¡Â»â€˜ 153.23/213.79 ms cÃ¡Â»Â§a phÃƒÂ©p Ã„â€˜o trÃ†Â°Ã¡Â»â€ºc cho bÃƒÂ¡o cÃƒÂ¡o cuÃ¡Â»â€˜i.

Ã„ÂÃƒÂ£ Ã„â€˜Ã¡Â»Âc vÃƒÂ  kiÃ¡Â»Æ’m SHA gÃƒÂ³i `C:\Users\vyx\Downloads\model_package_1311`: checkpoint khÃ¡Â»â€ºp hash trong thresholds (`d90bca482b4367579a70b3158ad62e331644cbc9731229a39b4c15233bfbe052`). README cÃ¡Â»Â§a gÃƒÂ³i ghi rÃƒÂµ chÃ¡Â»â€° thÃ¡Â»Â­ code ngÃƒÂ y 13/11, khÃƒÂ´ng dÃƒÂ¹ng bÃƒÂ¡o cÃƒÂ¡o. GÃƒÂ³i thiÃ¡ÂºÂ¿u locked_selection, val.npz vÃƒÂ  validation_scores.csv; khÃƒÂ´ng thay gÃƒÂ³i official v2 hiÃ¡Â»â€¡n tÃ¡ÂºÂ¡i bÃ¡ÂºÂ±ng gÃƒÂ³i nÃƒÂ y.

HÃ†Â°Ã¡Â»â€ºng dÃ¡ÂºÂ«n Ã¡ÂºÂ£nh Ã„â€˜Ã¡ÂºÂ§u vÃƒÂ o, nÃ¡Â»â„¢i dung chÃ¡Â»Â¥p vÃƒÂ  tÃƒÂªn file: [hÃ†Â°Ã¡Â»â€ºng dÃ¡ÂºÂ«n chÃ¡Â»Â¥p minh chÃ¡Â»Â©ng](huong_dan_chup_minh_chung.md).

KÃ¡ÂºÂ¿t quÃ¡ÂºÂ£ Windows RTX 3050 cÃ…Â© dÃƒÂ¹ng giao thÃ¡Â»Â©c 5 lÃ†Â°Ã¡Â»Â£t lÃƒÂ m nÃƒÂ³ng **cÃ¡Â»â„¢ng** 50 lÃ†Â°Ã¡Â»Â£t Ã„â€˜o nÃƒÂªn khÃƒÂ´ng dÃƒÂ¹ng lÃƒÂ m sÃ¡Â»â€˜ Ã„â€˜o bÃƒÂ n giao 50 lÃ†Â°Ã¡Â»Â£t tÃ¡Â»â€¢ng. CSV cÃ…Â© trong `outputs/study_1611_pre_v2/gpu/` Ã„â€˜Ã†Â°Ã¡Â»Â£c giÃ¡Â»Â¯ lÃƒÂ m lÃ¡Â»â€¹ch sÃ¡Â»Â­.

## MÃ¡Â»Å¸ web Ã„â€˜Ã¡Â»Æ’ chÃ¡Â»Â¥p minh chÃ¡Â»Â©ng

Trong thÃ†Â° mÃ¡Â»Â¥c dÃ¡Â»Â± ÃƒÂ¡n trÃƒÂªn Windows, mÃ¡Â»Å¸ hai cÃ¡Â»Â­a sÃ¡Â»â€¢ PowerShell:

```powershell
# CÃ¡Â»Â­a sÃ¡Â»â€¢ 1: nÃ¡Â»â€˜i cÃ¡Â»â€¢ng 8000 tÃ¡Â»â€ºi API GPU thanhkhoi; Ã„â€˜Ã¡Â»Æ’ cÃ¡Â»Â­a sÃ¡Â»â€¢ mÃ¡Â»Å¸
.\scripts\connect_thanhkhoi_web.ps1
```

```powershell
# CÃ¡Â»Â­a sÃ¡Â»â€¢ 2: chÃ¡Â»â€° chÃ¡ÂºÂ¡y giao diÃ¡Â»â€¡n Flask trÃƒÂªn Windows
.\.venv\Scripts\python.exe -m flask --app app run --port 5000
```

KiÃ¡Â»Æ’m `http://127.0.0.1:8000/health` cÃƒÂ³ `is_mock=false`, rÃ¡Â»â€œi mÃ¡Â»Å¸ `http://127.0.0.1:5000`. ChÃ¡Â»Ân lÃ¡ÂºÂ§n lÃ†Â°Ã¡Â»Â£t nÃ„Æ’m Ã¡ÂºÂ£nh Ã¡Â»Å¸ trÃƒÂªn; mÃ¡Â»â€”i ca chÃ¡Â»Â¥p tÃ¡Â»â€¢ng quan gÃ¡Â»â€œm Ã„â€˜Ã¡ÂºÂ§u vÃƒÂ o, trÃ¡ÂºÂ¡ng thÃƒÂ¡i, Top-5, Grad-CAM vÃƒÂ  chÃ¡Â»Â¥p phÃ¡ÂºÂ§n Ã¢â‚¬Å“Chi tiÃ¡ÂºÂ¿t mÃƒÂ´ hÃƒÂ¬nh vÃƒÂ  5 Ã„â€˜iÃ¡Â»Æ’m tÃ¡Â»Â« chÃ¡Â»â€˜iÃ¢â‚¬Â. LÃ†Â°u vÃƒÂ o `outputs/study_1611_pre_v2/screenshots/`; ghi rÃƒÂµ Ã¡ÂºÂ£nh cÃ¡Â»Â§a phiÃƒÂªn thanhkhoi. FeliBreed phÃ¡ÂºÂ£i thÃ¡Â»Æ’ hiÃ¡Â»â€¡n kÃ¡ÂºÂ¿t quÃ¡ÂºÂ£ Ragdoll hiÃ¡Â»â€¡n tÃ¡ÂºÂ¡i. Ã¡ÂºÂ¢nh chÃ¡Â»Â¥p Windows GPU cÃ…Â© khÃƒÂ´ng phÃ¡ÂºÂ£i minh chÃ¡Â»Â©ng cho phiÃƒÂªn thanhkhoi.

`chapter5_ready=false` cho Ã„â€˜Ã¡ÂºÂ¿n khi kiÃ¡Â»Æ’m Ã„â€˜Ã¡Â»Â§ Ã¡ÂºÂ£nh chÃ¡Â»Â¥p vÃƒÂ  ghi hai giÃ¡Â»â€ºi hÃ¡ÂºÂ¡n Mahalanobis cÃƒÂ¹ng mÃ¡Â»Â©c xÃƒÂ¡c minh nguÃ¡Â»â€œn DTD/FeliBreed. BÃ¡ÂºÂ£n nhÃƒÂ¡p chÃ†Â°Ã†Â¡ng 5 Ã¡Â»Å¸ `docs/chapter5_webapp_draft.md`.

## Sai khÃƒÂ¡c so vÃ¡Â»â€ºi bÃƒÂ n giao bÃ¡Â»â€¢ sung

`BanGiao_WebApp_Vy_BoSung_2026-11-16.md` yÃƒÂªu cÃ¡ÂºÂ§u app gÃ¡Â»Âi `src/catbreed_core` tÃ¡Â»Â« nhÃƒÂ¡nh `feat/inference-core`. YÃƒÂªu cÃ¡ÂºÂ§u nÃƒÂ y Ã„â€˜ÃƒÂ£ Ã„â€˜Ã†Â°Ã¡Â»Â£c tÃƒÂ­ch hÃ¡Â»Â£p vÃƒÂ  xÃƒÂ¡c nhÃ¡ÂºÂ­n bÃ¡ÂºÂ±ng kiÃ¡Â»Æ’m khÃ¡Â»â€ºp 20/20, kiÃ¡Â»Æ’m mÃ¡Â»Å¸ rÃ¡Â»â„¢ng 236/238 (hai sai khÃƒÂ¡c Mahalanobis Ã„â€˜ÃƒÂ£ biÃ¡ÂºÂ¿t) vÃƒÂ  benchmark 50 lÃ†Â°Ã¡Â»Â£t trÃƒÂªn cÃƒÂ¹ng mÃƒÂ¡y. PhÃ¡ÂºÂ§n chÃ¡Â»Â¥p Ã¡ÂºÂ£nh trÃƒÂ¬nh duyÃ¡Â»â€¡t vÃ¡ÂºÂ«n do ngÃ†Â°Ã¡Â»Âi dÃƒÂ¹ng thÃ¡Â»Â±c hiÃ¡Â»â€¡n; `chapter5_ready=false` cho Ã„â€˜Ã¡ÂºÂ¿n khi Ã„â€˜Ã¡Â»Â§ Ã¡ÂºÂ£nh vÃƒÂ  hoÃƒÂ n tÃ¡ÂºÂ¥t xÃƒÂ¡c minh nguÃ¡Â»â€œn DTD/FeliBreed.

Ã¡ÂºÂ¢nh mÃƒÂ¨o Ã„â€˜ÃƒÂºng Ã„â€˜ÃƒÂ£ Ã„â€˜Ã¡Â»â€¢i sang validation `Abyssinian_179` theo yÃƒÂªu cÃ¡ÂºÂ§u bÃ¡Â»Â watermark. File hiÃ¡Â»â€¡n tÃ¡ÂºÂ¡i `gpu/cases/id_correct.jpg` lÃƒÂ  bÃ¡ÂºÂ£n sao nguyÃƒÂªn Ã¡ÂºÂ£nh nÃƒÂ y; phÃ¡ÂºÂ£n hÃ¡Â»â€œi GPU mÃ¡Â»â€ºi nÃ¡ÂºÂ±m Ã¡Â»Å¸ `thanhkhoi/module_review/id_correct_replacement_http.json`. CÃƒÂ¡c manifest/run_summary vÃƒÂ  benchmark lÃ¡Â»â€¹ch sÃ¡Â»Â­ vÃ¡ÂºÂ«n ghi Ã¡ÂºÂ£nh Abyssinian_101 Ã„â€˜ÃƒÂºng vÃ¡Â»â€ºi phiÃƒÂªn Ã„â€˜o cÃ…Â©; khÃƒÂ´ng diÃ¡Â»â€¦n giÃ¡ÂºÂ£i chÃƒÂºng thÃƒÂ nh phÃƒÂ©p Ã„â€˜o cho Ã¡ÂºÂ£nh thay thÃ¡ÂºÂ¿. Ã¡ÂºÂ¢nh vÃƒÂ  kÃ¡ÂºÂ¿t quÃ¡ÂºÂ£ cÃ…Â© Ã„â€˜Ã†Â°Ã¡Â»Â£c lÃ†Â°u trong `gpu/cases/history_Abyssinian_101/`.
