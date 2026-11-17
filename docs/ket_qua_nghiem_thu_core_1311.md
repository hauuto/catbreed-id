# Nghiá»‡m thu module core vá»›i checkpoint 13 thÃ¡ng 11

Cáº­p nháº­t sau kiá»ƒm adapter: Ä‘Ã£ xá»­ lÃ½ thiáº¿u truyá»n method vÃ o decide; cáº£ gÃ³i thá»­ vÃ  official v2 Ä‘áº¡t 20/20 qua adapter app, 15 test há»“i quy Ä‘áº¡t khÃ´ng skip vÃ  10 ca predict Ä‘áº§y Ä‘á»§ Ä‘Ã£ kiá»ƒm. Xem `docs/kiem_tra_adapter_1311_sau_sua.md` vÃ  `outputs/study_1611_pre_v2/adapter_trial_review/`.

NgÃ y thá»±c hiá»‡n: 16/11/2026. Cháº¡y trÃªn tÃ i khoáº£n thanhkhoi, GPU RTX A6000, torch 2.15.0+cu132, float32, táº¯t TF32. Bá»™ thá»­ Ä‘áº·t riÃªng táº¡i `/mnt/thanhkhoi/catbreed-webapp-20261116-8295e721/trial_1311_review_20261116`.

## Nguá»“n dá»¯ liá»‡u vÃ  nháº­n dáº¡ng gÃ³i

- Checkpoint gá»‘c: `C:\Users\vyx\Downloads\model_package_1311\checkpoint.pt`.
- SHA-256: `d90bca482b4367579a70b3158ad62e331644cbc9731229a39b4c15233bfbe052`, khá»›p thresholds.json.
- Run: `20261113_145326_efficientnet_b0_s42_df1edde1`, frozen.
- Hai file tham chiáº¿u do ngÆ°á»i dÃ¹ng táº£i vá»: `C:\Users\vyx\Downloads\d90bca48\val.npz` vÃ  `validation_scores.csv`. KhÃ´ng dÃ¹ng bá»™ trong thÆ° má»¥c fbf86953.
- ÄÃ£ ghÃ©p thÃ nh gÃ³i cá»¥c bá»™ Ä‘áº§y Ä‘á»§ táº¡i `outputs/reference_packages/model_package_1311/`, kÃ¨m reference_manifest.json ghi nguá»“n, SHA vÃ  kÃ­ch thÆ°á»›c tá»«ng file. KhÃ´ng xÃ¡c minh Ä‘Æ°á»£c SHA tá»« server private; nguá»“n gÃ³i tham chiáº¿u dá»±a trÃªn thÆ° má»¥c ngÆ°á»i dÃ¹ng cáº¥p vÃ  kiá»ƒm ná»™i dung sá»‘.
- NPZ vÃ  CSV chá»©a cÃ¹ng 238 ID, Ä‘Ãºng thá»© tá»±; logits cÃ³ 12 lá»›p, Ä‘iá»ƒm há»¯u háº¡n. Má»i file chuyá»ƒn sang mÃ¡y GPU Ä‘Æ°á»£c kiá»ƒm SHA trÆ°á»›c khi cháº¡y.

## Káº¿t quáº£

1. `tests/test_catbreed_core.py`: **11 passed**, **0 skipped**, **0 failures/errors**. ÄÃ£ Ä‘á»‘i chiáº¿u cáº£ log pytest vÃ  JUnit XML.
2. Kiá»ƒm trá»±c tiáº¿p module vá»›i 20 áº£nh validation Ä‘áº§u tiÃªn, exif=False: **20/20 Ä‘áº¡t**. Logits vÃ  nÄƒm Ä‘iá»ƒm Ä‘á»‘i chiáº¿u NPZ/CSV. KhÃ´ng thay Ä‘á»•i dung sai; bá»™ test gá»‘c cÃ²n giá»¯ kiá»ƒm Mahalanobis theo sai sá»‘ tÆ°Æ¡ng Ä‘á»‘i.
3. Táº¥t cáº£ káº¿t quáº£ model 13/11 chá»‰ phá»¥c vá»¥ nghiá»‡m thu code; khÃ´ng thay sá»‘ benchmark, áº£nh chá»¥p hay káº¿t quáº£ official v2 trong ChÆ°Æ¡ng 5.

## Thay Ä‘á»•i Ä‘á»ƒ cháº¡y offline vÃ  sá»­a lá»—i phÃ¡t hiá»‡n

- Bá»™ test nháº­n thÃªm CATBREED_TEST_PACKAGE vÃ  OXFORD_PETS_IMAGES Ä‘á»ƒ dÃ¹ng gÃ³i vÃ  áº£nh cá»¥c bá»™; CATBREED_TEST_DEVICE chá»n thiáº¿t bá»‹. Giá»¯ cÃ¡c phÃ©p assert gá»‘c, khÃ´ng dÃ¹ng token giáº£ hay mock HF Ä‘á»ƒ bá» qua kiá»ƒm dá»¯ liá»‡u tháº­t.
- KhÃ´i phá»¥c `DistanceScores.save` cÃ²n thiáº¿u trong module Ä‘Ã£ tÃ­ch há»£p, theo Ä‘á»‹nh dáº¡ng gá»‘c cá»§a module chung. Hai test cáº§n lÆ°u gÃ³i tá»•ng há»£p Ä‘Ã£ Ä‘áº¡t sau sá»­a.
- `decide` há»— trá»£ Ä‘á»‘i tÆ°á»£ng Package tháº­t cÃ³ class_mapping vÃ  Ä‘á»‘i tÆ°á»£ng cáº¥u hÃ¬nh tá»‘i thiá»ƒu cá»§a test gá»‘c. Vá»›i Package tháº­t, nhÃ£n váº«n láº¥y tá»« checkpoint.
- Pytest Ä‘Æ°á»£c cÃ i riÃªng trong runtime_tests cá»§a thÆ° má»¥c thá»­, khÃ´ng cÃ i vÃ o mÃ´i trÆ°á»ng dÃ¹ng chung. Model official v2 vÃ  API web khÃ´ng bá»‹ thay gÃ³i.

## Báº±ng chá»©ng local

- `outputs/study_1611_pre_v2/trial_1311/pytest.log`
- `outputs/study_1611_pre_v2/trial_1311/pytest.xml`
- `outputs/study_1611_pre_v2/trial_1311/trial_consistency_20.csv`
- `outputs/study_1611_pre_v2/trial_1311/trial_summary.json`
- `outputs/study_1611_pre_v2/trial_1311/acceptance.json`

## Cáº­p nháº­t tráº¡ng thÃ¡i bÃ n giao

Merge nhÃ¡nh Ä‘Ã£ cÃ³ commit d649df4. Hai má»¥c cÃ²n thiáº¿u trÆ°á»›c Ä‘Ã¢y lÃ  bá»™ test core Ä‘áº§y Ä‘á»§ vÃ  kiá»ƒm khá»›p checkpoint 13/11 nay Ä‘Ã£ cÃ³ báº±ng chá»©ng Ä‘áº¡t. CÃ¡c báº£n Word/PDF táº¡o trÆ°á»›c bÆ°á»›c nÃ y váº«n ghi chÃºng lÃ  chÆ°a xÃ¡c nháº­n; cáº§n cáº­p nháº­t bÃ¡o cÃ¡o theo biÃªn báº£n nÃ y trÆ°á»›c khi ná»™p.

Hai ca Mahalanobis cá»§a kiá»ƒm má»Ÿ rá»™ng official v2 vÃ  giá»›i háº¡n xÃ¡c minh nguá»“n DTD/FeliBreed váº«n cÃ²n nhÆ° Ä‘Ã£ cÃ´ng bá»‘. Nhá»¯ng thay Ä‘á»•i má»›i trong module/test cáº§n Ä‘Æ°á»£c review vÃ  lÆ°u vÃ o Git theo quy trÃ¬nh nhÃ³m. KhÃ´ng cáº§n chá»¥p láº¡i nÄƒm ca chá»‰ vÃ¬ cÃ¡c sá»­a chá»¯a á»Ÿ hÃ m save vÃ  Ä‘á»‘i tÆ°á»£ng cáº¥u hÃ¬nh test khÃ´ng thay Ä‘Æ°á»ng tÃ­nh káº¿t quáº£ official v2.
