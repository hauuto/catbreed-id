# Ná»™i dung nhÃ¡p ChÆ°Æ¡ng 5 â€“ triá»ƒn khai CatBreedID

Tráº¡ng thÃ¡i: Ä‘Ã£ cÃ³ bá»™ model, kiá»ƒm khá»›p báº¯t buá»™c, 5 ca Ä‘áº§u vÃ o vÃ  sá»‘ Ä‘o trÃªn GPU `thanhkhoi`.
ÄÃ£ xÃ¡c minh Ä‘á»§ 10 áº£nh chá»¥p trÃ¬nh duyá»‡t; xem `docs/kiem_tra_anh_minh_chung.md`.
Cáº§n chÃ¨n cÃ¡c hÃ¬nh vÃ o bÃ¡o cÃ¡o vÃ  kiá»ƒm báº£n xuáº¥t cuá»‘i; Ä‘Ã¢y váº«n lÃ  báº£n nhÃ¡p.

## MÃ´ hÃ¬nh triá»ƒn khai

á»¨ng dá»¥ng gá»“m giao diá»‡n Flask táº¡i cá»•ng 5000 vÃ  API FastAPI táº¡i cá»•ng 8000. API náº¡p EfficientNet-B0
giai Ä‘oáº¡n frozen, seed 42, checkpoint epoch 9 cá»§a lÆ°á»£t `20261116_060136_efficientnet_b0_s42_b8c9dba0`.
Lá»±a chá»n mÃ´ hÃ¬nh, phÆ°Æ¡ng phÃ¡p tá»« chá»‘i vÃ  ngÆ°á»¡ng Ä‘Æ°á»£c Ä‘á»c tá»« locked_selection.json. SHA-256 cá»§a
checkpoint vÃ  cÃ¡c file hiá»‡u chá»‰nh Ä‘Æ°á»£c kiá»ƒm tra trÆ°á»›c khi náº¡p.

áº¢nh Ä‘áº§u vÃ o Ä‘Æ°á»£c chuyá»ƒn vá» RGB; alpha ghÃ©p vá»›i ná»n xÃ¡m 128. áº¢nh Ä‘Æ°á»£c resize cáº¡nh ngáº¯n 256,
crop giá»¯a 224Ã—224 vÃ  chuáº©n hÃ³a theo ImageNet. Khi dÃ¹ng web, áº£nh Ä‘Æ°á»£c xoay theo EXIF Ä‘á»ƒ há»— trá»£
áº£nh Ä‘iá»‡n thoáº¡i. Kiá»ƒm khá»›p vá»›i notebook táº¯t xoay EXIF; Ä‘Ã¢y lÃ  khÃ¡c biá»‡t tiá»n xá»­ lÃ½ cáº§n nÃªu rÃµ.

API tÃ­nh MSP, MaxLogit, Energy, Mahalanobis vÃ  KNN á»Ÿ T=1. Quyáº¿t Ä‘á»‹nh sá»­ dá»¥ng KNN vá»›i
Ï„=-1.0221926071848497; Ä‘iá»ƒm â‰¥ Ï„ Ä‘Æ°á»£c nháº­n. TrÆ°á»ng top_label báº±ng null khi áº£nh bá»‹ tá»« chá»‘i.
XÃ¡c suáº¥t Top-5 dÃ¹ng nhiá»‡t Ä‘á»™ T=0.6049425684624492. Grad-CAM dÃ¹ng gradient cá»§a logit lá»›p dá»± Ä‘oÃ¡n
táº¡i Ä‘áº§u ra cuá»‘i cá»§a features, ReLU vÃ  ná»™i suy song tuyáº¿n vá» 224Ã—224.

## Kiá»ƒm khá»›p vÃ  giá»›i háº¡n sá»‘ há»c

20 áº£nh validation ban Ä‘áº§u Ä‘áº¡t tiÃªu chÃ­ sai sá»‘ theo bÃ n giao trÃªn PyTorch 2.15.0+cu132,
torchvision 0.29.2+cu132, RTX A6000, float32 vÃ  táº¯t TF32. File
outputs/study_1611_pre_v2/thanhkhoi/module_review/mandatory_20.csv lÆ°u tá»«ng áº£nh,
sai sá»‘ logits vÃ  cáº£ 5 Ä‘iá»ƒm. 20 áº£nh Ä‘áº§u Ä‘á»u thuá»™c giá»‘ng Abyssinian, do Ä‘Ã³ Ä‘Ã£ kiá»ƒm tra bá»• sung
toÃ n bá»™ 238 áº£nh validation Ä‘á»ƒ Ä‘Ã¡nh giÃ¡ pháº¡m vi rá»™ng hÆ¡n.

ToÃ n bá»™ 238 áº£nh Ä‘áº¡t dung sai logits vÃ  Ä‘iá»ƒm KNN. Sai sá»‘ logits tuyá»‡t Ä‘á»‘i lá»›n nháº¥t khoáº£ng
6.44Ã—10â»â¶; sai sá»‘ KNN lá»›n nháº¥t khoáº£ng 5.59Ã—10â»â·. Cáº£ 5 Ä‘iá»ƒm Ä‘áº¡t trÃªn 236/238 áº£nh;
Bengal_143 vÃ  Siamese_167 vÆ°á»£t nháº¹ dung sai tÆ°Æ¡ng Ä‘á»‘i Mahalanobis 10â»â¶, láº§n lÆ°á»£t khoáº£ng
1.116Ã—10â»â¶ vÃ  1.452Ã—10â»â¶. Giá»¯ nguyÃªn ngÆ°á»¡ng/dung sai vÃ  ghi nháº­n hai ca nÃ y trong Phá»¥ lá»¥c.
KhÃ´ng diá»…n giáº£i 236/238 lÃ  Ä‘á»™ chÃ­nh xÃ¡c phÃ¢n loáº¡i. KhÃ´ng suy rá»™ng káº¿t quáº£ sang profile CPU.

## CÃ¡c tÃ¬nh huá»‘ng minh há»a

CÃ¡c áº£nh Ä‘Æ°á»£c chá»n tá»« validation hoáº·c chÃ³ trainval trong manifest near_val, khÃ´ng chá»n tá»« test.
Nguá»“n, SHA-256 vÃ  pháº£n há»“i GPU cÃ³ trong outputs/study_1611_pre_v2/thanhkhoi/module_review/review_summary.json.

- Abyssinian_179: Ä‘Æ°á»£c nháº­n Ä‘Ãºng lÃ  Abyssinian. [ChÃ¨n áº£nh Top-5 vÃ  Grad-CAM sau khi xÃ¡c minh UI.]
- american_bulldog_138: bá»‹ tá»« chá»‘i, top_label=null. [ChÃ¨n áº£nh giao diá»‡n.]
- Abyssinian_121: nhÃ£n tháº­t Abyssinian, há»‡ thá»‘ng nháº­n thÃ nh Maine Coon. ÄÃ¢y lÃ  ca sai Ä‘Æ°á»£c giá»¯ nguyÃªn.
- DTD: ngÆ°á»i dÃ¹ng táº¡o tá»« torchvision DTD val1[0], lá»›p banded theo annotation chÃ­nh thá»©c;
  há»‡ thá»‘ng tá»« chá»‘i. ChÆ°a so byte áº£nh vá»›i báº£n gá»‘c DTD.
- FeliBreed: ngÆ°á»i dÃ¹ng xÃ¡c nháº­n áº£nh tá»« thÆ° má»¥c Local; há»‡ thá»‘ng nháº­n sai thÃ nh Ragdoll.
  ChÆ°a cÃ³ tÃªn file gá»‘c Ä‘á»ƒ Ä‘á»‘i chiáº¿u vá»›i kho áº£nh.

## Hiá»‡u nÄƒng triá»ƒn khai

Äo trÃªn Linux 6.8, CPU Xeon Gold 6138, RAM 67,098,542,080 byte, GPU RTX A6000 48 GB.
Suy luáº­n dÃ¹ng PyTorch 2.15.0 CUDA 13.2, má»™t luá»“ng CPU cho cÃ¡c bÆ°á»›c tÃ­nh toÃ¡n CPU.
Cháº¡y Ä‘Ãºng 50 lÆ°á»£t tá»•ng trÃªn áº£nh validation Ä‘Æ°á»£c láº¥y Ä‘á»u theo danh sÃ¡ch ID; bá» 5 lÆ°á»£t Ä‘áº§u,
thá»‘ng kÃª 45 lÆ°á»£t cÃ²n láº¡i.

Tá»•ng thá»i gian cÃ³ median 169.75 ms vÃ  p95 226.83 ms. CÃ¡c cáº·p median/p95 theo ms lÃ :
tiá»n xá»­ lÃ½ 61.78/89.51, suy luáº­n 11.08/13.38, Ä‘iá»ƒm tá»« chá»‘i 5.86/6.43,
Grad-CAM 2.91/3.88 vÃ  háº­u xá»­ lÃ½ 92.13/122.14. KhÃ´ng cá»™ng cÃ¡c phÃ¢n vá»‹ tá»«ng pháº§n Ä‘á»ƒ suy ra phÃ¢n vá»‹ tá»•ng.

Pháº¡m vi Ä‘o báº¯t Ä‘áº§u sau giáº£i mÃ£ áº£nh, káº¿t thÃºc sau táº¡o PNG/Base64; khÃ´ng gá»“m Ä‘á»c file,
upload, mÃ£ hÃ³a JSON hay render trÃ¬nh duyá»‡t. GPU Ä‘Æ°á»£c Ä‘á»“ng bá»™ táº¡i cÃ¡c má»‘c Ä‘o. Grad-CAM gá»“m
backward, chuáº©n hÃ³a vÃ  chuyá»ƒn dá»¯ liá»‡u vá» CPU; mÃ£ hÃ³a PNG tÃ­nh vÃ o háº­u xá»­ lÃ½.
ÄÃ¢y khÃ´ng pháº£i phÃ©p Ä‘o Ä‘á»™ trá»… tá»« thao tÃ¡c ngÆ°á»i dÃ¹ng Ä‘áº¿n khi trÃ¬nh duyá»‡t hiá»‡n káº¿t quáº£.

Checkpoint cÃ³ kÃ­ch thÆ°á»›c 16,390,147 byte; ngÃ¢n hÃ ng/tham sá»‘ khoáº£ng cÃ¡ch cÃ³ kÃ­ch thÆ°á»›c
21,994,386 byte. Tá»•ng hai file lÃ  38,384,533 byte (~38.38 MB). Cáº§n cáº¥p cáº£ file KNN khi triá»ƒn khai;
kÃ­ch thÆ°á»›c file khÃ´ng Ä‘á»“ng nghÄ©a lÆ°á»£ng RAM/VRAM sá»­ dá»¥ng.

## Phá»¥ lá»¥c cáº§n Ä‘Ã­nh kÃ¨m

outputs/study_1611_pre_v2/thanhkhoi/module_review/mandatory_20.csv vÃ  summary; káº¿t quáº£ má»Ÿ rá»™ng 238 áº£nh vÃ  hai ca Mahalanobis;
latency_raw.csv, latency_measured.csv, latency_summary.json; review_summary.json vÃ  áº£nh chá»¥p giao diá»‡n Ä‘Æ°á»£c xÃ¡c minh;
thÃ´ng tin SHA-256/checkpoint/run vÃ  nguá»“n cÃ¡c áº£nh minh há»a.
