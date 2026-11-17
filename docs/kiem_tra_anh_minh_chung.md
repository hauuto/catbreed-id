# Kiá»ƒm tra áº£nh minh chá»©ng â€” 16/11/2026

ÄÃ£ xem trá»±c tiáº¿p toÃ n bá»™ 10 PNG trong `outputs/study_1611_pre_v2/screenshots/module_review/`. Káº¿t luáº­n: Ä‘á»§ 5 ca, má»—i ca cÃ³ áº£nh tá»•ng quan vÃ  áº£nh chi tiáº¿t; khÃ´ng cáº§n chá»¥p láº¡i Ä‘á»ƒ Ä‘Ã¡p á»©ng ná»™i dung bÃ n giao.

| Ca | Ná»™i dung Ä‘Ã£ xÃ¡c nháº­n trÃªn áº£nh chá»¥p | Káº¿t luáº­n |
| --- | --- | --- |
| 01 mÃ¨o Ä‘Ãºng | ÄÃºng áº£nh Abyssinian_179 thay tháº¿ khÃ´ng tháº¥y watermark; Abyssinian 96%, KNN -0.9415 â‰¥ -1.0222; cÃ³ Grad-CAM vÃ  bá»‘n kháº£ nÄƒng khÃ¡c | Äáº¡t |
| 02 chÃ³ | ÄÃºng áº£nh chÃ³, tráº¡ng thÃ¡i tá»« chá»‘i; KNN -1.2891 < -1.0222; cÃ³ Top-5 tham kháº£o vÃ  Grad-CAM | Äáº¡t |
| 03 DTD | ÄÃºng áº£nh váº¡ch dá»c; bá»‹ tá»« chá»‘i; KNN -1.3201 < -1.0222; cÃ³ Top-5 vÃ  Grad-CAM | Äáº¡t |
| 04 FeliBreed | ÄÃºng áº£nh ngÆ°á»i dÃ¹ng bÃ¡o tá»« Local; app nháº­n Ragdoll 37%; KNN -1.0043 â‰¥ -1.0222; giá»¯ nguyÃªn káº¿t quáº£ nháº­n sai theo nhÃ£n nguá»“n | Äáº¡t |
| 05 mÃ¨o sai | ÄÃºng áº£nh Abyssinian_121; app nháº­n Maine Coon 59%; KNN -0.9116 â‰¥ -1.0222; cÃ³ Top-5 vÃ  Grad-CAM | Äáº¡t |

Cáº£ nÄƒm áº£nh chi tiáº¿t Ä‘á»u nhÃ¬n tháº¥y Ä‘á»§ 5 Ä‘iá»ƒm vÃ  5 ngÆ°á»¡ng, EfficientNet-B0 / frozen / seed 42, T = 0.6049425684624492, cuda / float32, run `20261116_060136_efficientnet_b0_s42_b8c9dba0` vÃ  SHA checkpoint Ä‘Ã£ khÃ³a. Top-5 cá»§a ca nháº­n gá»“m lá»›p chÃ­nh vÃ  bá»‘n kháº£ nÄƒng khÃ¡c; khÃ´ng yÃªu cáº§u nÄƒm hÃ ng riÃªng dÆ°á»›i lá»›p chÃ­nh.

## LÆ°u Ã½ khi Ä‘Æ°a vÃ o bÃ¡o cÃ¡o

- áº¢nh tá»•ng quan háº¹p khoáº£ng 440â€“470 px; cÃ³ thá»ƒ dÃ¹ng nguyÃªn báº£n trong pháº§n minh há»a, nhÆ°ng trÃ¬nh bÃ y áº£nh chi tiáº¿t riÃªng Ä‘á»ƒ báº£ng vÃ  metadata Ä‘á»c Ä‘Æ°á»£c. KhÃ´ng kÃ©o giÃ£n áº£nh vÆ°á»£t Ä‘á»™ phÃ¢n giáº£i gá»‘c.
- NhÃ£n tháº­t Abyssinian_121 vÃ  Local khÃ´ng thá»ƒ suy ra tá»« tÃªn dá»± Ä‘oÃ¡n trÃªn web; ghi nhÃ£n nguá»“n trong chÃº thÃ­ch hÃ¬nh.
- FeliBreed hiá»ƒn thá»‹ 4578 ms. Giá»¯ nguyÃªn áº£nh; Ä‘Ã¢y lÃ  thá»i gian yÃªu cáº§u Ä‘Æ¡n láº», khÃ´ng pháº£i median/p95 benchmark. KhÃ´ng chá»n láº¡i thá»i gian Ä‘áº¹p Ä‘á»ƒ thay tháº¿ phÃ©p Ä‘o chuáº©n.
- Thá»‘ng kÃª hiá»‡u nÄƒng dÃ¹ng `outputs/study_1611_pre_v2/thanhkhoi/module_review/latency_summary.json`: 50 lÆ°á»£t tá»•ng, bá» 5 Ä‘áº§u, 45 lÆ°á»£t Ä‘o; median 169.75 ms, p95 226.83 ms trÃªn RTX A6000. Pháº¡m vi Ä‘o lÃ  hÃ m predict sau giáº£i mÃ£ áº£nh, khÃ´ng bao gá»“m máº¡ng vÃ  render trÃ¬nh duyá»‡t.
- Kiá»ƒm báº¯t buá»™c 20/20; má»Ÿ rá»™ng 236/238. Hai ca Mahalanobis vÆ°á»£t dung sai lÃ  Bengal_143 vÃ  Siamese_167. KhÃ´ng gá»i 236/238 lÃ  Ä‘á»™ chÃ­nh xÃ¡c phÃ¢n loáº¡i.
- DTD cÃ³ quy trÃ¬nh táº£i do ngÆ°á»i dÃ¹ng cung cáº¥p vÃ  annotation val1 Ä‘á»‘i chiáº¿u; chÆ°a Ä‘á»‘i chiáº¿u byte áº£nh gá»‘c. FeliBreed Local do ngÆ°á»i dÃ¹ng xÃ¡c nháº­n; chÆ°a cÃ³ tÃªn file gá»‘c trong kho Ä‘á»ƒ Ä‘á»‘i chiáº¿u Ä‘á»™c láº­p.

## Viá»‡c cÃ²n láº¡i

1. HoÃ n thiá»‡n ChÆ°Æ¡ng 5 tá»« bá»™ káº¿t quáº£ cuá»‘i; chÃ¨n 5 cáº·p áº£nh, chÃº thÃ­ch nhÃ£n tháº­t/káº¿t quáº£, báº£ng hiá»‡u nÄƒng, dung lÆ°á»£ng vÃ  cÃ¡c giá»›i háº¡n trÃªn.
2. ÄÃ³ng phá»¥ lá»¥c kiá»ƒm khá»›p, benchmark, metadata/hash vÃ  nguá»“n áº£nh; trÃ¡nh láº¥y sá»‘ liá»‡u tá»« gÃ³i thá»­ model_package_1311 hoáº·c cÃ¡c phiÃªn Ä‘o lá»‹ch sá»­.
3. Kiá»ƒm báº£n bÃ¡o cÃ¡o xuáº¥t cuá»‘i: áº£nh/báº£ng Ä‘á»c Ä‘Æ°á»£c, sá»‘ liá»‡u thá»‘ng nháº¥t vÃ  khÃ´ng cÃ²n chá»— trá»‘ng chá» chÃ¨n hÃ¬nh. Sau Ä‘Ã³ má»›i Ä‘Ã¡nh dáº¥u bÃ¡o cÃ¡o/bÃ n giao hoÃ n táº¥t.

Bá»™ áº£nh Ä‘Ã£ Ä‘á»§ Ä‘á»ƒ viáº¿t bÃ¡o cÃ¡o. CÃ¡c JSON cá»§a phiÃªn Ä‘o gá»‘c giá»¯ `chapter5_ready=false` pháº£n Ã¡nh tráº¡ng thÃ¡i khi Ä‘o; tÃ i liá»‡u nÃ y ghi nháº­n bÆ°á»›c xÃ¡c minh áº£nh sau Ä‘Ã³, khÃ´ng sá»­a há»“i tá»‘ báº±ng chá»©ng gá»‘c.
