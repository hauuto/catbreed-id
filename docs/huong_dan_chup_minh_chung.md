# Chá»¥p minh chá»©ng web official v2

Chá»‰ chá»¥p khi API Ä‘Ã£ Ä‘Æ°á»£c khá»Ÿi Ä‘á»™ng láº¡i vá»›i code tÃ­ch há»£p `catbreed_core`. DÃ¹ng model official v2 run `20261116_060136_efficientnet_b0_s42_b8c9dba0`, frozen, seed 42, thiáº¿t bá»‹ cuda. GÃ³i `C:\Users\vyx\Downloads\model_package_1311` lÃ  gÃ³i thá»­ ngÃ y 13/11, khÃ´ng dÃ¹ng chá»¥p cho bÃ¡o cÃ¡o.

## Má»Ÿ web

Táº¡i `D:\LEARNING\Nam4_Ki1\ThiGiacMayTinh\Project\catbreed-id`, náº¿u web vÃ  SSH tunnel Ä‘Ã£ cháº¡y thÃ¬ dÃ¹ng luÃ´n. Khi cáº§n má»Ÿ láº¡i, cháº¡y á»Ÿ hai cá»­a sá»• PowerShell:

```powershell
.\scripts\connect_thanhkhoi_web.ps1
```

```powershell
.\.venv\Scripts\python.exe -m flask --app app run --port 5000
```

Má»Ÿ http://127.0.0.1:5000 vÃ  nháº¥n Ctrl+F5. Má»¥c chi tiáº¿t pháº£i ghi Ä‘Ãºng run á»Ÿ trÃªn, KNN, Ï„ = -1.0221926071848497, T = 0.6049425684624492, cuda/float32. KhÃ´ng cháº¡y API GPU Windows.

## áº¢nh Ä‘áº§u vÃ o vÃ  káº¿t quáº£ cáº§n giá»¯

Táº¥t cáº£ áº£nh cÃ³ sáºµn táº¡i **`D:\LEARNING\Nam4_Ki1\ThiGiacMayTinh\Project\catbreed-id\outputs\official_v2\gpu\cases`**. Chá»n áº£nh gá»‘c JPG dÆ°á»›i Ä‘Ã¢y Ä‘á»ƒ táº£i lÃªn; `dtd.png` lÃ  áº£nh chá»¥p mÃ n hÃ¬nh cÅ©, khÃ´ng pháº£i áº£nh Ä‘áº§u vÃ o.

| Thá»© tá»± | File táº£i lÃªn | NhÃ£n nguá»“n vÃ  káº¿t quáº£ thá»±c táº¿ | TÃªn áº£nh chá»¥p |
| --- | --- | --- | --- |
| 01 | `id_correct.jpg` | Abyssinian_179; nháº­n Ä‘Ãºng Abyssinian | `01_meo_dung_tongquan.png`, `01_meo_dung_chitiet.png` |
| 02 | `dog.jpg` | ChÃ³ Oxford; bá»‹ tá»« chá»‘i | `02_cho_tongquan.png`, `02_cho_chitiet.png` |
| 03 | `dtd_val_sample.jpg` | DTD val1[0], banded theo quy trÃ¬nh ngÆ°á»i dÃ¹ng; bá»‹ tá»« chá»‘i | `03_dtd_tongquan.png`, `03_dtd_chitiet.png` |
| 04 | `felibreed.jpg` | NgÆ°á»i dÃ¹ng xÃ¡c nháº­n thÆ° má»¥c Local; Ä‘Æ°á»£c nháº­n lÃ  Ragdoll | `04_felibreed_tongquan.png`, `04_felibreed_chitiet.png` |
| 05 | `id_error.jpg` | Abyssinian_121; Ä‘Æ°á»£c nháº­n sai thÃ nh Maine Coon | `05_meo_sai_tongquan.png`, `05_meo_sai_chitiet.png` |

## Chá»¥p má»—i ca

1. Táº£i Ä‘Ãºng áº£nh, nháº¥n phÃ¢n tÃ­ch vÃ  chá» káº¿t quáº£ cÃ¹ng Grad-CAM táº£i xong.
2. Chá»¥p tá»•ng quan: áº£nh Ä‘áº§u vÃ o, tÃªn dá»± Ä‘oÃ¡n hoáº·c tráº¡ng thÃ¡i tá»« chá»‘i, Top-5 vÃ  Grad-CAM. Náº¿u trang dÃ i, chá»¥p toÃ n trang hoáº·c thÃªm áº£nh pháº§n dÆ°á»›i.
3. Má»Ÿ â€œChi tiáº¿t mÃ´ hÃ¬nh vÃ  5 Ä‘iá»ƒm tá»« chá»‘iâ€, chá»¥p Ä‘á»§ báº£ng nÄƒm phÆ°Æ¡ng phÃ¡p/Ä‘iá»ƒm/ngÆ°á»¡ng vÃ  thÃ´ng tin model. CÃ³ thá»ƒ chá»¥p thÃªm khi báº£ng khÃ´ng vá»«a mÃ n hÃ¬nh.
4. LÆ°u vÃ o **`D:\LEARNING\Nam4_Ki1\ThiGiacMayTinh\Project\catbreed-id\outputs\official_v2\screenshots\module_review`** theo tÃªn trong báº£ng. Hai áº£nh cÅ© náº±m á»Ÿ thÆ° má»¥c screenshots gá»‘c chÆ°a pháº£i bá»™ minh chá»©ng cuá»‘i.

Má»¥c tiÃªu tá»‘i thiá»ƒu lÃ  10 áº£nh mÃ n hÃ¬nh Ä‘á»§ ná»™i dung. Giá»¯ nguyÃªn ca sai vÃ  FeliBreed Ä‘Æ°á»£c nháº­n. Náº¿u káº¿t quáº£ khÃ¡c báº£ng, giá»¯ áº£nh chá»¥p thá»±c táº¿ vÃ  bÃ¡o láº¡i trÆ°á»›c khi viáº¿t bÃ¡o cÃ¡o. NhÃ£n Local do ngÆ°á»i dÃ¹ng xÃ¡c nháº­n; chÆ°a Ä‘á»‘i chiáº¿u tÃªn file gá»‘c FeliBreed. DTD chÆ°a so byte áº£nh vá»›i kho gá»‘c. NÃªu rÃµ hai giá»›i háº¡n nguá»“n nÃ y trong bÃ¡o cÃ¡o.
