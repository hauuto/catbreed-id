# Báº±ng chá»©ng nghiá»‡m thu web official v2

Báº£n sao cÃ¡c káº¿t quáº£ nhá» Ä‘á»ƒ lÆ°u Git vÃ  táº¡o láº¡i bÃ¡o cÃ¡o. KhÃ´ng chá»©a trá»ng sá»‘ hoáº·c bá»™ dá»¯ liá»‡u huáº¥n luyá»‡n.

- `deployment/`: kiá»ƒm mandatory 20/20, má»Ÿ rá»™ng 236/238, benchmark 50 lÆ°á»£t tá»•ng bá» 5 Ä‘áº§u trÃªn RTX A6000. CÃ¡c tá»‡p pháº£n Ã¡nh phiÃªn Ä‘o gá»‘c, giá»¯ nguyÃªn cá» tráº¡ng thÃ¡i khi Ä‘o.
- `adapter/`: nghiá»‡m thu cuá»‘i sau sá»­a method, 15 test khÃ´ng skip; gÃ³i 13/11 vÃ  official v2 Ä‘á»u 20/20 qua adapter, tá»•ng 10 ca predict Ä‘áº§y Ä‘á»§.
- `core_1311/`: nghiá»‡m thu 11 test core vÃ  Ä‘á»‘i chiáº¿u 20 áº£nh cá»§a gÃ³i thá»­.
- `screenshots/`: 10 áº£nh ngÆ°á»i dÃ¹ng chá»¥p Ä‘Ã£ kiá»ƒm trá»±c tiáº¿p; mÃ¨o Ä‘Ãºng lÃ  Abyssinian_179.
- `screenshots_manifest.json`: SHA-256 tá»«ng áº£nh, giá»¯ tráº¡ng thÃ¡i lÃºc kiá»ƒm áº£nh trÆ°á»›c khi hoÃ n thiá»‡n bÃ¡o cÃ¡o.
- `provenance/`: nguá»“n áº£nh DTD/FeliBreed vÃ  má»©c xÃ¡c minh.

CÃ¡c run_summary lá»‹ch sá»­ cÃ³ thá»ƒ ghi mÃ¨o Ä‘Ãºng cÅ© Abyssinian_101; áº£nh chá»¥p cuá»‘i dÃ¹ng Abyssinian_179. KhÃ´ng thay lá»‹ch sá»­ benchmark theo áº£nh minh há»a má»›i. KhÃ´ng diá»…n giáº£i káº¿t quáº£ model 13/11 thÃ nh sá»‘ Ä‘o official v2.

Táº¡o láº¡i Word báº±ng `scripts/build_official_web_report.py` vá»›i Python cÃ³ python-docx vÃ  Pillow. Bá»™ táº¡o Ä‘á»c sá»‘ liá»‡u vÃ  áº£nh tá»« thÆ° má»¥c báº±ng chá»©ng nÃ y. Cháº¡y `render_docx.py` theo quy trÃ¬nh documents Ä‘á»ƒ kiá»ƒm báº£n xuáº¥t; báº£n PDF bÃ n giao náº±m cÃ¹ng Word trong BaoCao.
