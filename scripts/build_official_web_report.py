"""Build the evidence based official web report with the bundled document runtime."""
import csv
import json
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'BaoCao/BaoCao_TrienKhai_WebApp_OfficialV2_Vy_2026-11-16.docx'
E = ROOT / 'docs/evidence/official_v2/deployment'
S = ROOT / 'docs/evidence/official_v2/screenshots'
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
lat = read(E / 'latency_summary.json')
mandatory = read(E / 'mandatory_20.summary.json')
extended = read(E / 'all_validation.summary.json')
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
sec.top_margin, sec.bottom_margin = Inches(.7), Inches(.65)
sec.left_margin, sec.right_margin = Inches(.7), Inches(.7)
for name in ('Normal', 'Title', 'Subtitle', 'Heading 1', 'Heading 2', 'Caption'):
    style = doc.styles[name]
    style.font.name = 'Arial'
    style.font.color.rgb = RGBColor(0, 0, 0)
    style.font.size = Pt(11 if name == 'Normal' else 10 if name == 'Caption' else 14)
    style.paragraph_format.space_after = Pt(7)
    style.paragraph_format.line_spacing = 1.12
    style.paragraph_format.keep_with_next = False
    for border in list(style.element.xpath('.//w:pBdr')):
        border.getparent().remove(border)
doc.styles['Title'].font.size = Pt(23)
doc.styles['Heading 1'].font.size = Pt(16)
doc.styles['Heading 2'].font.size = Pt(12)
doc.styles['Heading 1'].paragraph_format.keep_with_next = True
doc.styles['Heading 2'].paragraph_format.keep_with_next = True
doc.styles['Normal'].paragraph_format.widow_control = True
header = sec.header.paragraphs[0]
header.text = 'CATBREEDID  |  TRIỂN KHAI WEB OFFICIAL V2'
header.runs[0].font.size = Pt(8)
footer = sec.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
footer.add_run('Vỹ  •  16/11/2026  |  Trang ').font.size = Pt(8)
field = OxmlElement('w:fldSimple'); field.set(qn('w:instr'), 'PAGE'); footer._p.append(field)

def p(text, style=None):
    return doc.add_paragraph(text, style)

def h(text):
    doc.add_heading(text, level=1)

def page(text):
    heading = doc.add_heading(text, level=1)
    heading.paragraph_format.page_break_before = True

def table(headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    if widths:
        for col, width in zip(t.columns, widths): col.width = Inches(width)
    for c, value in zip(t.rows[0].cells, headers): c.text = str(value)
    repeat = OxmlElement('w:tblHeader'); t.rows[0]._tr.get_or_add_trPr().append(repeat)
    for row in rows:
        for c, value in zip(t.add_row().cells, row): c.text = str(value)
    for i, row in enumerate(t.rows):
        for j, c in enumerate(row.cells):
            if widths: c.width = Inches(widths[j])
            c.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            props = c._tc.get_or_add_tcPr()
            shade = OxmlElement('w:shd'); shade.set(qn('w:fill'), 'DBE7ED' if i == 0 else 'F3F6F8' if i % 2 == 0 else 'FFFFFF'); props.append(shade)
            borders = OxmlElement('w:tcBorders')
            for edge in ('top', 'left', 'bottom', 'right'):
                tag = OxmlElement('w:' + edge); tag.set(qn('w:val'), 'single'); tag.set(qn('w:sz'), '4'); tag.set(qn('w:color'), 'D9D9D9'); borders.append(tag)
            props.append(borders)
            margins = OxmlElement('w:tcMar')
            for edge in ('top','left','bottom','right'):
                tag=OxmlElement('w:'+edge); tag.set(qn('w:w'),'80'); tag.set(qn('w:type'),'dxa'); margins.append(tag)
            props.append(margins)
            for para in c.paragraphs:
                para.paragraph_format.space_after = Pt(3)
                para.paragraph_format.space_before = Pt(3)
                for run in para.runs:
                    run.font.size=Pt(9)
                    run.bold=i==0
    p('')
    return t

p('Báo cáo triển khai và kiểm chứng web CatBreedID', 'Title')
p('Nội dung phục vụ Chương 5 và rà soát bàn giao bổ sung', 'Subtitle')
p('Người thực hiện Vỹ     Ngày lập 16 tháng 11 năm 2026')
p('Ứng dụng đã tích hợp module suy luận dùng chung catbreed_core, nạp gói EfficientNet-B0 official v2, trả quyết định từ chối KNN, xác suất Top-5 đã hiệu chỉnh và bản đồ Grad-CAM. Kiểm bắt buộc đạt 20/20 ảnh; kiểm mở rộng đạt đồng thời năm điểm trên 236/238 ảnh. Đã xác minh đủ 10 ảnh giao diện cho năm tình huống.')
p('Báo cáo ghi nhận kết quả triển khai thực tế và những mục chưa được xác nhận đầy đủ trong bàn giao. Hai sai lệch Mahalanobis và giới hạn xác minh nguồn ảnh ngoại lai được giữ nguyên. Các số liệu từ gói thử ngày 13/11 không được dùng làm kết quả model official v2.')
h('1 Rà soát bàn giao bổ sung')
table(['Yêu cầu', 'Kết quả rà soát'], [
('Dùng module chung', 'Đã gọi load_package, predict_arrays và decide trong adapter app.'),
('EXIF và Top-5', 'Kiểm khớp exif=False; web exif=True; Top-5 lấy từ probs đã hiệu chỉnh.'),
('Từ chối và metadata', 'KNN theo khóa, top_label=null khi từ chối; đủ T, τ, seed, run và SHA.'),
('Grad-CAM và gói smoke', 'Grad-CAM dùng model chung; adapter từ chối Package.is_smoke_test.'),
('Kiểm khớp và ảnh chụp', '20/20 bắt buộc; 236/238 mở rộng; đủ năm cặp ảnh đã xem trực tiếp.'),
('Đúng đường dẫn checker', 'Đã thêm app/consistency_check.py gọi checker hiện có, cùng logic app.'),
('Quy trình Git và test gốc', 'Đã merge inference-core tại d649df4. Bộ core có 11 test đạt; cả bộ hồi quy gồm 15 test đạt, không skip.'),
], [2.05,5.05])
p('Đã bổ sung file tham chiếu cho gói 13/11 theo SHA d90bca48 và kiểm qua chính adapter app trên GPU: 20/20 ảnh đạt. Gói không có locked_selection nhận method=KNN tường minh. Gói official v2 vẫn chọn KNN trong khóa khi truyền thử method=MSP. Các kết quả gói thử chỉ dùng nghiệm thu code, không dùng làm số liệu model triển khai.')

page('2 Kiến trúc và quy trình suy luận')
p('Giao diện Flask phục vụ trên máy người dùng ở cổng 5000. Trình duyệt gửi ảnh tới API FastAPI thông qua cổng 8000, được SSH chuyển tiếp sang API GPU trên tài khoản thanhkhoi. Mô hình chạy trên RTX A6000; máy Windows hiển thị giao diện và không thực hiện suy luận GPU trong phiên được báo cáo.')
p('Adapter app nạp Package từ một thư mục cấu hình. load_package đọc checkpoint, ngưỡng, nhiệt độ và ngân hàng khoảng cách, kiểm mã băm và dựng đúng kiến trúc. predict_arrays trả logits thô, đặc trưng và năm điểm ở T=1. decide lấy phương pháp trong locked_selection, so điểm với ngưỡng và tính xác suất đã hiệu chỉnh.')
table(['Thành phần', 'Cấu hình triển khai'], [
('Mô hình', 'EfficientNet-B0; frozen; seed 42; epoch 9'),
('Run', lat['run_id']),
('Phương pháp chính', 'KNN; k=10; τ = -1.0221926071848497'),
('Nhiệt độ Top-5', 'T = 0.6049425684624492'),
('Tiền xử lý', 'RGB; alpha ghép nền xám 128; resize cạnh ngắn 256 bilinear, antialias; CenterCrop 224; chuẩn hóa ImageNet'),
('Quyết định', 'accepted nếu score ≥ τ; rejected nếu score < τ. Khi rejected, top_label=null.'),
], [1.8,5.3])
p('EXIF bị tắt khi kiểm khớp để giữ cách đọc ảnh của notebook. Web bật xoay EXIF nhằm hiển thị ảnh điện thoại đúng hướng. Hai chế độ được ghi rõ vì có thể cho tensor đầu vào khác nhau với ảnh có cờ xoay.')
table(['Điểm ở T bằng 1', 'Cách tính và quy ước'], [
('MSP', 'Xác suất softmax lớn nhất từ logits thô.'),
('MaxLogit', 'Logit lớn nhất.'),
('Energy', 'Logsumexp của logits.'),
('Mahalanobis', 'Âm khoảng cách bình phương nhỏ nhất đến trung bình lớp theo ma trận precision; chặn giá trị bình phương dưới 0.'),
('KNN', 'Âm khoảng cách Euclid đến láng giềng thứ k trong ngân hàng đặc trưng train đã chuẩn hóa L2.'),
], [1.8,5.3])
p('Mọi điểm đều dùng quy ước lớn hơn là gần dữ liệu huấn luyện hơn. Bảng nâng cao hiển thị cả năm điểm và ngưỡng T1, nhưng quyết định triển khai chỉ dùng KNN. Grad-CAM lấy gradient logit lớp dự đoán tại features cuối, gộp gradient theo không gian, ReLU và nội suy về 224 × 224.')

page('3 Kiểm khớp với notebook')
p('Kiểm tra gọi trực tiếp cùng adapter mà API sử dụng, đối chiếu logits với val.npz và năm điểm với validation_scores.csv theo image_id. Ảnh gốc thuộc validation Oxford; exif=False. Cả mã và gói dùng trên máy thanhkhoi đã được ghi SHA-256 trong review_summary.json.')
table(['Phạm vi', 'Kết quả', 'Diễn giải'], [
('20 ảnh bắt buộc', '20/20 đạt', '20 ảnh đầu thuộc Abyssinian; không đại diện đầy đủ 12 giống.'),
('238 ảnh validation', '236/238 đạt cả 5 điểm', 'Logits và KNN đạt 238/238; hai ảnh chỉ lệch Mahalanobis.'),
('Ngưỡng số học', 'abs < 0.0001', 'Mahalanobis cho phép thêm relative < 0.000001 theo phép OR.'),
], [1.65,1.8,3.65])
table(['Đại lượng', 'Sai số tuyệt đối lớn nhất trên 238 ảnh'], [('Logits', f"{extended['max_logits_abs']:.9g}"), *[(name, f'{value:.9g}') for name,value in extended['max_scores_abs'].items()]], [2.0,5.1])
with (E/'all_validation.csv').open(encoding='utf-8') as stream:
    failures=[r for r in csv.DictReader(stream) if r['image_id'] in ('Bengal_143','Siamese_167')]
table(['Ảnh', 'Sai số tương đối Mahalanobis', 'Đánh giá'], [(r['image_id'],f"{float(r['Mahalanobis_relative']):.9g}",'Vượt 1e-6') for r in failures],[1.7,3.5,1.9])
p('Không nới dung sai để đánh dấu hai ca trên là đạt. Kết quả 236/238 thể hiện mức khớp số giữa các phép suy luận, không phải độ chính xác phân loại. KNN và logits của toàn bộ mẫu kiểm mở rộng vẫn nằm trong dung sai; chưa có chứng minh khớp từng bit giữa các phần cứng hoặc phiên bản thư viện.')
p('API đã được khởi động lại sau sửa mã và kiểm đủ năm ảnh qua HTTP. Mỗi phản hồi có năm xác suất Top-5, năm điểm/ngưỡng, metadata CUDA float32, ảnh đầu vào mô hình và Grad-CAM kích thước 224 × 224. Ảnh mèo đúng sau đó đổi từ Abyssinian_101 sang Abyssinian_179 để bỏ watermark; có phản hồi HTTP mới riêng, không sửa lịch sử benchmark.')

page('4 Hiệu năng và giới hạn triển khai')
p('Đo trên máy megalab-HP-Z8-G4-Workstation bằng tài khoản thanhkhoi. Máy dùng Linux 6.8.0-138, Xeon Gold 6138, 40 CPU logic, RAM 67,098,542,080 byte, GPU NVIDIA RTX A6000 48 GB. Môi trường Python 3.13.18, PyTorch 2.15.0+cu132, torchvision 0.29.2+cu132, CUDA 13.2, float32, tắt TF32 và một luồng CPU cho tính toán.')
p('Giao thức gồm đúng 50 lượt từ danh sách validation lấy đều theo thứ tự ID, bỏ năm lượt đầu làm nóng và tính thống kê trên 45 lượt còn lại. Bộ dùng cho bảng là outputs/official_v2/thanhkhoi/module_review; không lấy các số đo lịch sử của Windows hoặc phiên module_integration.')
table(['Giai đoạn', 'Median ms', 'p95 ms'], [(label,f"{lat['latency_ms'][key]['median']:.2f}", f"{lat['latency_ms'][key]['p95']:.2f}") for key,label in [('preprocess','Tiền xử lý'),('inference','Suy luận'),('rejection','Điểm từ chối và quyết định'),('gradcam','Grad-CAM'),('postprocess','Hậu xử lý PNG Base64'),('total','Tổng')]], [3.5,1.8,1.8])
p('Phạm vi đo bắt đầu sau giải mã ảnh PIL, kết thúc khi tạo xong ảnh PNG/Base64. CUDA được đồng bộ ở các mốc đo. Tổng được đo trực tiếp, không cộng median từng giai đoạn. Thời gian không bao gồm đọc file, upload, mạng SSH, mã hóa JSON và render trình duyệt. Vì vậy không dùng bảng này để khẳng định độ trễ từ thao tác người dùng đến khi màn hình xuất hiện kết quả.')
table(['File', 'Dung lượng byte', 'Dung lượng MB thập phân'], [('checkpoint.pt','16,390,147','16.39'),('distance_parameters.npz','21,994,386','21.99'),('Tổng hai file','38,384,533','38.38')],[3.1,2,2])
p('Ngân hàng KNN là một phần bắt buộc của gói chạy. Dung lượng file không đồng nghĩa mức sử dụng RAM hoặc VRAM. Ảnh FeliBreed trên trình duyệt hiển thị 4578 ms cho một yêu cầu riêng; ảnh đó được giữ nguyên, không thay bằng một lượt có thời gian ngắn hơn.')
p('Năm ví dụ tiếp theo minh họa hành vi quan sát được, gồm cả hai ca nhận sai theo nhãn nguồn. Chúng không tạo thành phép đánh giá toàn bộ DTD hay FeliBreed. Nguồn ảnh DTD dựa trên quy trình người dùng và annotation val1; ảnh Local dựa trên xác nhận người dùng, chưa được đối chiếu độc lập với tên file gốc trong kho.')

cases = [
('01_meo_dung','5 Mèo validation được nhận đúng','Abyssinian_179 thuộc validation Oxford. Ảnh thay thế được chọn vì không thấy watermark; nhãn thật Abyssinian và API nhận đúng Abyssinian khoảng 96%. Điểm KNN -0.9415 lớn hơn ngưỡng -1.0222 nên hệ thống chấp nhận.', 'id_correct.jpg'),
('02_cho','6 Ảnh chó bị từ chối','Ảnh american_bulldog_138 thuộc nguồn chó Oxford trainval trong manifest near_val. KNN -1.2891 thấp hơn ngưỡng -1.0222 nên hệ thống từ chối và top_label=null. Những xác suất giống mèo trên giao diện chỉ là các giả thuyết tham khảo khi buộc mạng so sánh trong 12 lớp.', 'dog.jpg'),
('03_dtd','7 Ảnh kết cấu DTD bị từ chối','Ảnh vạch dọc do người dùng tạo bằng torchvision DTD với split val, partition 1 và phần tử số 0. Annotation val1 chính thức cho biết ID đầu là banded/banded_0006.jpg. KNN -1.3201 thấp hơn ngưỡng nên ảnh bị từ chối. Chưa so byte ảnh với bản gốc DTD.', 'dtd_val_sample.jpg'),
('04_felibreed','8 Ảnh FeliBreed được nhận thành Ragdoll','Người dùng xác nhận ảnh lấy từ thư mục Local của FeliBreed. Hệ thống vẫn chấp nhận vì KNN -1.0043 lớn hơn -1.0222 và dự đoán Ragdoll khoảng 37%. Đây là ca nhận sai theo nhãn nguồn Local, cho thấy giới hạn của cơ chế từ chối. Chưa có tên file gốc để xác minh độc lập danh tính ảnh trong kho.', 'felibreed.jpg'),
('05_meo_sai','9 Mèo validation bị dự đoán sai','Ảnh Abyssinian_121 có nhãn thật Abyssinian nhưng hệ thống chấp nhận và dự đoán Maine Coon khoảng 59%. KNN -0.9116 lớn hơn -1.0222. Ca này minh họa ảnh gần phân bố huấn luyện vẫn có thể bị phân loại sai; Grad-CAM không bảo đảm nhãn dự đoán là đúng.', 'id_error.jpg'),
]
from PIL import Image
for index,(prefix,title,description,input_name) in enumerate(cases,1):
    page(title)
    p(description)
    p('Ảnh đầu vào ' + input_name + ' trong outputs/official_v2/gpu/cases. Hai ảnh bên dưới lần lượt là tổng quan giao diện và chi tiết điểm cùng mô hình.', 'Caption')
    para=doc.add_paragraph(); para.alignment=WD_ALIGN_PARAGRAPH.CENTER
    for suffix in ('tongquan','chitiet'):
        path=S/(prefix+'_'+suffix+'.png')
        with Image.open(path) as im: w,h_px=im.size
        height=min(5.45,2.98*h_px/w)
        width=height*w/h_px
        para.add_run().add_picture(str(path),width=Inches(width),height=Inches(height))
        if suffix=='tongquan': para.add_run('  ')
    cap=p(f'Hình {index}  Tổng quan và chi tiết của ca minh chứng', 'Caption');cap.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p('Mô hình hiển thị EfficientNet-B0 / frozen / 42, cuda / float32; cùng run và checkpoint official v2. Bảng chi tiết dùng năm ngưỡng T1; quyết định chính lấy từ KNN.', 'Caption')

page('10 Phụ lục và trạng thái bàn giao')
p('Các đường dẫn dưới đây tính từ thư mục gốc dự án. Bộ ảnh đã được xem trực tiếp và lưu mã băm từng PNG trong module_review_manifest.json. Báo cáo này là bản nội dung triển khai phục vụ Chương 5; chưa thay thế các chương dữ liệu, huấn luyện và đánh giá toàn bộ của báo cáo đề tài.')
table(['Bằng chứng', 'Đường dẫn'], [
('Kiểm bắt buộc', 'outputs/official_v2/thanhkhoi/module_review/mandatory_20.csv và .summary.json'),
('Kiểm mở rộng', 'outputs/official_v2/thanhkhoi/module_review/all_validation.csv và .summary.json'),
('Benchmark', 'outputs/official_v2/thanhkhoi/module_review/latency_raw.csv, latency_measured.csv, latency_summary.json'),
('Năm ca HTTP', 'outputs/official_v2/thanhkhoi/module_review/*_http.json; mèo đúng dùng id_correct_replacement_http.json'),
('Ảnh chụp', 'outputs/official_v2/screenshots/module_review/ và module_review_manifest.json ở thư mục cha'),
('Nguồn ảnh', 'outputs/official_v2/provenance/external_sources.json; gpu/cases/cases.json'),
('Kiểm gói 13/11 và hồi quy', 'outputs/official_v2/adapter_trial_review/: pytest.xml, pytest.log, trial_adapter_20.csv, official_adapter_20.csv, adapter_review.json'),
],[1.6,5.5])
doc.add_heading('Thông tin truy vết checkpoint',level=2)
p('SHA-256 checkpoint official v2 ' + lat['checkpoint_sha256'])
p('Run ' + lat['run_id'])
doc.add_heading('Cách chạy lại kiểm khớp',level=2)
p('Trên máy thanhkhoi, tại thư mục dự án, dùng đúng môi trường model và ảnh Oxford đã cung cấp. Lệnh entry point: python app/consistency_check.py --package model_package --images inputs/oxford --device cuda --count 20 --output app_consistency.csv. Đổi --count 0 để kiểm toàn bộ validation. Với gói thử không có khóa, truyền thêm --method KNN. Phiên GPU đã lưu gọi cùng hàm kiểm trong scripts/check_app_consistency.py qua adapter app.')
page('11 Kết quả nghiệm thu và giới hạn còn lại')
p('Các yêu cầu chức năng và kiểm thử trong bàn giao bổ sung đã có bằng chứng: merge d649df4, module chung, chặn smoke, EXIF theo từng chế độ, phương pháp từ khóa, Top-5, Grad-CAM và metadata. Bộ hồi quy cuối có 15 test đạt, không skip; gồm 11 test core, 2 test adapter và 2 test giao thức benchmark. Tham chiếu 13/11 do người dùng cung cấp cục bộ được dùng đúng cặp SHA d90bca48; phép assert số học của test gốc được giữ nguyên, không dùng mock mạng hay token giả để bỏ qua dữ liệu thật.')
p('Đã sửa thiếu DistanceScores.save và việc predict chưa truyền method tường minh vào decide cho gói không có khóa. Kiểm gói 13/11 và official v2 qua adapter đều đạt 20/20. Truyền MSP vào gói official không ghi đè KNN đã khóa. Kết quả nghiệm thu mới được lưu riêng, không thay thế số benchmark của model official v2.')
p('Hai sai lệch Mahalanobis trong 238 ảnh mở rộng và mức xác minh nguồn DTD/FeliBreed còn là giới hạn đã công bố. Chưa khẳng định đạt toàn bộ năm điểm trên 238/238 ảnh; chưa xác minh byte ảnh DTD hoặc tên file gốc FeliBreed trong kho. Hai ảnh ngoài tập chỉ là ca minh họa, không đại diện hiệu quả từ chối trên toàn bộ dữ liệu ngoại lai.')
p('Mã, tài liệu và bản báo cáo đã được rà soát để lưu theo phạm vi bàn giao web. Bằng chứng nhỏ và 10 ảnh chụp được lưu tại docs/evidence/official_v2 để kiểm lại và tạo lại báo cáo. Trọng số, ngân hàng KNN, môi trường Python và các bộ dữ liệu được cấp riêng, không đưa vào commit bàn giao.')
doc.add_heading('Tài liệu đối chiếu',level=2)
p('BanGiao_WebApp_Vy_2026-11-16.md; BanGiao_WebApp_Vy_BoSung_2026-11-16.md; docs/official_v2_web_validation.md; docs/kiem_tra_anh_minh_chung.md. Nguồn bộ ảnh được ghi trong manifest: https://www.robots.ox.ac.uk/~vgg/data/dtd/ và https://data.mendeley.com/datasets/ng6tv57j5c/2. Các URL này ghi nguồn dữ liệu, không có nghĩa ảnh Local đã được so khớp độc lập với kho.')
p('Bằng chứng nghiệm thu cuối: docs/evidence/official_v2/adapter/pytest.xml và adapter_review.json; docs/kiem_tra_adapter_1311_sau_sua.md; docs/ket_qua_nghiem_thu_core_1311.md. Các đường dẫn outputs ở phụ lục phản ánh nơi sinh kết quả; bản sao để lưu Git nằm trong docs/evidence/official_v2.')

OUT.parent.mkdir(exist_ok=True)
doc.save(OUT)
assert len(doc.inline_shapes)==10
print(OUT)
