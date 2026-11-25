# NBA Jumpshot Analyzer

Phân tích cú ném rổ từ video bằng computer vision: nhận diện tư thế, đếm số cú ném trúng và trượt, đo các chỉ số cơ học của động tác ném, và so sánh với dáng ném của cầu thủ NBA.

Trạng thái: đang lên kế hoạch. Chưa có code.

## Bài toán

Người tập bóng rổ thường không tự thấy lỗi trong động tác ném của mình, và việc thuê huấn luyện viên phân tích video tốn kém. Dự án này trả lời ba câu hỏi từ một video quay bằng điện thoại:

1. Mỗi cú ném trúng hay trượt?
2. Động tác ném có các chỉ số cơ học như thế nào (góc khuỷu tay, độ khuỵu gối, điểm nhả bóng, follow-through)?
3. Dáng ném giống cầu thủ NBA nào nhất, và khác ở điểm nào?

Input là video quay từ một góc cố định (bên hông hoặc chéo 45°). Output là video có khung xương, bảng thống kê trúng/trượt, chỉ số từng cú ném và báo cáo so sánh.

## Phương pháp

```
video ─▶ pose estimation ─▶ tách từng cú ném ─▶ chỉ số cơ học ─┬─▶ so sánh với cầu thủ NBA
     └─▶ theo dõi bóng và rổ ─▶ trúng / trượt ─────────────────┴─▶ báo cáo
```

### Nhận diện tư thế

MediaPipe Pose trích 33 điểm khớp mỗi khung hình. Toạ độ được làm mượt theo thời gian (bộ lọc One Euro hoặc Savitzky–Golay) để giảm rung, rồi chuẩn hoá theo chiều dài thân người để không phụ thuộc khoảng cách tới camera.

### Tách cú ném

Một cú ném được xác định từ lúc bóng rời vùng thắt lưng đi lên cho tới khi bóng rời tay. Mốc thời gian lấy từ vị trí cổ tay so với vai và khoảng cách giữa bóng với bàn tay.

### Theo dõi bóng và rổ

Một mô hình phát hiện vật thể (YOLO, fine-tune bằng PyTorch) tìm bóng và vành rổ trong từng khung hình. Cú ném tính là trúng khi quỹ đạo bóng đi qua vành rổ từ trên xuống.

### Chỉ số cơ học

| Chỉ số | Cách tính |
|---|---|
| Góc khuỷu tay lúc nhả bóng | Góc vai – khuỷu – cổ tay |
| Độ khuỵu gối | Góc hông – gối – cổ chân nhỏ nhất trước khi bật nhảy |
| Chiều cao điểm nhả bóng | Độ cao cổ tay lúc bóng rời tay, chia cho chiều cao người |
| Thời gian ném | Từ lúc bắt đầu đưa bóng lên tới lúc nhả bóng |
| Follow-through | Góc cổ tay và thời gian giữ tay sau khi nhả bóng |
| Độ ổn định | Độ lệch chuẩn của từng chỉ số qua các cú ném |

### So sánh với cầu thủ NBA

Mỗi cú ném được biểu diễn thành một chuỗi thời gian các góc khớp. Cú ném của người dùng được so với cú ném mẫu của từng cầu thủ bằng Dynamic Time Warping (DTW), để hai động tác có tốc độ khác nhau vẫn so được. Cầu thủ có khoảng cách DTW nhỏ nhất là dáng ném gần nhất. Báo cáo chỉ ra những khớp và giai đoạn có độ lệch lớn nhất.

Các phiên bản sau có thể thay DTW bằng embedding học từ mạng nơ-ron chuỗi thời gian (PyTorch), huấn luyện để các cú ném của cùng một cầu thủ nằm gần nhau.

## Dữ liệu

| Nguồn | Dùng cho |
|---|---|
| Video tự quay | Phát triển và kiểm thử toàn bộ pipeline |
| Video luyện tập và highlight NBA có giấy phép sử dụng | Dựng bộ dáng ném mẫu của cầu thủ |
| [NBA.com Stats: Player Shooting](https://www.nba.com/stats/players/shooting) | Tỉ lệ ném theo khoảng cách, đối chiếu với dáng ném |
| [Kaggle NBA datasets](https://www.kaggle.com/search?q=NBA+dataset) | Dữ liệu bổ sung về vị trí ném |
| Bộ dữ liệu bóng rổ công khai trên Roboflow Universe | Huấn luyện mô hình phát hiện bóng và rổ |

Video NBA có bản quyền chỉ được dùng để trích dữ liệu khớp phục vụ nghiên cứu, không lưu hay phân phối lại trong repo.

## Đánh giá

| Thành phần | Chỉ số | Mục tiêu ban đầu |
|---|---|---|
| Phát hiện trúng/trượt | Accuracy trên video gán nhãn tay | ≥ 90% |
| Phát hiện bóng và rổ | mAP@0.5 | ≥ 0.85 |
| Tách cú ném | Sai số mốc nhả bóng | ≤ 3 khung hình |
| Chỉ số cơ học | Sai số góc so với đo tay trên ảnh | ≤ 5° |
| So sánh cầu thủ | Top-1 accuracy khi nhận diện cầu thủ từ cú ném của chính họ | ≥ 70% |

## Công nghệ

Python, OpenCV, MediaPipe, PyTorch (YOLO), NumPy, Pandas, Streamlit cho giao diện.

## Cấu trúc dự kiến

```
jumpshot/
  pose.py        Trích và làm mượt điểm khớp
  ball.py        Phát hiện và theo dõi bóng, rổ
  segment.py     Tách từng cú ném
  metrics.py     Chỉ số cơ học
  compare.py     DTW và so sánh với cầu thủ
  report.py      Video có khung xương và báo cáo
references/      Dữ liệu khớp đã trích của cầu thủ mẫu
app.py           Giao diện Streamlit
analyze.py       Chạy pipeline từ dòng lệnh
```

## Lộ trình

| Giai đoạn | Nội dung | Trạng thái |
|---|---|---|
| 1 | Trích điểm khớp bằng MediaPipe, vẽ khung xương lên video | Dự kiến |
| 2 | Tách cú ném và tính chỉ số cơ học | Dự kiến |
| 3 | Phát hiện bóng và rổ, đếm trúng/trượt | Dự kiến |
| 4 | Dựng bộ dáng ném mẫu của 5–10 cầu thủ NBA | Dự kiến |
| 5 | So sánh bằng DTW và báo cáo độ lệch | Dự kiến |
| 6 | Giao diện Streamlit: tải video lên, xem kết quả | Dự kiến |
| 7 | Embedding học bằng PyTorch thay cho DTW | Dự kiến |

## Hạn chế dự kiến

- Pose estimation 2D từ một camera mất thông tin chiều sâu, nên góc khớp phụ thuộc góc quay.
- Kết quả kém hơn khi thiếu sáng, chuyển động nhanh gây nhoè, hoặc có nhiều người trong khung hình.
- Dáng ném mẫu của cầu thủ lấy từ video phát sóng, góc quay không đồng nhất với video người dùng.
- Độ giống về hình học chưa chắc tương ứng với hiệu quả ném; đây là công cụ tham khảo, không thay thế huấn luyện viên.
