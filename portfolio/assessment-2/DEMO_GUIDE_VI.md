# Chuẩn bị demo Portfolio Assessment 2

Tệp hướng dẫn này giúp bạn tự học và trình bày. Phần demo trực tiếp chiếm 20 điểm;
báo cáo được tạo không thay thế việc bạn hiểu, chạy và giải thích bài.

## Trình tự trình bày khoảng 6-8 phút

1. **Bài toán và dữ liệu:** UCI Energy Efficiency có 768 cấu hình công trình mô phỏng,
   8 đầu vào. Dự đoán ba nhóm nhu cầu sưởi tương đối. Đã đổi dữ liệu vì concrete trùng
   dataset mẫu Studio 2 bị Studio 3 loại khỏi portfolio. EDA mới nằm trong chính báo cáo.
2. **Target:** chia 537 train và 231 test trước; học hai ngưỡng từ train, khoảng 14,4633
   và 28,4067 đơn vị dataset. Mỗi lớp train có 179 mẫu. Đây không phải nhãn xếp hạng
   năng lượng theo tiêu chuẩn. Không đưa heating load hoặc cooling load vào predictor.
3. **Tiền xử lý:** 6 biến số; orientation và glazing distribution là mã danh mục.
   One-hot tạo tổng cộng 16 cột. Quy tắc scaling dựa trên skew và IQR của từng training
   fold; các dummy 0/1 giữ nguyên. Không fit scaler/selector/PCA trên test.
4. **Feature engineering:** giới thiệu wall/surface, roof/surface, glazing x wall,
   glazing x height, compactness x glazing. Mỗi feature là giả thuyết kỹ thuật;
   tương quan không chứng minh nhân quả. Tích glazing x wall là proxy, không khẳng
   định đó là diện tích cửa kính thật vì metadata UCI không nêu mẫu số của glazing.
5. **Năm Decision Tree:** raw encoded, scaled, engineered scaled, K-Best5 scaled,
   cùng K-Best5 raw. M3 có CV 91,81%, test 95,24%. Scaling không đổi kết quả cây;
   engineering tăng CV khoảng 1,12 điểm phần trăm so với M2. Chọn bộ bằng CV.
6. **Bốn SVM:** baseline, tuning, K-Best, PCA. GridSearch thử 32 cấu hình kernel/C/gamma/degree.
   S3 K-Best được chọn bằng nested CV: 92,36% (SD khoảng 3,34 điểm), test 96,10%.
   K=10 trên 16 cột encoded; RBF C=100, gamma=scale khi refit toàn train.
   PCA giữ 7/16 thành phần, khoảng 91,37% variance; variance không phải độ chính xác.
7. **Giới hạn quan trọng:** tất cả 12 hình dạng công trình đều có trong train và test
   khi chia theo dòng. Khi tách trọn hình dạng chưa thấy, accuracy trung bình giảm còn
   65,28% cho DT và 68,65% cho SVM. Không quảng bá 96,10% như kết quả cho hình dạng mới.
8. **GenAI:** nói đúng phần Codex đã hỗ trợ, phần nào bạn đã tự kiểm tra hoặc sửa.
   Không nói đã tự làm hoặc đã hiểu nếu bạn chưa kiểm chứng.

## Các câu thường được hỏi

**Tại sao không dùng cooling_load để dự đoán heating_load?**
Cooling load là một kết quả mô phỏng liên quan, không phải thông số thiết kế đầu vào
có sẵn cho bài toán này. Đưa nó vào có thể làm bài toán quá dễ và sai phạm vi dự đoán.

**Tại sao phải encode orientation dù lưu là số?**
Mã 2, 3, 4, 5 biểu thị các hướng khác nhau. Khoảng cách số giữa mã không đại diện cho
quan hệ vật lý tuyến tính. One-hot không áp đặt thứ tự đó.

**Tại sao fit scaler trong Pipeline?**
Để mỗi lần CV, median/mean/scale chỉ học từ phần training fold. Validation và test chỉ
dùng transform bằng tham số đã học, tránh rò rỉ thông tin.

**CV và nested CV khác nhau thế nào?**
Vòng trong chọn tham số/k/PCA. Vòng ngoài đánh giá toàn bộ quy trình chọn đó trên
phần dữ liệu không tham gia lựa chọn. Không lấy best_score_ của vòng trong làm kết quả cuối.

**Tại sao K-Best có k=10 khi dữ liệu chỉ có 8 feature?**
Hai feature nominal đã được one-hot thành 10 indicator; cộng 6 số đo thành 16 cột.
K chọn cột trong không gian đã mã hóa, không chọn 10 biến gốc.

**Tại sao PCA có thể kém K-Best?**
PCA giữ phương sai X mà không tối đa hóa phân biệt lớp. Nó xoay các cột sang thành phần
khó diễn giải hơn; hướng phương sai nhỏ vẫn có thể chứa thông tin target.

**Có thể nói tuning cải thiện “có ý nghĩa thống kê” không?**
Không. Bài báo cáo chênh lệch điểm phần trăm và SD giữa fold, không thực hiện kiểm định
giả thuyết được định trước. “Tốt hơn” ở đây là mức quan sát được trong thí nghiệm.

**Vì sao grouped CV giảm mạnh?**
Đây là câu hỏi tổng quát hóa khó hơn: học ở hình dạng này để dự đoán hình dạng khác.
Chỉ 12 hình dạng tạo ít nhóm độc lập và phân phối hình học khác biệt. Cần thêm công
trình độc lập hoặc dữ liệu thực tế trước khi sử dụng ngoài bài học.

## Việc nên tự thực hiện trước demo

- Mở notebook, Restart Kernel and Run All; xem từng bảng và đồ thị.
- Tự giải thích 5 engineered features bằng một dòng dữ liệu cụ thể.
- Kiểm tra một dự đoán trong holdout_predictions.csv và đối chiếu confusion matrix.
- Giải thích vì sao chọn S3 bằng CV, và giới hạn khi dự đoán hình dạng mới.
- Đọc lại khai báo GenAI và sửa trung thực theo phần bạn đã tự kiểm chứng.
