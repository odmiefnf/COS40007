# Hướng dẫn đọc và chuẩn bị demo Portfolio 3

Đây là tài liệu tham khảo do Codex tạo. Đề cấm GenAI tạo toàn bộ mã, nên không được
trình bày tài liệu này như bằng chứng bạn tự viết bài. Bạn cần tự làm phần công việc
được phép, hiểu và kiểm tra kết quả của mình. Demo trực tiếp chiếm 20 điểm và chưa
được thực hiện bởi tài liệu này.

## Mạch trình bày khoảng 5 phút

1. **Bài toán:** dự đoán mức nhu cầu sưởi Lower/Middle/Higher của 768 cấu hình nhà
   mô phỏng từ UCI Energy Efficiency, tiếp tục bản Energy của Portfolio 2.
2. **Nhãn:** dùng đúng ngưỡng PA2: 14.463333 và 28.406667. Y1 tạo nhãn; Y1 và Y2
   không xuất hiện trong đầu vào. Đây không phải nhãn xếp hạng năng lượng chính thức.
3. **Chia dữ liệu:** giữ nguyên 537 development / 231 comparison của PA2. Trong 537,
   lấy 429 fit và 108 validation. Tất cả mô hình chính dùng cùng 429 hàng để fit.
4. **Tiền xử lý:** 6 biến số được StandardScaler; orientation và glazing distribution
   được one-hot. Tổng 16 đầu vào. Fit scaler/encoder bằng fit-only, và fit lại bên
   trong từng fold CV để tránh rò rỉ.
5. **Part A:** Random Forest 300 cây và sklearn MLP. Mỗi loại thử 4 cấu hình,
   stratified CV 5 fold. RF được chọn bằng CV accuracy 91.84%, SD 1.29 điểm phần trăm.
6. **Part B:** ANN 64-32-16, ReLU, dropout 0.2 sau hai lớp đầu, softmax 3 đầu ra.
   Tổng 3,747 tham số. Adam 0.001; sparse categorical crossentropy. Không dùng LSTM
   vì dữ liệu không có thứ tự thời gian; không dùng CNN vì không phải ảnh.
7. **Thử nghiệm:** batch 16/32/64, cùng seed 42, tối đa 50 epochs. Early stopping
   patience 10; giảm learning rate một nửa nếu val_loss chững 5 epochs. Chọn bằng
   val_loss, không bằng comparison accuracy.
8. **Kết quả:** RF 93.07%; sklearn MLP 82.68%; ANN batch16 86.15%, batch32 82.68%,
   batch64 83.55%. ANN được chọn là batch16, best epoch 23, dừng sau epoch 33.
   Weighted F1: RF 0.9300; ANN được chọn 0.8585. RF hơn ANN 6.93 điểm phần trăm.
9. **Giới hạn:** cả 12 hình dạng nhà đều xuất hiện ở hai bên cách chia ngẫu nhiên.
   Khi tách toàn bộ hình dạng bằng grouped stress check: RF trung bình 64.06%,
   MLP 67.84%, ANN batch32 69.27%. Đây là cấu hình cố định, không phải grouped tuning.
   Test của PA2 đã được xem trước đó, nên ở đây gọi là reused comparison set.
10. **Kết luận:** RF tốt nhất cho so sánh ngẫu nhiên này. Chưa đủ căn cứ triển khai
    thực tế; cần dữ liệu nhà thật độc lập, kiểm tra theo geometry, calibration và
    xác định chi phí sai nhãn. Trình bày trung thực mức hỗ trợ GenAI.

## Câu hỏi cần tự trả lời trước demo

- **Accuracy khác weighted F1 thế nào?** Accuracy là tỷ lệ đúng. Weighted F1 lấy
  trung bình F1 từng lớp có trọng số theo số mẫu. Macro F1 cho mỗi lớp trọng số bằng nhau.
- **Softmax khác dự đoán lớp thế nào?** Softmax trả ba xác suất tổng gần 1. Argmax
  lấy chỉ số xác suất cao nhất thành lớp. Crossentropy phạt xác suất sai quá tự tin.
- **Vì sao loss và accuracy không luôn đi cùng nhau?** Xác suất có thể thay đổi mà
  argmax giữ nguyên; loss vẫn đổi dù accuracy không đổi.
- **Validation có phải CV không?** Không. Validation là 108 hàng cố định dùng chọn
  batch/checkpoint. CV gồm 5 vòng fit/đánh giá trong 429 hàng fitting cho Part A.
- **Vì sao MLP và ANN đều là neural network?** Tên Part A/Part B theo rubric.
  sklearn MLP có cách regularisation và training khác ANN Keras, nhưng không phải
  đối lập về bản chất giữa học máy và mạng nơ-ron.
- **Vì sao EarlyStopping patience10 chạy đến33 nếu best23?** Dừng sau 10 epochs
  không cải thiện val_loss; checkpoint nạp lại epoch23, không dùng trọng số epoch33.
- **Có chứng minh dropout giúp tăng điểm không?** Không. Chưa có ablation không
  dropout. Kết quả chỉ so sánh batch size khi kiến trúc giữ nguyên.
- **Vì sao nhóm geometry kém hơn?** Mô hình phải xử lý hình dạng chưa thấy, thay vì
  biến thể orientation/glazing của hình dạng đã biết. Đây là câu hỏi tổng quát hóa khác.
- **Tệp .h5 đã đủ để dự đoán chưa?** Cần ann_preprocessor.pkl, thứ tự 8 feature nguồn,
  16 feature sau encode và class order trong model_metadata.json.
- **GenAI đã làm gì?** Tạo phần lớn thiết kế, mã, chạy và phân tích. Không được nói
  chỉ dùng để sửa lỗi hoặc nói đã tự viết nếu không đúng. Xem GENAI_EVIDENCE.md.

## Mở và chạy

Mở notebook trong thư mục gói, chọn Python tương thích requirements.txt và Restart
Kernel / Run All. Có thể kiểm tra độc lập bằng `python tools/verify_analysis.py`.
Ví dụ dự đoán: `python tools/predict.py --input data/energy_data.csv --output predictions.csv`.

Các hình cần nắm: `01_data_and_classes.png`, `02_ann_learning_curves.png`,
`03_master_comparison.png`, `04_confusion_matrices.png`, `06_geometry_stress.png`.
Đọc lỗi cụ thể trong confusion matrix, không chỉ học thuộc accuracy.
