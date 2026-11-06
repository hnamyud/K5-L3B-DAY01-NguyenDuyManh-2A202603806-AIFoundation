# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:**
- **Họ và tên:**
- **SEED** (in ở ô đầu notebook):

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | x |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | x |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | x |

## 2. Kết quả chính

*Chép bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin) của chính bạn.*

| Mô hình | F1 test thường (SUB) | F1 50 case khó | Acc 50 case khó |
|---|---:|---:|---:|
| 1. BoW + LogReg | 0.824 | 0.256 | 0.473 |
| 2. BoW + MLP | 0.923 | 0.333 | 0.491 |
| 3a. Emb + kNN | 0.976 | 0.698 | 0.655 |
| 3b. Emb + LogReg | 0.974 | 0.582 | 0.582 |
| 3c. Emb + MLP head | 1.000 | 0.655 | 0.655 |
| 4a. LLM nhỏ zero-shot | 0.927 | 0.970 | 0.964 |
| 4d. LLM nhỏ + prompt phòng thủ | 0.927 | 0.985 | 0.982 |


## 3. Dự đoán và thực tế

### Dự đoán 1: Vị trí của LLM trên đường Pareto

- **Dự đoán:** LLM (đặc biệt là Few-shot) sẽ nằm trên đường Pareto ở cực "ít lỗi nhất" vì năng lực hiểu ngôn ngữ vượt trội.
- **Thực tế:** Toàn bộ các mô hình LLM bị loại khỏi đường Pareto và bị lấn át hoàn toàn bởi Encoder (MiniLM) + MLP head.
- **Vì sao lệch:** Chưa lường trước được độ trễ quá lớn của LLM qua API ($p95 > 600 - 1800\text{ ms}$), gấp hàng chục lần SLA $50\text{ ms}$. Trong khi đó, trên tập dữ liệu nhỏ (SUB 40 tin), mô hình Encoder kết hợp MLP head đã chạm mốc lỗi $0\%$ ($F_1 = 1.000$).

### Dự đoán 2: Khả năng chống chịu của BoW trên 50 case khó

- **Dự đoán:** BoW + LogReg vẫn sẽ giữ được độ chính xác khoảng $40\% - 50\%$ và $F_1 \approx 0.50$ nhờ nhận diện được các từ khóa cơ bản.
- **Thực tế:** $F_1$ của BoW + LogReg giảm mạnh, từ $0.824$ xuống còn $0.256$ (giảm gần $0.57$ điểm so với test thường).
- **Vì sao lệch:** Đánh giá thấp độ tinh vi của các thủ thuật ngụy trang. Khi kẻ tấn công né từ khóa bằng leetspeak hoặc viết Spam theo giọng tự nhiên, BoW gặp khó khăn vì những từ bị biến đổi có thể trở thành Out-of-Vocabulary (OOV).

### Dự đoán 3: Sức mạnh của Deep Learning (MLP) so với Logistic Regression trên BoW

- **Dự đoán:** Mạng nơ-ron nhiều tầng (MLP) với $128,129$ tham số và hàm kích hoạt phi tuyến ReLU sẽ bỏ xa mô hình tuyến tính đơn giản như Logistic Regression.
- **Thực tế:** Kết quả TEST chênh lệch không đáng kể ($F_1$ quanh mức $0.91 - 0.92$), thậm chí MLP bắt đầu có dấu hiệu overfitting từ sau epoch 9.
- **Vì sao lệch:** Đầu vào Bag-of-Words đã làm mất cấu trúc chuỗi và ngữ cảnh. Logistic Regression có thể khai thác tốt các tín hiệu từ khóa, nên việc tăng số lượng tham số không nhất thiết cải thiện độ chính xác mà còn làm tăng nguy cơ overfitting.

---

## 4. Phân tích 50 case khó

### 4.1. Mô hình tụt nhiều nhất

Hai mô hình có mức suy giảm F1 lớn nhất:

| Mô hình | F1 test thường | F1 50 case khó | Mức giảm |
|---|---:|---:|---:|
| BoW + LogReg | 0.824 | 0.256 | 0.568 |
| BoW + MLP | 0.923 | 0.333 | **0.590** |

**BoW + MLP là mô hình tụt nhiều nhất**, tiếp theo là BoW + LogReg.

**Nhóm case gây khó khăn lớn nhất:**

- **Spam giọng tự nhiên (`natural_spam`):** Tỷ lệ đoán đúng khoảng $0\% - 10\%$.
- **Né từ khóa (`obfuscation`):** Tỷ lệ đoán đúng khoảng $0\% - 17\%$.

### 4.2. Hai case cụ thể và cơ chế khiến mô hình sai

**Case 1: Né từ khóa (Obfuscation)**

- **Văn bản:** `"C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4r..."`
- **Nhãn thực tế:** `spam`.
- **Cơ chế sai:** Kẻ gửi Spam thay thế chữ cái bằng số (`0` thay `o`, `4` thay `a`). BoW nhìn văn bản dưới dạng các token riêng lẻ nên những từ bị biến đổi có thể không tồn tại trong từ điển huấn luyện (OOV). Điều này làm mất các tín hiệu từ khóa quan trọng để nhận diện Spam.

**Case 2: Ham trông giống Spam (`ham_looks_spam`)**

- **Văn bản:** `"Congrats!!! You WON the bet lol, I owe you a b..."`
- **Nhãn thực tế:** `ham`.
- **Cơ chế sai:** Tin nhắn chứa nhiều từ có thể liên quan đến Spam như `congrats`, `won`, `bet`. Do BoW chủ yếu dựa vào các đặc trưng từ vựng mà không hiểu đầy đủ ngữ cảnh trêu đùa giữa bạn bè, mô hình có thể cộng dồn trọng số và phân loại nhầm thành Spam (False Positive).

---
## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

## 6. Phương án cho sản phẩm thật

**Yêu cầu:** Xây dựng hệ thống lọc 10 triệu SMS/ngày với độ trễ $p95 < 50\text{ ms/tin}$.

### 6.1. Phương án lựa chọn

**Pre-trained Encoder (MiniLM) + MLP head (hoặc Logistic Regression).**

### 6.2. Dẫn chứng số liệu

**Độ trễ:**

- Đạt $p95 \approx 40\text{ ms}$, nằm trong ngưỡng SLA $50\text{ ms}$.
- Tốc độ suy luận theo lô có thể đạt hàng nghìn tin/giây trên GPU tầm trung, tùy cấu hình triển khai.
- Tải trung bình cần đáp ứng:

$$
\frac{10,000,000}{86,400} \approx 116\text{ tin/giây}
$$

**Chất lượng:**

- Đạt $F_1 = 1.000$ trên tập kiểm tra SUB.
- Trên 50 case khó, $F_1 = 0.655$, cao hơn đáng kể so với các phương pháp BoW.

**Chi phí:**

- Có thể triển khai on-premise, không phát sinh chi phí gọi API cho mỗi tin nhắn.
- Có khả năng tiết kiệm chi phí API so với việc sử dụng LLM cho toàn bộ lưu lượng, nhưng vẫn cần tính chi phí phần cứng, điện năng và vận hành.

### 6.3. Đánh đổi chấp nhận

1. **Tài nguyên phần cứng:** MiniLM cần nhiều RAM/VRAM và tài nguyên tính toán hơn so với BoW thuần túy, vốn có thể chạy trên CPU cấu hình thấp.
2. **Khả năng xử lý ngữ cảnh phức tạp:** Chấp nhận việc mô hình có thể bỏ sót một số tình huống lừa đảo tinh vi mà LLM có khả năng xử lý tốt hơn.
3. **Chi phí triển khai:** Cần đầu tư hạ tầng và tối ưu inference, nhưng có thể giảm sự phụ thuộc vào API bên ngoài.

**Kết luận:** MiniLM + MLP head là phương án đáng cân nhắc nhờ cân bằng giữa chất lượng phân loại, độ trễ và khả năng triển khai nội bộ. Cần kiểm thử thêm trên lưu lượng thực tế trước khi xác nhận đáp ứng SLA.

---

## 7. AI log và điều đã học

### 7.1. Nhật ký sử dụng AI

| Phần | Bạn hỏi AI gì (Prompt) | AI trả lời / hỗ trợ gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| **Phần 2 — Deep Learning** | Nếu xóa `nn.ReLU()`, mạng hai lớp Linear liên tiếp tương đương với mô hình nào và F1 thay đổi ra sao? | Tương đương mô hình tuyến tính như Logistic Regression vì tích của hai phép biến đổi tuyến tính vẫn là một phép biến đổi tuyến tính. | Tự chứng minh toán học bằng cách gộp các ma trận trọng số. Chạy thử nghiệm bỏ ReLU và thấy F1 đạt 0.909 so với 0.921 ban đầu, phù hợp với lý thuyết. |
| **Phần 3 — Embedding** | Vì sao Sentence Embedding không được train trực tiếp để lọc Spam nhưng Spam và Ham vẫn có thể tách thành hai cụm trên bản đồ PCA? | Encoder đã học biểu diễn ngữ nghĩa tổng quát, giúp những câu có nội dung tương tự nằm gần nhau trong không gian vector. | Quan sát đồ thị PCA 2D thấy hai nhóm có sự phân hóa, sau đó đối chiếu ngữ nghĩa của các mẫu trong tập tricky để kiểm chứng. |
| **Phần 4 — LLM** | Vì sao khi phân loại bằng LLM phải đặt `temperature=0` và `max_tokens` nhỏ? Nếu `max_tokens=1` thì có rủi ro gì? | Temperature thấp giúp giảm tính ngẫu nhiên; giới hạn token nhỏ giúp tránh phản hồi dài. Nếu quá nhỏ, phản hồi có thể bị cắt trước khi hoàn thành nhãn. | Thử nghiệm trên API cho thấy khi giới hạn token quá thấp hoặc không cấu hình reasoning phù hợp, mô hình có thể hết ngân sách token và trả về nội dung rỗng. |

### 7.2. Điều đã học

Qua bài lab, em nhận ra rằng mô hình có kiến trúc phức tạp hơn chưa chắc đạt kết quả tốt hơn. Chất lượng phụ thuộc rất lớn vào cách biểu diễn dữ liệu, đặc điểm của bài toán và khả năng tổng quát hóa trên những trường hợp chưa từng xuất hiện trong tập huấn luyện.

Em cũng hiểu rằng F1 cao trên một tập test nhỏ không đủ để khẳng định mô hình phù hợp với sản phẩm thực tế. Khi triển khai hệ thống AI, cần đánh giá đồng thời chất lượng, độ trễ, chi phí vận hành và khả năng xử lý những trường hợp khó.