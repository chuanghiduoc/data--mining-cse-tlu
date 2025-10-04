# BÁO CÁO THUẬT TOÁN RANDOM FOREST

## 1. Ý TƯỞNG CHÍNH CỦA THUẬT TOÁN RANDOM FOREST

### 1.1 Khái niệm cơ bản
Random Forest (Rừng ngẫu nhiên) là một thuật toán học máy thuộc nhóm **ensemble learning**, kết hợp nhiều cây quyết định (Decision Trees) để tạo thành một mô hình mạnh mẽ hơn. Thuật toán này được phát triển bởi Leo Breiman vào năm 2001.

### 1.2 Nguyên lý hoạt động
Random Forest dựa trên hai nguyên lý chính:

1. **Bootstrap Aggregating (Bagging)**: Tạo nhiều mẫu con từ tập dữ liệu gốc bằng phương pháp lấy mẫu có hoàn lại
2. **Random Feature Selection**: Tại mỗi nút phân chia, chỉ xem xét một tập con ngẫu nhiên các đặc trưng

### 1.3 Ưu điểm
- Giảm hiện tượng overfitting
- Xử lý tốt dữ liệu thiếu
- Có thể đánh giá tầm quan trọng của các đặc trưng
- Hiệu suất cao với dữ liệu lớn
- Hoạt động tốt với cả dữ liệu số và phân loại

## 2. CÁC BƯỚC THỰC HIỆN THUẬT TOÁN

### Bước 1: Tạo Bootstrap Samples
```
Từ tập dữ liệu gốc D có n mẫu:
- Tạo k tập con D₁, D₂, ..., Dₖ
- Mỗi tập con có n mẫu được chọn ngẫu nhiên có hoàn lại
```

### Bước 2: Xây dựng Decision Trees
```
Với mỗi bootstrap sample Dᵢ:
- Xây dựng một cây quyết định Tᵢ
- Tại mỗi nút phân chia:
  * Chọn ngẫu nhiên m đặc trưng (m < tổng số đặc trưng)
  * Tìm đặc trưng tốt nhất trong m đặc trưng này để phân chia
```

### Bước 3: Kết hợp kết quả
```
Phân loại: Sử dụng majority voting
Hồi quy: Tính trung bình các dự đoán
```

## 3. VÍ DỤ TÍNH TOÁN MINH HỌA

### Tình huống thực tế: Dự đoán hiệu suất học tập sinh viên
Giả sử chúng ta có dữ liệu về các yếu tố ảnh hưởng đến việc sinh viên có đậu môn "Khai phá dữ liệu" hay không:

| ID | Giờ_học | Bài_tập | Điểm_danh | Kinh_nghiệm_lập_trình | Kết_quả |
|----|---------|---------|-----------|----------------------|---------|
| 1  | 20      | 8       | 85%       | Có                   | Đậu     |
| 2  | 15      | 6       | 90%       | Không                | Rớt     |
| 3  | 25      | 10      | 95%       | Có                   | Đậu     |
| 4  | 12      | 4       | 70%       | Không                | Rớt     |
| 5  | 30      | 9       | 88%       | Có                   | Đậu     |
| 6  | 18      | 7       | 92%       | Có                   | Đậu     |
| 7  | 10      | 3       | 65%       | Không                | Rớt     |
| 8  | 22      | 8       | 80%       | Không                | Rớt     |
| 9  | 28      | 10      | 96%       | Có                   | Đậu     |
| 10 | 16      | 5       | 75%       | Không                | Rớt     |

### Chi tiết thực hiện Random Forest

#### Bước 1: Tạo Bootstrap Samples (n_estimators = 3)

**Bootstrap Sample 1**: {1, 3, 5, 6, 9, 2, 8, 5, 1, 10}
```
ID | Giờ_học | Bài_tập | Điểm_danh | Kinh_nghiệm | Kết_quả
1  | 20      | 8       | 85%       | Có          | Đậu
3  | 25      | 10      | 95%       | Có          | Đậu  
5  | 30      | 9       | 88%       | Có          | Đậu
6  | 18      | 7       | 92%       | Có          | Đậu
9  | 28      | 10      | 96%       | Có          | Đậu
2  | 15      | 6       | 90%       | Không       | Rớt
8  | 22      | 8       | 80%       | Không       | Rớt
5  | 30      | 9       | 88%       | Có          | Đậu (lặp lại)
1  | 20      | 8       | 85%       | Có          | Đậu (lặp lại)
10 | 16      | 5       | 75%       | Không       | Rớt
```

**Bootstrap Sample 2**: {2, 4, 7, 8, 10, 1, 3, 6, 9, 4}
**Bootstrap Sample 3**: {1, 2, 5, 7, 9, 3, 6, 8, 10, 9}

#### Bước 2: Xây dựng Decision Trees với Random Feature Selection

**Tree 1** (từ Sample 1, max_features = 2):
```
Nút gốc: Random chọn {Giờ_học, Kinh_nghiệm_lập_trình}

Tính Information Gain:
- Kinh_nghiệm_lập_trình: IG = 0.693
- Giờ_học (≥20): IG = 0.421

→ Chọn Kinh_nghiệm_lập_trình làm root

Tree 1:
            Kinh_nghiệm_lập_trình
                  /        \
               Có             Không
              /                 \
           Đậu (6/7)          Rớt (3/3)
```

**Tree 2** (từ Sample 2, max_features = 2):
```
Nút gốc: Random chọn {Bài_tập, Điểm_danh}

Tính Information Gain:
- Bài_tập (≥7): IG = 0.544
- Điểm_danh (≥85%): IG = 0.468

→ Chọn Bài_tập làm root

Tree 2:
              Bài_tập ≥ 7
                /        \
             Có            Không  
            /                \
      Điểm_danh ≥90%        Rớt (4/4)
         /        \
      Đậu (3/4)   Rớt (1/4)
```

**Tree 3** (từ Sample 3, max_features = 2):
```
Nút gốc: Random chọn {Giờ_học, Điểm_danh}

Tree 3:
            Giờ_học ≥ 18
                /        \
             Có            Không
            /                \
         Đậu (5/6)         Rớt (3/4)
```

#### Bước 3: Dự đoán cho sinh viên mới

**Sinh viên X**: {Giờ_học: 24, Bài_tập: 8, Điểm_danh: 87%, Kinh_nghiệm: Có}

**Dự đoán từng cây:**
- **Tree 1**: Kinh_nghiệm = "Có" → **Đậu** (probability: 6/7 = 0.857)
- **Tree 2**: Bài_tập = 8 ≥ 7 → Điểm_danh = 87% < 90% → **Rớt** (probability: 1/4 = 0.25 để đậu)  
- **Tree 3**: Giờ_học = 24 ≥ 18 → **Đậu** (probability: 5/6 = 0.833)

#### Bước 4: Kết hợp kết quả (Ensemble)

**Majority Voting**:
- Đậu: 2 votes (Tree 1, Tree 3)
- Rớt: 1 vote (Tree 2)

**Kết quả cuối cùng**: **Đậu** với confidence = 2/3 = 67%

**Probability Averaging**:
P(Đậu) = (0.857 + 0.25 + 0.833) / 3 = 0.647 = **64.7%**

### Tính toán chi tiết Information Gain

#### Cho Tree 1 - Feature: Kinh_nghiệm_lập_trình

**Entropy ban đầu**:
- Total samples: 10
- Đậu: 6, Rớt: 4
- Entropy(S) = -6/10 × log₂(6/10) - 4/10 × log₂(4/10) = 0.971

**Sau khi split theo Kinh_nghiệm**:
- Kinh_nghiệm = "Có": 7 samples (6 Đậu, 1 Rớt)
  - Entropy = -6/7 × log₂(6/7) - 1/7 × log₂(1/7) = 0.592
- Kinh_nghiệm = "Không": 3 samples (0 Đậu, 3 Rớt)  
  - Entropy = 0 (pure)

**Information Gain**:
IG = 0.971 - (7/10 × 0.592 + 3/10 × 0) = 0.971 - 0.414 = **0.557**

### Ưu điểm thể hiện qua ví dụ

1. **Giảm Overfitting**: Tree 2 có thể overfit với rule phức tạp, nhưng được cân bằng bởi Tree 1 và 3
2. **Robust**: Nếu thiếu dữ liệu "Kinh_nghiệm", vẫn có Tree 2 và 3 để dự đoán
3. **Feature Importance**: Có thể thấy "Kinh_nghiệm_lập_trình" quan trọng nhất
4. **Uncertainty Quantification**: Probability 64.7% cho thấy độ tin cậy của dự đoán

## 4. CÔNG THỨC TOÁN HỌC

### 4.1 Information Gain
```
IG(S,A) = Entropy(S) - Σ(|Sᵥ|/|S|) × Entropy(Sᵥ)
```

### 4.2 Entropy
```
Entropy(S) = -Σ pᵢ × log₂(pᵢ)
```

### 4.3 Gini Impurity
```
Gini(S) = 1 - Σ pᵢ²
```

### 4.4 Out-of-Bag Error
```
OOB Error = (1/n) × Σ I(yᵢ ≠ ŷᵢ^(OOB))
```

## 5. THAM SỐ QUAN TRỌNG

1. **n_estimators**: Số lượng cây trong rừng
2. **max_features**: Số đặc trưng xem xét tại mỗi phân chia
3. **max_depth**: Độ sâu tối đa của cây
4. **min_samples_split**: Số mẫu tối thiểu để phân chia nút
5. **min_samples_leaf**: Số mẫu tối thiểu tại nút lá

## 6. ỨNG DỤNG THỰC TẾ

- **Y tế**: Chẩn đoán bệnh, phân tích gene
- **Tài chính**: Đánh giá rủi ro tín dụng, phát hiện gian lận
- **Marketing**: Phân khúc khách hàng, dự đoán hành vi mua hàng
- **Công nghệ**: Hệ thống gợi ý, xử lý ngôn ngữ tự nhiên

## 7. SO SÁNH VỚI CÁC THUẬT TOÁN KHÁC

| Thuật toán | Ưu điểm | Nhược điểm |
|------------|---------|------------|
| Random Forest | Chống overfitting, xử lý noise tốt | Khó giải thích, tốn bộ nhớ |
| Decision Tree | Dễ hiểu, nhanh | Dễ overfitting |
| Naive Bayes | Nhanh, ít dữ liệu | Giả định độc lập mạnh |
| SVM | Hiệu quả với dữ liệu ít | Chậm với dữ liệu lớn |

---

*Báo cáo này cung cấp cái nhìn tổng quan về thuật toán Random Forest, từ lý thuyết cơ bản đến ứng dụng thực tế.*
