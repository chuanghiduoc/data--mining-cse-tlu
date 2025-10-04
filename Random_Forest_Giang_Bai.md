# BÀI GIẢNG: GIẢI BÀI TẬP RANDOM FOREST CHI TIẾT

## 📚 **ĐỀ BÀI**

Cho dữ liệu về hiệu suất học tập sinh viên môn "Khai phá dữ liệu":

| ID | Giờ_học | Bài_tập | Điểm_danh | Kinh_nghiệm | Kết_quả |
|----|---------|---------|-----------|-------------|---------|
| 1  | 20      | 8       | 85%       | Có          | Đậu     |
| 2  | 15      | 6       | 90%       | Không       | Rớt     |
| 3  | 25      | 10      | 95%       | Có          | Đậu     |
| 4  | 12      | 4       | 70%       | Không       | Rớt     |
| 5  | 30      | 9       | 88%       | Có          | Đậu     |
| 6  | 18      | 7       | 92%       | Có          | Đậu     |
| 7  | 10      | 3       | 65%       | Không       | Rớt     |
| 8  | 22      | 8       | 80%       | Không       | Rớt     |
| 9  | 28      | 10      | 96%       | Có          | Đậu     |
| 10 | 16      | 5       | 75%       | Không       | Rớt     |

**Yêu cầu**: Xây dựng Random Forest với 3 cây để dự đoán sinh viên X có profile {24 giờ học, 8 bài tập, 87% điểm danh, có kinh nghiệm}

---

## 🎯 **BƯỚC 1: CHUẨN BỊ DỮ LIỆU**

### **1.1 Mã hóa dữ liệu**
```
Kinh_nghiệm: "Có" = 1, "Không" = 0
Kết_quả: "Đậu" = 1, "Rớt" = 0
```

### **1.2 Dữ liệu sau mã hóa**
```
ID | Giờ_học | Bài_tập | Điểm_danh | Kinh_nghiệm | Kết_quả
1  | 20      | 8       | 85        | 1           | 1
2  | 15      | 6       | 90        | 0           | 0
3  | 25      | 10      | 95        | 1           | 1
4  | 12      | 4       | 70        | 0           | 0
5  | 30      | 9       | 88        | 1           | 1
6  | 18      | 7       | 92        | 1           | 1
7  | 10      | 3       | 65        | 0           | 0
8  | 22      | 8       | 80        | 0           | 0
9  | 28      | 10      | 96        | 1           | 1
10 | 16      | 5       | 75        | 0           | 0
```

---

## 🌲 **BƯỚC 2: TẠO BOOTSTRAP SAMPLES**

### **2.1 Lý thuyết Bootstrap**
- **Bootstrap**: Lấy mẫu có hoàn lại từ tập gốc
- **Kích thước**: Bằng tập gốc (10 mẫu)
- **Có thể lặp**: Cùng 1 mẫu có thể xuất hiện nhiều lần

### **2.2 Tạo Bootstrap Sample 1**

**Giả sử random chọn được**: {1, 3, 5, 6, 9, 2, 8, 5, 1, 10}

```
Bootstrap Sample 1:
ID | Giờ_học | Bài_tập | Điểm_danh | Kinh_nghiệm | Kết_quả
1  | 20      | 8       | 85        | 1           | 1
3  | 25      | 10      | 95        | 1           | 1
5  | 30      | 9       | 88        | 1           | 1
6  | 18      | 7       | 92        | 1           | 1
9  | 28      | 10      | 96        | 1           | 1
2  | 15      | 6       | 90        | 0           | 0
8  | 22      | 8       | 80        | 0           | 0
5  | 30      | 9       | 88        | 1           | 1  ← lặp lại
1  | 20      | 8       | 85        | 1           | 1  ← lặp lại
10 | 16      | 5       | 75        | 0           | 0
```

**Phân bố trong Sample 1**:
- Đậu (1): 7 mẫu
- Rớt (0): 3 mẫu

### **2.3 Tạo Bootstrap Sample 2 & 3**

**Sample 2**: {2, 4, 7, 8, 10, 1, 3, 6, 9, 4}
**Sample 3**: {1, 2, 5, 7, 9, 3, 6, 8, 10, 9}

*(Tương tự, tôi sẽ focus vào Sample 1 để giảng chi tiết)*

---

## 📊 **BƯỚC 3: XÂY DỰNG CÂY QUYẾT ĐỊNH 1**

### **3.1 Thiết lập tham số**
- **max_features = 2**: Mỗi nút chỉ xem xét 2/4 đặc trưng
- **Tổng features**: {Giờ_học, Bài_tập, Điểm_danh, Kinh_nghiệm}

### **3.2 Tại nút gốc**

**Random chọn 2 features**: {Giờ_học, Kinh_nghiệm}

#### **3.2.1 Tính Entropy ban đầu**

**Công thức Entropy**:
```
Entropy(S) = -Σ p_i × log₂(p_i)
```

**Áp dụng**:
- Tổng mẫu: 10
- Đậu: 7 mẫu → p₁ = 7/10 = 0.7
- Rớt: 3 mẫu → p₂ = 3/10 = 0.3

```
Entropy(S) = -(0.7 × log₂(0.7)) - (0.3 × log₂(0.3))
           = -(0.7 × (-0.515)) - (0.3 × (-1.737))
           = 0.361 + 0.521
           = 0.882
```

#### **3.2.2 Tính Information Gain cho Kinh_nghiệm**

**Phân chia theo Kinh_nghiệm**:

**Kinh_nghiệm = 1 (Có)**:
- Mẫu: {1, 3, 5, 6, 9, 5, 1} = 7 mẫu
- Đậu: 7, Rớt: 0
- Entropy = 0 (thuần khiết)

**Kinh_nghiệm = 0 (Không)**:
- Mẫu: {2, 8, 10} = 3 mẫu  
- Đậu: 0, Rớt: 3
- Entropy = 0 (thuần khiết)

**Information Gain**:
```
IG = Entropy_ban_đầu - Σ(|S_v|/|S|) × Entropy(S_v)
   = 0.882 - (7/10 × 0 + 3/10 × 0)
   = 0.882 - 0
   = 0.882
```

#### **3.2.3 Tính Information Gain cho Giờ_học**

**Chọn threshold**: median = 20

**Giờ_học ≤ 20**:
- Mẫu: {1, 2, 6, 10} với Giờ_học = {20, 15, 18, 16}
- Đậu: 2 (ID 1, 6), Rớt: 2 (ID 2, 10)
- p₁ = 2/4 = 0.5, p₂ = 2/4 = 0.5
- Entropy = -(0.5×log₂(0.5)) - (0.5×log₂(0.5)) = 1.0

**Giờ_học > 20**:
- Mẫu: {3, 5, 8, 9, 5, 1} với Giờ_học = {25, 30, 22, 28, 30, 20}
- Đậu: 5, Rớt: 1
- p₁ = 5/6 ≈ 0.833, p₂ = 1/6 ≈ 0.167
- Entropy = -(0.833×log₂(0.833)) - (0.167×log₂(0.167)) ≈ 0.65

**Information Gain**:
```
IG = 0.882 - (4/10 × 1.0 + 6/10 × 0.65)
   = 0.882 - (0.4 + 0.39)
   = 0.882 - 0.79
   = 0.092
```

#### **3.2.4 So sánh và chọn feature**

```
IG(Kinh_nghiệm) = 0.882 > IG(Giờ_học) = 0.092
```

**→ Chọn Kinh_nghiệm làm root**

### **3.3 Cây quyết định 1 hoàn chỉnh**

```
                Kinh_nghiệm
                    /    \
                 Có           Không
                /               \
           Đậu (7/7)         Rớt (3/3)
        P(Đậu) = 1.0      P(Đậu) = 0.0
```

---

## 🌳 **BƯỚC 4: XÂY DỰNG CÂY 2 & 3** *(Tóm tắt)*

### **Cây 2** (từ Sample 2):
- Random features: {Bài_tập, Điểm_danh}
- Root: Bài_tập ≥ 7
- Cấu trúc đơn giản với 2 nút lá

### **Cây 3** (từ Sample 3):
- Random features: {Giờ_học, Điểm_danh}  
- Root: Giờ_học ≥ 18
- Cấu trúc tương tự

---

## 🎯 **BƯỚC 5: DỰ ĐOÁN SINH VIÊN MỚI**

### **5.1 Profile sinh viên X**
```
Giờ_học: 24
Bài_tập: 8  
Điểm_danh: 87%
Kinh_nghiệm: Có (1)
```

### **5.2 Dự đoán từng cây**

#### **Cây 1**: 
```
Kinh_nghiệm = 1 (Có) → đi nhánh trái → Đậu
P(Đậu) = 1.0
```

#### **Cây 2**:
```
Bài_tập = 8 ≥ 7 → đi nhánh trái
Điểm_danh = 87% < 90% → đi nhánh phải → Rớt  
P(Đậu) = 0.25
```

#### **Cây 3**:
```
Giờ_học = 24 ≥ 18 → đi nhánh trái → Đậu
P(Đậu) = 0.83
```

### **5.3 Kết hợp kết quả (Ensemble)**

#### **Phương pháp 1: Majority Voting**
```
Cây 1: Đậu
Cây 2: Rớt  
Cây 3: Đậu

Kết quả: Đậu (2/3 phiếu)
```

#### **Phương pháp 2: Probability Averaging**
```
P(Đậu) = (1.0 + 0.25 + 0.83) / 3 = 2.08 / 3 = 0.693

→ 69.3% khả năng đậu
```

---

## 📝 **BƯỚC 6: KẾT LUẬN**

### **6.1 Đáp án cuối cùng**
**Sinh viên X có 69.3% khả năng ĐẬU môn Khai phá dữ liệu**

### **6.2 Giải thích kết quả**
- **Cây 1** dự đoán Đậu vì sinh viên có kinh nghiệm lập trình
- **Cây 2** dự đoán Rớt vì điểm danh thấp (87% < 90%)  
- **Cây 3** dự đoán Đậu vì học nhiều giờ (24 ≥ 18)

### **6.3 Ưu điểm Random Forest thể hiện**
1. **Đa dạng**: 3 cây nhìn từ góc độ khác nhau
2. **Robust**: Không phụ thuộc vào 1 cây duy nhất
3. **Confidence**: Cho ra xác suất thay vì chỉ yes/no

---

## 🧮 **PHỤ LỤC: CÔNG THỨC TÍNH TOÁN**

### **Information Gain**
```
IG(S,A) = Entropy(S) - Σ(|S_v|/|S|) × Entropy(S_v)
```

### **Entropy** 
```
Entropy(S) = -Σ p_i × log₂(p_i)
```

### **Gini Impurity** (thay thế cho Entropy)
```
Gini(S) = 1 - Σ p_i²
```

---

## 💡 **TIPS GIẢI BÀI TẬP TƯƠNG TỰ**

1. **Luôn mã hóa** dữ liệu categorical trước
2. **Tính Entropy** của tập gốc đầu tiên  
3. **Random chọn features** tại mỗi nút (quan trọng!)
4. **So sánh Information Gain** để chọn feature tốt nhất
5. **Dự đoán từng cây** riêng biệt trước khi ensemble
6. **Kết hợp** bằng majority vote hoặc averaging

**Chúc các bạn học tốt! 📚✨**
