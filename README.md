# 🔬 So sánh Random Forest vs Naive Bayes - Khai phá dữ liệu nâng cao

## 📋 Mô tả dự án

Thực nghiệm **nâng cao** so sánh hiệu quả thuật toán **Random Forest** và **Naive Bayes** trên **6 datasets khác nhau** với nhiều kỹ thuật tiền xử lý và khai thác dữ liệu.

### 🎯 Mục tiêu
- So sánh hiệu suất RF vs NB trên nhiều loại dữ liệu
- Áp dụng kỹ thuật tiền xử lý phù hợp cho từng dataset
- Xử lý imbalanced data, missing values, high-dimensional data
- Đánh giá toàn diện với nhiều metrics

---

## 📊 Datasets sử dụng

| Dataset | Samples | Features | Loại dữ liệu | Mô tả |
|---------|---------|----------|--------------|-------|
| **Wine Quality** | 1,599 | 12 | Numeric | Phân loại chất lượng rượu vang |
| **Diabetes** | 768 | 9 | Numeric | Dự đoán bệnh tiểu đường |
| **Adult Census** | 32,561 | 15 | Mixed | Dự đoán thu nhập >50K |
| **Mushroom** | 8,124 | 23 | Categorical | Phân loại nấm độc/ăn được |
| **Sonar** | 208 | 61 | Numeric | Phân biệt đá/mìn từ sonar |
| **Credit Card** | 284,807 | 31 | Numeric | Phát hiện gian lận thẻ tín dụng |

### 📥 Download Datasets

**Datasets KHÔNG được lưu trên GitHub** (do file quá lớn). Vui lòng tải về:

1. **Wine Quality**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)
2. **Diabetes**: [Kaggle - Pima Indians Diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
3. **Adult**: [Kaggle - Adult Census Income](https://www.kaggle.com/datasets/uciml/adult-census-income)
4. **Mushroom**: [Kaggle - Mushroom Classification](https://www.kaggle.com/datasets/uciml/mushroom-classification)
5. **Sonar**: [Kaggle - Sonar Mines vs Rocks](https://www.kaggle.com/datasets/mattcarter865/mines-vs-rocks)
6. **Credit Card**: [Kaggle - Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

**Sau khi tải về, đặt vào thư mục `datasets/` với tên:**
```
datasets/
├── winequality-red.csv
├── diabetes.csv
├── adult.csv
├── mushrooms.csv
├── sonar-all-data.csv
└── creditcard.csv
```

---

## 📁 Cấu trúc thư mục

```
btl/
├── datasets/              # Datasets (tải về riêng)
├── random_forest_experiment.py  # Code thực nghiệm chính
├── requirements.txt       # Dependencies
├── README.md             # File này
├── .gitignore            # Git ignore (datasets)
│
├── Random_Forest_Theory_Report.md   # Báo cáo lý thuyết
├── Random_Forest_Giang_Bai.md      # Giảng bài
│
└── Output files (generated):
    ├── eda_analysis_all.png
    ├── overall_comparison.png
    └── confusion_matrices_all.png
```

---

## 🚀 Cách chạy

### Bước 1: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

**Dependencies cần thiết:**
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- imbalanced-learn

### Bước 2: Download datasets

Tải 6 datasets từ links ở trên và đặt vào `datasets/`

### Bước 3: Chạy experiment

```bash
python random_forest_experiment.py
```

---

## 🔬 Quy trình thực nghiệm

### **Bước 1: Thu thập dữ liệu**
Load 6 datasets và kiểm tra cấu trúc

### **Bước 2: Khám phá dữ liệu (EDA)**
- Phân tích phân bố target
- Phát hiện missing values
- Phân loại feature types
- Visualizations

### **Bước 3: Tiền xử lý dữ liệu**

#### Kỹ thuật áp dụng theo dataset:

| Dataset | Techniques |
|---------|-----------|
| **Wine** | RobustScaler → Feature Engineering (interaction features) |
| **Diabetes** | Medical impossibility handling → Median imputation → StandardScaler → **SMOTE** |
| **Adult** | Missing value handling → Mixed encoding → StandardScaler |
| **Mushroom** | Label Encoding → **Mutual Information** feature selection (23→15 features) |
| **Sonar** | StandardScaler → **PCA** (60→30 features, 95% variance) |
| **CreditCard** | StandardScaler → **Random Undersampling** (284K→1.5K samples) |

### **Bước 3.5: Giảm chiều**
Áp dụng **PCA** cho Wine, Adult, CreditCard (so sánh trước/sau PCA)

### **Bước 4: Xây dựng mô hình**

#### Random Forest
```python
RandomForestClassifier(
    n_estimators=100,      # 100 trees
    max_depth=15,          # Tránh overfitting
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1              # Parallel processing
)
```

#### Naive Bayes
```python
GaussianNB()  # Cho numeric data
```

### **Bước 5: Đánh giá**

**Metrics sử dụng:**
- ✅ Accuracy: Tổng thể đúng bao nhiêu %
- ✅ Precision: False Positive cost
- ✅ Recall: False Negative cost  
- ✅ F1-Score: Harmonic mean (imbalanced data)
- ✅ AUC-ROC: Khả năng phân biệt classes

### **Bước 6: Visualization**
- Overall comparison (accuracy, F1, win rate)
- Confusion matrices
- Performance comparison

### **Bước 7: Báo cáo**
Tổng hợp kết quả và kết luận

---

## 📈 Kết quả chính

### 🏆 **Hiệu suất theo dataset**

| Dataset | RF Accuracy | NB Accuracy | Winner | Chênh lệch |
|---------|-------------|-------------|--------|------------|
| **Mushroom** | **100.0%** | 89.3% | RF | +10.7% |
| **CreditCard** | **96.0%** | 93.6% | RF | +2.4% |
| **CreditCard_PCA** | **96.3%** | 93.9% | RF | +2.4% |
| **Adult** | **86.2%** | 80.6% | RF | +5.6% |
| **Adult_PCA** | **83.8%** | 78.5% | RF | +5.3% |
| **Sonar** | **83.3%** | 61.9% | RF | +21.4% |
| **Diabetes** | **82.0%** | 68.0% | RF | +14.0% |
| **Wine_PCA** | **79.7%** | 72.8% | RF | +6.9% |
| **Wine** | **79.1%** | 72.5% | RF | +6.6% |

### 📊 **Tổng kết**

- **Random Forest**: 87.37% (average)
- **Naive Bayes**: 79.01% (average)
- **Win Rate**: RF thắng 9/9 datasets (100%)

---

## 💡 Kết luận

### ✅ **Khi nào dùng Random Forest?**
- ✨ Dữ liệu numeric với relationships phức tạp
- ✨ Mixed data types (numeric + categorical)
- ✨ Cần xử lý outliers và missing values
- ✨ Features có correlation cao
- ✨ Cần feature importance
- ✨ Dataset lớn, cần accuracy cao

### ✅ **Khi nào dùng Naive Bayes?**
- ✨ Text classification (TF-IDF)
- ✨ Categorical data với features độc lập
- ✨ Cần tốc độ training nhanh
- ✨ Dataset nhỏ
- ✨ Real-time prediction
- ✨ Interpretability quan trọng

---

## 🎓 Kỹ thuật đã áp dụng

### Preprocessing:
- [x] **Imputation**: Median/Mode strategy
- [x] **Scaling**: StandardScaler, RobustScaler
- [x] **Feature Engineering**: Interaction features
- [x] **Encoding**: Label Encoding cho categorical

### Data Handling:
- [x] **SMOTE**: Synthetic oversampling (Diabetes)
- [x] **Undersampling**: Random undersampling (CreditCard)
- [x] **PCA**: Dimensionality reduction (Sonar: 60→30)
- [x] **Feature Selection**: Mutual Information, SelectKBest

### Evaluation:
- [x] **Multiple metrics**: Accuracy, Precision, Recall, F1, AUC
- [x] **Confusion Matrix**: Phân tích chi tiết errors
- [x] **Cross-validation**: 5-fold CV
- [x] **Stratified split**: Giữ phân bố classes

---

## 🛠️ Yêu cầu hệ thống

- **Python**: 3.8+
- **RAM**: Tối thiểu 8GB (vì CreditCard dataset lớn)
- **Disk**: ~500MB cho datasets
- **OS**: Windows/Linux/MacOS

---

## 📚 Code Features

### ✨ **Highlights:**

1. **Comment đầy đủ bằng tiếng Việt**
   - Giải thích từng bước chi tiết
   - Có công thức toán học
   - Ví dụ cụ thể

2. **Preprocessing thông minh**
   - Medical impossibility handling (Diabetes)
   - Smart imputation strategies
   - Appropriate scaling cho từng dataset

3. **Visualization chất lượng cao**
   - Multiple resolutions (150 DPI, 300 DPI)
   - Professional charts
   - Easy to interpret

4. **Comprehensive evaluation**
   - 5 metrics per model
   - Confusion matrices
   - Win/loss analysis

---

## ❓ Troubleshooting

### Lỗi: "No such file or directory: 'datasets/...'"
→ Bạn chưa download datasets. Xem mục **📥 Download Datasets** ở trên

### Lỗi: "Module 'imblearn' not found"
```bash
pip install imbalanced-learn
```

### Warning về sklearn features
→ Có thể ignore, không ảnh hưởng kết quả

### Memory Error với CreditCard dataset
→ CreditCard dataset sẽ tự động undersampling. Nếu vẫn lỗi, có thể comment out dòng 87-91 trong code

---

## 📧 Tác giả

**Sinh viên**: [Tên của bạn]  
**Trường**: Đại học Thủy Lợi  
**Môn**: Khai phá dữ liệu  
**Năm**: 2025

---

## 📝 Ghi chú quan trọng

⚠️ **Datasets không được push lên GitHub** vì:
- File `creditcard.csv` là 143.84 MB (vượt giới hạn 100 MB của GitHub)
- Tuân thủ license của Kaggle/UCI

⚠️ **Khi clone repo:**
1. Clone project: `git clone [repo-url]`
2. Tạo thư mục: `mkdir datasets`
3. Download 6 datasets từ links trên
4. Chạy: `python random_forest_experiment.py`

---

## 🎯 Quick Start

```bash
# 1. Clone repository
git clone https://github.com/chuanghiduoc/data-mining-cse-tlu.git
cd btl

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download datasets (see links above)
# Đặt vào thư mục datasets/

# 4. Run experiment
python random_forest_experiment.py

# 5. Xem kết quả
# - eda_analysis_all.png
# - overall_comparison.png
# - confusion_matrices_all.png
```

---

## 📄 License

Dự án này dành cho mục đích học tập và nghiên cứu.

Datasets thuộc quyền sở hữu của:
- UCI Machine Learning Repository
- Kaggle Contributors

---

**⭐ Nếu thấy hữu ích, hãy star repo này!**
