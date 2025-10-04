# 🔬 THỰC NGHIỆM SO SÁNH RANDOM FOREST VÀ NAIVE BAYES (NÂNG CAO)

## 📋 Mô tả dự án

Thực nghiệm **nâng cao** so sánh hiệu quả của thuật toán **Random Forest** và **Naive Bayes** trên **7 datasets** khác nhau với **nhiều kỹ thuật tiền xử lý và khai thác dữ liệu**:

### 📊 Datasets được sử dụng:

1. **SMS Spam** - Text classification (5,572 samples)
2. **Wine Quality** - Numeric features (1,599 samples)
3. **Diabetes** - Medical data với missing values (768 samples)
4. **Adult Census** - Mixed data types (32,561 samples)
5. **Mushroom** - Categorical features (8,124 samples)
6. **Sonar** - High-dimensional numeric (208 samples, 60 features)
7. **Credit Card Fraud** - Large imbalanced dataset (284,807 samples)

## 📁 Cấu trúc thư mục

```
btl/
├── datasets/
│   ├── spam.csv              # SMS Spam dataset
│   ├── winequality-red.csv   # Wine Quality dataset
│   ├── diabetes.csv          # Pima Indians Diabetes
│   ├── adult.csv             # Adult Census Income
│   ├── mushrooms.csv         # Mushroom Classification
│   ├── sonar-all-data.csv    # Sonar Mines vs Rocks
│   └── creditcard.csv        # Credit Card Fraud Detection
├── random_forest_experiment.py     # Code thực nghiệm nâng cao
├── requirements.txt                # Thư viện cần thiết
├── TECHNIQUES_GUIDE.md             # Hướng dẫn chi tiết các kỹ thuật
├── setup_and_run.bat               # Script tự động cài đặt và chạy
├── Random_Forest_Theory_Report.md  # Báo cáo lý thuyết
└── README.md                       # Hướng dẫn này
```

## 🚀 Cách chạy thực nghiệm

### Phương pháp 1: Tự động (Khuyến nghị)
```bash
setup_and_run.bat
```
Script này sẽ tự động:
1. Cài đặt tất cả dependencies
2. Kiểm tra datasets
3. Chạy thực nghiệm

### Phương pháp 2: Thủ công

#### Bước 1: Cài đặt thư viện
```bash
pip install -r requirements.txt
```

#### Bước 2: Chạy thực nghiệm
```bash
python random_forest_experiment.py
```

## 📊 Kết quả sẽ được tạo ra

1. **eda_analysis_all.png** - Khám phá dữ liệu cho tất cả datasets
2. **overall_comparison.png** - So sánh tổng thể RF vs NB
3. **confusion_matrices_all.png** - Confusion matrices cho từng dataset

## 🔬 Quy trình khai phá dữ liệu (Data Mining Process)

### 1. Thu thập dữ liệu
- Tải 7 datasets từ thư mục `datasets/`
- Kiểm tra kích thước, kiểu dữ liệu

### 2. Khám phá dữ liệu (EDA)
- Phân tích phân bố target
- Phát hiện missing values
- Phân loại feature types (numeric/categorical)
- Visualizations

### 3. Tiền xử lý dữ liệu (NÂNG CAO)

#### Techniques được áp dụng:
- **Text Processing**: TF-IDF với n-grams, Feature Selection (Chi-squared)
- **Missing Values**: Smart imputation (median/mode), xử lý medical impossibility
- **Scaling**: StandardScaler, RobustScaler (robust to outliers)
- **Feature Engineering**: Interaction features, domain-specific features
- **Imbalanced Data**: SMOTE (oversampling), Random Undersampling
- **Dimensionality Reduction**: PCA (giảm từ 60→30 features cho Sonar)
- **Categorical Encoding**: Label Encoding, One-Hot Encoding
- **Feature Selection**: Chi2, Mutual Information, SelectKBest

#### Chi tiết cho từng dataset:

| Dataset | Techniques |
|---------|-----------|
| **SMS** | TF-IDF → Chi2 Feature Selection (3000→500) |
| **Wine** | RobustScaler → Feature Engineering (interaction) |
| **Diabetes** | Missing imputation → StandardScaler → SMOTE |
| **Adult** | Mixed encoding → Imputation → StandardScaler |
| **Mushroom** | Label Encoding → Mutual Info Selection |
| **Sonar** | StandardScaler → PCA (60→30 features) |
| **CreditCard** | StandardScaler → Random Undersampling |

### 4. Xây dựng mô hình
- **Random Forest**: 100 trees, max_depth=15, hyperparameter tuned
- **Naive Bayes**: 
  - MultinomialNB (text)
  - GaussianNB (numeric)
  - BernoulliNB (binary)

### 5. Đánh giá mô hình
- **Metrics**: Accuracy, Precision, Recall, F1-score, AUC-ROC
- **Visualizations**: Confusion Matrix, ROC Curves
- **Validation**: Cross-validation (5-fold)

### 6. Trực quan hóa kết quả
- So sánh performance trên từng dataset
- Average metrics across all datasets
- Win/Loss analysis
- Feature importance

### 7. Báo cáo kết quả
- Best performers cho từng dataset
- Kết luận và khuyến nghị
- Techniques summary

## 📈 Kết quả dự kiến

| Dataset | Features | Random Forest | Naive Bayes | Tốt hơn |
|---------|----------|---------------|-------------|---------|
| **SMS** | 500 (text) | ~95% | **~97%** | NB |
| **Wine** | 13 (numeric) | **~85%** | ~78% | RF |
| **Diabetes** | 8 (numeric) | **~78%** | ~75% | RF |
| **Adult** | 14 (mixed) | **~86%** | ~82% | RF |
| **Mushroom** | 15 (categorical) | **~100%** | ~97% | RF |
| **Sonar** | 30 (PCA) | **~82%** | ~75% | RF |
| **CreditCard** | 30 (numeric) | **~95%** | ~88% | RF |

## 💡 Kết luận chi tiết

### ✅ Random Forest tốt hơn khi:
- Dữ liệu numeric với relationships phức tạp
- Mixed data types
- Cần xử lý outliers và missing values
- Features có correlation cao
- Cần feature importance
- Dataset lớn

### ✅ Naive Bayes tốt hơn khi:
- Text classification (TF-IDF)
- Categorical data với features độc lập
- Cần tốc độ training nhanh
- Dataset nhỏ
- Real-time prediction
- Interpretability quan trọng

## 🛠️ Yêu cầu hệ thống

- **Python**: 3.8 trở lên
- **RAM**: Tối thiểu 8GB (vì Credit Card dataset lớn)
- **Disk**: ~500MB cho datasets và kết quả
- **OS**: Windows/Linux/MacOS

## 📦 Dependencies

```
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.2.0
matplotlib>=3.5.0
seaborn>=0.12.0
imbalanced-learn>=0.10.0
scipy>=1.9.0
```

## 🎓 Các kỹ thuật đã áp dụng

### Tiền xử lý (Preprocessing):
✅ TF-IDF Vectorization  
✅ Feature Selection (Chi2, Mutual Information)  
✅ Missing Value Imputation  
✅ Feature Scaling (Standard, Robust, MinMax)  
✅ Feature Engineering  
✅ Categorical Encoding  

### Xử lý dữ liệu (Data Handling):
✅ Imbalanced Data (SMOTE, Undersampling)  
✅ Dimensionality Reduction (PCA)  
✅ Cross-validation  
✅ Stratified splitting  

### Đánh giá (Evaluation):
✅ Multiple metrics (Accuracy, Precision, Recall, F1, AUC)  
✅ Confusion Matrix  
✅ ROC Curves  
✅ Feature Importance  

## 📚 Tài liệu tham khảo

- **`TECHNIQUES_GUIDE.md`**: Hướng dẫn chi tiết về từng kỹ thuật
- **`Random_Forest_Theory_Report.md`**: Lý thuyết về Random Forest
- Code có comments đầy đủ

## ❓ Troubleshooting

### Lỗi: "Module 'imblearn' not found"
```bash
pip install imbalanced-learn
```

### Lỗi: "Memory Error" với Credit Card dataset
→ Giảm `sampling_strategy` trong code hoặc skip dataset này

### Warning về sklearn features
→ Có thể ignore, không ảnh hưởng kết quả

## 📧 Liên hệ & Hỗ trợ

Nếu có vấn đề trong quá trình chạy thực nghiệm, vui lòng kiểm tra:

1. ✅ Đã cài đặt đúng thư viện chưa (`pip list`)
2. ✅ Datasets có trong thư mục `datasets/` chưa
3. ✅ Python version >= 3.8 (`python --version`)
4. ✅ Đủ RAM (8GB recommended)

### Quick Test:
```bash
python -c "import pandas, numpy, sklearn, imblearn; print('All dependencies OK!')"
```

## 🎯 Mục tiêu học tập

Thực nghiệm này giúp bạn:
- ✨ Hiểu rõ Random Forest vs Naive Bayes
- ✨ Áp dụng nhiều kỹ thuật tiền xử lý nâng cao
- ✨ Xử lý các loại dữ liệu khác nhau (text, numeric, mixed, categorical)
- ✨ Giải quyết vấn đề imbalanced data
- ✨ Áp dụng dimensionality reduction
- ✨ Đánh giá mô hình một cách toàn diện

---

**📌 Note**: Dataset Credit Card (~300MB) có thể mất vài phút để xử lý. Bạn có thể comment out trong code nếu muốn chạy nhanh hơn.

*Thực nghiệm này được thiết kế để minh họa toàn diện các kỹ thuật khai phá dữ liệu và so sánh hiệu suất Random Forest vs Naive Bayes trên nhiều loại dữ liệu khác nhau.*
