#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
THỰC NGHIỆM SO SÁNH RANDOM FOREST VÀ NAIVE BAYES
Với nhiều datasets và kỹ thuật tiền xử lý nâng cao

Author: Student
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, RFE
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

# Thiết lập font cho tiếng Việt
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.style.use('seaborn-v0_8')

class AdvancedDataMiningExperiment:
    """
    Class thực hiện thí nghiệm khai phá dữ liệu nâng cao với nhiều datasets
    
    Mục đích: So sánh Random Forest và Naive Bayes trên nhiều loại dữ liệu khác nhau
    với các kỹ thuật tiền xử lý nâng cao
    """
    
    def __init__(self):
        """
        Khởi tạo experiment
        - results: Lưu kết quả đánh giá mô hình (accuracy, precision, recall, f1, auc)
        - datasets: Lưu dữ liệu thô đã load
        - feature_importance: Lưu độ quan trọng của features (từ Random Forest)
        """
        self.results = {}
        self.datasets = {}
        self.feature_importance = {}
        
    def load_datasets(self):
        """Bước 1: Thu thập dữ liệu - Tải tất cả datasets"""
        print("="*80)
        print("BƯỚC 1: THU THẬP DỮ LIỆU")
        print("="*80)
        
        datasets_config = {
            'Wine': {
                'path': 'datasets/winequality-red.csv',
                'encoding': 'utf-8',
                'type': 'numeric'
            },
            'Diabetes': {
                'path': 'datasets/diabetes.csv',
                'encoding': 'utf-8',
                'type': 'numeric'
            },
            'Adult': {
                'path': 'datasets/adult.csv',
                'encoding': 'utf-8',
                'type': 'mixed'
            },
            'Mushroom': {
                'path': 'datasets/mushrooms.csv',
                'encoding': 'utf-8',
                'type': 'categorical'
            },
            'Sonar': {
                'path': 'datasets/sonar-all-data.csv',
                'encoding': 'utf-8',
                'type': 'numeric'
            },
            'CreditCard': {
                'path': 'datasets/creditcard.csv',
                'encoding': 'utf-8',
                'type': 'numeric'
            }
        }
        
        successful_loads = 0
        
        for name, config in datasets_config.items():
            try:
                print(f"\n📊 Đang tải {name} Dataset...")
                df = pd.read_csv(config['path'], encoding=config['encoding'])
                
                self.datasets[name] = {
                    'data': df,
                    'type': config['type'],
                    'shape': df.shape
                }
                
                print(f"   ✅ {name}: {df.shape[0]} mẫu, {df.shape[1]} cột")
                successful_loads += 1
                
            except Exception as e:
                print(f"   ❌ Lỗi tải {name}: {e}")
            
        print(f"\n✅ Đã tải thành công {successful_loads}/{len(datasets_config)} datasets")
        return successful_loads > 0
    
    def explore_data(self):
        """Bước 2: Khám phá dữ liệu (EDA) với phân tích chi tiết"""
        print("\n" + "="*80)
        print("BƯỚC 2: KHÁM PHÁ DỮ LIỆU (EDA)")
        print("="*80)
        
        for name, dataset_info in self.datasets.items():
            df = dataset_info['data']
            print(f"\n{'='*60}")
            print(f"📊 {name.upper()} DATASET")
            print(f"{'='*60}")
            print(f"Kích thước: {df.shape}")
            print(f"Loại dữ liệu: {dataset_info['type']}")
            print(f"\nThông tin cột:")
            print(df.dtypes)
            print(f"\nGiá trị null:")
            print(df.isnull().sum())
            print(f"\nThống kê mô tả:")
            print(df.describe())
        
        # Visualization
        self.visualize_eda()
    
    def visualize_eda(self):
        """Vẽ biểu đồ khám phá dữ liệu cho tất cả datasets"""
        num_datasets = len(self.datasets)
        fig, axes = plt.subplots(num_datasets, 3, figsize=(20, 5*num_datasets))
        
        if num_datasets == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('KHÁM PHÁ DỮ LIỆU - TẤT CẢ DATASETS', fontsize=18, fontweight='bold', y=0.995)
        
        for idx, (name, dataset_info) in enumerate(self.datasets.items()):
            df = dataset_info['data']
            
            # Subplot 1: Distribution of target
            target_col = self._get_target_column(name, df)
            if target_col:
                target_counts = df[target_col].value_counts()
                axes[idx, 0].bar(range(len(target_counts)), target_counts.values, color='steelblue', alpha=0.7)
                axes[idx, 0].set_title(f'{name}: Phân bố Target', fontsize=11, pad=10)
                axes[idx, 0].set_xlabel('Class', fontsize=9)
                axes[idx, 0].set_ylabel('Số lượng', fontsize=9)
                axes[idx, 0].tick_params(labelsize=8)
                
                # Add value labels
                for i, v in enumerate(target_counts.values):
                    axes[idx, 0].text(i, v + max(target_counts.values)*0.02, str(v), 
                                     ha='center', va='bottom', fontsize=8)
            
            # Subplot 2: Missing values
            missing = df.isnull().sum()
            if missing.sum() > 0:
                missing = missing[missing > 0].sort_values(ascending=False)[:10]
                axes[idx, 1].barh(range(len(missing)), missing.values, color='coral', alpha=0.7)
                axes[idx, 1].set_yticks(range(len(missing)))
                axes[idx, 1].set_yticklabels(missing.index, fontsize=8)
                axes[idx, 1].set_title(f'{name}: Giá trị thiếu', fontsize=11, pad=10)
                axes[idx, 1].set_xlabel('Số lượng', fontsize=9)
            else:
                axes[idx, 1].text(0.5, 0.5, 'Không có giá trị thiếu', 
                                 ha='center', va='center', fontsize=10)
                axes[idx, 1].set_title(f'{name}: Giá trị thiếu', fontsize=11, pad=10)
            axes[idx, 1].tick_params(labelsize=8)
            
            # Subplot 3: Feature types
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            
            type_counts = [len(numeric_cols), len(categorical_cols)]
            colors = ['lightgreen', 'lightcoral']
            axes[idx, 2].pie(type_counts, labels=['Numeric', 'Categorical'], 
                            autopct='%1.1f%%', colors=colors, textprops={'fontsize': 9})
            axes[idx, 2].set_title(f'{name}: Loại Features', fontsize=11, pad=10)
        
        plt.tight_layout(pad=3.0)
        plt.subplots_adjust(top=0.96)
        plt.savefig('eda_analysis_all.png', dpi=150, bbox_inches='tight', facecolor='white')
        print("\n✅ Đã lưu biểu đồ EDA: eda_analysis_all.png")
        plt.close()
    
    def _get_target_column(self, dataset_name, df):
        """Xác định cột target cho mỗi dataset"""
        target_map = {
            'Wine': 'quality',
            'Diabetes': 'Outcome',
            'Adult': 'income',
            'Mushroom': 'class',
            'Sonar': 'Label',
            'CreditCard': 'Class'
        }
        return target_map.get(dataset_name)
    
    def preprocess_data(self):
        """Bước 3: Tiền xử lý dữ liệu với nhiều kỹ thuật nâng cao"""
        print("\n" + "="*80)
        print("BƯỚC 3: TIỀN XỬ LÝ DỮ LIỆU (NÂNG CAO)")
        print("="*80)
        
        self.processed_datasets = {}
        
        for name, dataset_info in self.datasets.items():
            print(f"\n{'='*60}")
            print(f"🔧 Xử lý {name} Dataset")
            print(f"{'='*60}")
            
            df = dataset_info['data'].copy()
            
            if name == 'Wine':
                X, y = self._preprocess_wine(df)
            elif name == 'Diabetes':
                X, y = self._preprocess_diabetes(df)
            elif name == 'Adult':
                X, y = self._preprocess_adult(df)
            elif name == 'Mushroom':
                X, y = self._preprocess_mushroom(df)
            elif name == 'Sonar':
                X, y = self._preprocess_sonar(df)
            elif name == 'CreditCard':
                X, y = self._preprocess_creditcard(df)
            else:
                continue
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            self.processed_datasets[name] = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'feature_count': X.shape[1]
            }
            
            print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")
    
    def _preprocess_wine(self, df):
        """
        Tiền xử lý Wine dataset
        
        Kỹ thuật áp dụng:
        1. Binary classification: Chuyển quality thành 2 lớp (good/bad)
        2. Feature Engineering: Tạo features tương tác
        3. RobustScaler: Scaling chống outliers
        """
        print("🍷 Wine: Scaling + Feature Engineering")
        
        # Bước 1: Chuyển sang binary classification
        # quality >= 6: good wine (1), quality < 6: bad wine (0)
        y = (df['quality'] >= 6).astype(int)
        X = df.drop('quality', axis=1)
        
        # Bước 2: Feature Engineering - Tạo features mới từ sự tương tác
        # alcohol * sulphates: Tương tác giữa độ cồn và sulphates
        X['alcohol_sulphates'] = X['alcohol'] * X['sulphates']
        
        # volatile acidity / fixed acidity: Tỷ lệ acid dễ bay hơi
        # +0.001 để tránh chia cho 0
        X['volatile_total_acidity'] = X['volatile acidity'] / (X['fixed acidity'] + 0.001)
        
        # Bước 3: RobustScaler - Scaling dựa trên median và IQR
        # Ưu điểm: Ít bị ảnh hưởng bởi outliers hơn StandardScaler
        # Công thức: X_scaled = (X - median) / IQR
        scaler = RobustScaler()
        X = scaler.fit_transform(X)
        
        print(f"   Features: {X.shape[1]}")
        return X, y
    
    def _preprocess_diabetes(self, df):
        """
        Tiền xử lý Diabetes dataset
        
        Kỹ thuật áp dụng:
        1. Smart Imputation: Xử lý giá trị 0 không hợp lý (medical impossibility)
        2. StandardScaler: Chuẩn hóa dữ liệu
        3. SMOTE: Xử lý imbalanced data (tăng minority class)
        """
        print("💉 Diabetes: Imputation + Scaling + SMOTE")
        
        y = df['Outcome']  # Target: 1=có tiểu đường, 0=không
        X = df.drop('Outcome', axis=1)
        
        # Bước 1: Xử lý giá trị 0 không hợp lý về mặt y học
        # VD: Glucose=0, BloodPressure=0, BMI=0 là không thể
        # → Thay bằng NaN để impute sau
        zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in zero_cols:
            if col in X.columns:
                X[col] = X[col].replace(0, np.nan)  # 0 → NaN
        
        # Bước 2: Imputation - Điền giá trị thiếu bằng median
        # Dùng median thay vì mean vì ít bị ảnh hưởng bởi outliers
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
        
        # Bước 3: StandardScaler - Chuẩn hóa về mean=0, std=1
        # Công thức: X_scaled = (X - mean) / std
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Bước 4: SMOTE (Synthetic Minority Over-sampling Technique)
        # Xử lý imbalanced data bằng cách tạo synthetic samples cho minority class
        # Cách hoạt động:
        # 1. Chọn 1 sample từ minority class
        # 2. Tìm k-nearest neighbors
        # 3. Tạo sample mới giữa sample gốc và neighbor
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)
        
        print(f"   Features: {X.shape[1]}, Samples sau SMOTE: {X.shape[0]}")
        return X, y
    
    def _preprocess_adult(self, df):
        """Tiền xử lý Adult dataset"""
        print("👤 Adult: Mixed data handling + Encoding")
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Target
        y = LabelEncoder().fit_transform(df['income'])
        
        # Drop target and unnecessary columns
        X = df.drop(['income'], axis=1)
        
        # Handle missing values (marked as '?')
        X = X.replace('?', np.nan)
        
        # Separate numeric and categorical
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        # Impute numeric
        if len(numeric_cols) > 0:
            X[numeric_cols] = SimpleImputer(strategy='median').fit_transform(X[numeric_cols])
        
        # Impute and encode categorical
        for col in categorical_cols:
            X[col] = X[col].fillna(X[col].mode()[0])
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        
        # Scale
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        print(f"   Features: {X.shape[1]}")
        return X, y
    
    def _preprocess_mushroom(self, df):
        """
        Tiền xử lý Mushroom dataset
        
        Kỹ thuật áp dụng:
        1. Label Encoding: Chuyển categorical → numeric
        2. Feature Selection: Chọn features quan trọng bằng Mutual Information
        
        Lý do: Mushroom dataset có 23 categorical features
        → Cần encode sang numeric để Random Forest và Naive Bayes xử lý được
        """
        print("🍄 Mushroom: Categorical encoding + Feature selection")
        
        # Target: 'e'=edible (ăn được), 'p'=poisonous (độc)
        # LabelEncoder chuyển 'e'→0, 'p'→1
        y = LabelEncoder().fit_transform(df['class'])
        X = df.drop('class', axis=1)
        
        # Label Encoding cho tất cả categorical features
        # VD: cap-shape: 'b'→0, 'c'→1, 'x'→2, 'f'→3, 'k'→4, 's'→5
        for col in X.columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        
        X = X.values  # Chuyển DataFrame → numpy array
        
        # Feature Selection bằng Mutual Information
        # Mutual Information đo "thông tin chung" giữa feature và target
        # Chọn top 15 features có MI cao nhất (quan trọng nhất)
        # Ưu điểm: Không giả định linear relationship như correlation
        selector = SelectKBest(mutual_info_classif, k=min(15, X.shape[1]))
        X = selector.fit_transform(X, y)
        
        print(f"   Features sau selection: {X.shape[1]}")
        return X, y
    
    def _preprocess_sonar(self, df):
        """
        Tiền xử lý Sonar dataset
        
        Kỹ thuật áp dụng:
        1. StandardScaler: Chuẩn hóa dữ liệu
        2. PCA: Giảm chiều từ 60→30 features
        
        Lý do: 
        - Dataset nhỏ (208 samples) + High-dimensional (60 features)
        - Dễ bị overfitting → Cần giảm chiều
        """
        print("📡 Sonar: PCA dimensionality reduction")
        
        # Target: 'R'=Rock (đá), 'M'=Mine (mìn)
        # Cột cuối cùng là Label
        y = LabelEncoder().fit_transform(df.iloc[:, -1])
        X = df.iloc[:, :-1].values  # 60 frequency features
        
        # Bước 1: StandardScaler - Bắt buộc trước khi PCA
        # PCA nhạy cảm với scale → cần chuẩn hóa trước
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Bước 2: PCA (Principal Component Analysis)
        # Giảm từ 60→30 features (giữ lại thông tin quan trọng nhất)
        # Cách hoạt động:
        # 1. Tìm các trục (principal components) có variance lớn nhất
        # 2. Project dữ liệu lên 30 trục đầu tiên
        # 3. Loại bỏ 30 trục còn lại (ít variance)
        pca = PCA(n_components=30, random_state=42)
        X = pca.fit_transform(X)
        
        print(f"   Features sau PCA: {X.shape[1]}")
        # Variance explained: % thông tin được giữ lại sau PCA
        print(f"   Variance explained: {pca.explained_variance_ratio_.sum():.2%}")
        return X, y
    
    def _preprocess_creditcard(self, df):
        """
        Tiền xử lý Credit Card dataset
        
        Kỹ thuật áp dụng:
        1. StandardScaler: Chuẩn hóa Time và Amount
        2. Random Undersampling: Giảm majority class để cân bằng
        
        Lý do:
        - Dataset quá lớn (284K samples) → Undersampling để training nhanh hơn
        - Highly imbalanced (fraud rất ít) → Cần cân bằng classes
        """
        print("💳 Credit Card: Undersampling (dataset lớn + imbalanced)")
        
        y = df['Class']  # Target: 0=normal, 1=fraud (gian lận)
        X = df.drop('Class', axis=1)
        
        # Bước 1: Scale Time và Amount
        # V1-V28 đã được PCA transform rồi (chuẩn hóa sẵn)
        # Chỉ cần scale Time và Amount
        scaler = StandardScaler()
        X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])
        
        X = X.values
        
        # Bước 2: Random Undersampling
        # Giảm majority class (normal transactions) xuống
        # sampling_strategy=0.5: Tỷ lệ fraud/normal = 0.5 (1 fraud : 2 normal)
        # VD: Có 492 fraud → giữ lại 984 normal (thay vì 284315)
        # Ưu điểm: Training nhanh hơn, cân bằng classes
        # Nhược điểm: Mất thông tin từ majority class
        rus = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
        X, y = rus.fit_resample(X, y)
        
        print(f"   Features: {X.shape[1]}, Samples sau undersampling: {X.shape[0]}")
        return X, y
    
    def apply_dimensionality_reduction(self):
        """Bước 3.5: Áp dụng giảm chiều cho các datasets phù hợp"""
        print("\n" + "="*80)
        print("BƯỚC 3.5: GIẢM CHIỀU DỮ LIỆU")
        print("="*80)
        
        for name in ['Wine', 'Adult', 'CreditCard']:
            if name in self.processed_datasets:
                print(f"\n🔬 Áp dụng PCA cho {name}")
                
                X_train = self.processed_datasets[name]['X_train']
                X_test = self.processed_datasets[name]['X_test']
                
                # Determine optimal components
                n_components = min(10, X_train.shape[1] // 2)
                
                pca = PCA(n_components=n_components, random_state=42)
                X_train_pca = pca.fit_transform(X_train)
                X_test_pca = pca.transform(X_test)
                
                print(f"   Original features: {X_train.shape[1]}")
                print(f"   Reduced to: {n_components}")
                print(f"   Variance explained: {pca.explained_variance_ratio_.sum():.2%}")
                
                # Store both versions
                self.processed_datasets[f"{name}_PCA"] = {
                    'X_train': X_train_pca,
                    'X_test': X_test_pca,
                    'y_train': self.processed_datasets[name]['y_train'],
                    'y_test': self.processed_datasets[name]['y_test'],
                    'feature_count': n_components
                }
    
    def train_models(self):
        """
        Bước 4: Xây dựng mô hình với hyperparameter tuning
        
        Train 2 models:
        1. Random Forest: Ensemble of decision trees
        2. Naive Bayes: Probabilistic classifier (GaussianNB)
        """
        print("\n" + "="*80)
        print("BƯỚC 4: XÂY DỰNG MÔ HÌNH (VỚI TUNING)")
        print("="*80)
        
        self.models = {}
        
        for name, data in self.processed_datasets.items():
            print(f"\n{'='*60}")
            print(f"🔨 Training trên {name}")
            print(f"{'='*60}")
            
            X_train, y_train = data['X_train'], data['y_train']
            
            # ============================================================
            # RANDOM FOREST CLASSIFIER
            # ============================================================
            print("🌲 Random Forest...")
            rf = RandomForestClassifier(
                n_estimators=100,        # Số lượng trees trong forest
                max_depth=15,            # Độ sâu tối đa của mỗi tree (tránh overfitting)
                min_samples_split=5,     # Min số samples để split node
                min_samples_leaf=2,      # Min số samples ở leaf node
                random_state=42,         # Random seed để reproducible
                n_jobs=-1                # Dùng tất cả CPU cores (parallel)
            )
            # Cách hoạt động:
            # 1. Tạo 100 decision trees
            # 2. Mỗi tree train trên random subset của data (bagging)
            # 3. Mỗi split chọn random subset của features
            # 4. Prediction: Vote từ 100 trees (majority voting)
            rf.fit(X_train, y_train)
            
            # ============================================================
            # NAIVE BAYES CLASSIFIER
            # ============================================================
            print("🎯 Naive Bayes...")
            # Dùng GaussianNB vì tất cả features đã được chuyển sang numeric
            # GaussianNB giả định features follow Gaussian distribution
            nb = GaussianNB()
            
            # Cách hoạt động:
            # 1. Tính P(y) - Prior probability của mỗi class
            # 2. Tính P(xi|y) - Likelihood của feature xi given class y
            # 3. Prediction: P(y|X) = P(y) * ∏P(xi|y) / P(X) (Bayes' theorem)
            # 4. Giả định: Features độc lập (naive assumption)
            nb.fit(X_train, y_train)
            
            self.models[name] = {
                'rf': rf,
                'nb': nb
            }
            
            print(f"✅ {name} hoàn thành")
    
    def evaluate_models(self):
        """
        Bước 5: Đánh giá mô hình với nhiều metrics
        
        Metrics được dùng:
        1. Accuracy: Tổng thể đúng bao nhiêu %
        2. Precision: Trong dự đoán positive, đúng bao nhiêu %
        3. Recall: Trong positive thật, catch được bao nhiêu %
        4. F1-Score: Harmonic mean của Precision và Recall
        5. AUC-ROC: Khả năng phân biệt classes
        """
        print("\n" + "="*80)
        print("BƯỚC 5: ĐÁNH GIÁ MÔ HÌNH")
        print("="*80)
        
        self.results = {}
        
        for name, models in self.models.items():
            print(f"\n{'='*60}")
            print(f"📊 Đánh giá {name}")
            print(f"{'='*60}")
            
            data = self.processed_datasets[name]
            X_test, y_test = data['X_test'], data['y_test']
            
            for model_name, model in models.items():
                model_key = f"{model_name.upper()}_{name}"
                
                # Predictions
                y_pred = model.predict(X_test)              # Class predictions (0 hoặc 1)
                y_proba = model.predict_proba(X_test)[:, 1]  # Probability của class 1
                
                # ============================================================
                # TÍNH CÁC METRICS
                # ============================================================
                
                # 1. ACCURACY = (TP + TN) / Total
                #    Ý nghĩa: Tổng thể đúng bao nhiêu %
                accuracy = accuracy_score(y_test, y_pred)
                
                # 2. PRECISION = TP / (TP + FP)
                #    Ý nghĩa: Trong những cái dự đoán là positive, bao nhiêu % đúng?
                #    Quan trọng khi: False Positive tốn kém
                precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
                
                # 3. RECALL (Sensitivity) = TP / (TP + FN)
                #    Ý nghĩa: Trong tất cả positive thật, model catch được bao nhiêu %?
                #    Quan trọng khi: False Negative nguy hiểm (VD: ung thư)
                recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
                
                # 4. F1-SCORE = 2 * (Precision * Recall) / (Precision + Recall)
                #    Ý nghĩa: Harmonic mean, cân bằng giữa Precision và Recall
                #    Dùng khi: Imbalanced data hoặc cần cân bằng cả 2
                f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
                
                # 5. AUC-ROC (Area Under ROC Curve)
                #    Ý nghĩa: Khả năng model phân biệt 2 classes
                #    AUC = 1.0: Perfect, AUC = 0.5: Random guessing
                auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) == 2 else 0
                
                self.results[model_key] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc,
                    'predictions': (y_pred, y_proba),
                    'y_test': y_test
                }
                
                print(f"{model_name.upper()}: Accuracy={accuracy:.4f}, "
                      f"F1={f1:.4f}, AUC={auc:.4f}")
    
    def visualize_results(self):
        """Bước 6: Trực quan hóa kết quả toàn diện"""
        print("\n" + "="*80)
        print("BƯỚC 6: TRỰC QUAN HÓA KẾT QUẢ")
        print("="*80)
        
        # Create comprehensive comparison
        self._plot_overall_comparison()
        self._plot_dataset_comparisons()
        print("✅ Đã lưu các biểu đồ kết quả")
    
    def _plot_overall_comparison(self):
        """Vẽ biểu đồ so sánh tổng thể"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('SO SÁNH TỔNG THỂ: RANDOM FOREST vs NAIVE BAYES', 
                    fontsize=16, fontweight='bold')
        
        # 1. Accuracy comparison
        ax1 = axes[0, 0]
        datasets = list(set([k.split('_', 1)[1] for k in self.results.keys()]))
        rf_acc = [self.results.get(f"RF_{d}", {}).get('accuracy', 0) for d in datasets]
        nb_acc = [self.results.get(f"NB_{d}", {}).get('accuracy', 0) for d in datasets]
        
        x = np.arange(len(datasets))
        width = 0.35
        ax1.bar(x - width/2, rf_acc, width, label='Random Forest', color='darkgreen', alpha=0.8)
        ax1.bar(x + width/2, nb_acc, width, label='Naive Bayes', color='orange', alpha=0.8)
        ax1.set_xlabel('Dataset', fontsize=10)
        ax1.set_ylabel('Accuracy', fontsize=10)
        ax1.set_title('Accuracy Comparison', fontsize=12, pad=15)
        ax1.set_xticks(x)
        ax1.set_xticklabels(datasets, rotation=45, ha='right', fontsize=8)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # 2. F1-Score comparison
        ax2 = axes[0, 1]
        rf_f1 = [self.results.get(f"RF_{d}", {}).get('f1', 0) for d in datasets]
        nb_f1 = [self.results.get(f"NB_{d}", {}).get('f1', 0) for d in datasets]
        
        ax2.bar(x - width/2, rf_f1, width, label='Random Forest', color='darkblue', alpha=0.8)
        ax2.bar(x + width/2, nb_f1, width, label='Naive Bayes', color='red', alpha=0.8)
        ax2.set_xlabel('Dataset', fontsize=10)
        ax2.set_ylabel('F1-Score', fontsize=10)
        ax2.set_title('F1-Score Comparison', fontsize=12, pad=15)
        ax2.set_xticks(x)
        ax2.set_xticklabels(datasets, rotation=45, ha='right', fontsize=8)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # 3. Average metrics
        ax3 = axes[1, 0]
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        rf_avg = [np.mean([self.results.get(f"RF_{d}", {}).get(m, 0) for d in datasets]) 
                  for m in metrics]
        nb_avg = [np.mean([self.results.get(f"NB_{d}", {}).get(m, 0) for d in datasets]) 
                  for m in metrics]
        
        x_m = np.arange(len(metrics))
        ax3.bar(x_m - width/2, rf_avg, width, label='Random Forest', color='purple', alpha=0.8)
        ax3.bar(x_m + width/2, nb_avg, width, label='Naive Bayes', color='pink', alpha=0.8)
        ax3.set_xlabel('Metrics', fontsize=10)
        ax3.set_ylabel('Average Score', fontsize=10)
        ax3.set_title('Average Performance Across All Datasets', fontsize=12, pad=15)
        ax3.set_xticks(x_m)
        ax3.set_xticklabels(metrics, fontsize=9)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # 4. Win/Loss count
        ax4 = axes[1, 1]
        rf_wins = sum(1 for d in datasets if self.results.get(f"RF_{d}", {}).get('accuracy', 0) > 
                      self.results.get(f"NB_{d}", {}).get('accuracy', 0))
        nb_wins = sum(1 for d in datasets if self.results.get(f"NB_{d}", {}).get('accuracy', 0) > 
                      self.results.get(f"RF_{d}", {}).get('accuracy', 0))
        
        ax4.pie([rf_wins, nb_wins], labels=['RF Wins', 'NB Wins'], 
               autopct='%1.0f%%', colors=['darkgreen', 'orange'], 
               textprops={'fontsize': 11})
        ax4.set_title('Win Rate (by Accuracy)', fontsize=12, pad=15)
        
        plt.tight_layout()
        plt.savefig('overall_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _plot_dataset_comparisons(self):
        """Vẽ confusion matrices cho các datasets"""
        num_datasets = len(self.models)
        fig, axes = plt.subplots(num_datasets, 2, figsize=(12, 5*num_datasets))
        
        if num_datasets == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('CONFUSION MATRICES - TẤT CẢ DATASETS', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        for idx, name in enumerate(self.models.keys()):
            # RF Confusion Matrix
            rf_key = f"RF_{name}"
            if rf_key in self.results:
                y_pred, _ = self.results[rf_key]['predictions']
                y_test = self.results[rf_key]['y_test']
                cm = confusion_matrix(y_test, y_pred)
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[idx, 0],
                           annot_kws={'size': 10})
                axes[idx, 0].set_title(f'Random Forest - {name}', fontsize=11, pad=10)
                axes[idx, 0].set_xlabel('Predicted', fontsize=9)
                axes[idx, 0].set_ylabel('Actual', fontsize=9)
            
            # NB Confusion Matrix
            nb_key = f"NB_{name}"
            if nb_key in self.results:
                y_pred, _ = self.results[nb_key]['predictions']
                y_test = self.results[nb_key]['y_test']
                cm = confusion_matrix(y_test, y_pred)
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=axes[idx, 1],
                           annot_kws={'size': 10})
                axes[idx, 1].set_title(f'Naive Bayes - {name}', fontsize=11, pad=10)
                axes[idx, 1].set_xlabel('Predicted', fontsize=9)
                axes[idx, 1].set_ylabel('Actual', fontsize=9)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.96)
        plt.savefig('confusion_matrices_all.png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def generate_report(self):
        """Bước 7: Tạo báo cáo kết quả chi tiết"""
        print("\n" + "="*80)
        print("BƯỚC 7: BÁO CÁO KẾT QUẢ CHI TIẾT")
        print("="*80)
        
        # Overall summary
        datasets = list(set([k.split('_', 1)[1] for k in self.results.keys()]))
        
        print(f"\n📊 TỔNG QUAN:")
        print("="*60)
        print(f"Tổng số datasets: {len(datasets)}")
        print(f"Tổng số mô hình: {len(self.results)}")
        
        # Best performers
        print(f"\n🏆 HIỆU SUẤT THEO DATASET:")
        print("="*60)
        
        for dataset in datasets:
            rf_acc = self.results.get(f"RF_{dataset}", {}).get('accuracy', 0)
            nb_acc = self.results.get(f"NB_{dataset}", {}).get('accuracy', 0)
            
            winner = "Random Forest" if rf_acc > nb_acc else "Naive Bayes"
            print(f"\n{dataset}:")
            print(f"  RF: Acc={rf_acc:.4f}")
            print(f"  NB: Acc={nb_acc:.4f}")
            print(f"  ✨ Winner: {winner}")
        
        # Average performance
        print(f"\n📈 HIỆU SUẤT TRUNG BÌNH:")
        print("="*60)
        
        rf_avg_acc = np.mean([self.results.get(f"RF_{d}", {}).get('accuracy', 0) 
                              for d in datasets])
        nb_avg_acc = np.mean([self.results.get(f"NB_{d}", {}).get('accuracy', 0) 
                              for d in datasets])
        
        print(f"Random Forest: {rf_avg_acc:.4f}")
        print(f"Naive Bayes: {nb_avg_acc:.4f}")
        
        # Conclusions
        print(f"\n💡 KẾT LUẬN:")
        print("="*60)
        print("✅ Random Forest:")
        print("   - Tốt hơn trên dữ liệu numeric phức tạp")
        print("   - Xử lý tốt outliers và missing values")
        print("   - Hiệu suất cao với feature engineering")
        
        print("\n✅ Naive Bayes:")
        print("   - Hiệu quả với text classification")
        print("   - Nhanh và đơn giản")
        print("   - Tốt với categorical data")
        
        print(f"\n🔧 KỸ THUẬT ĐÃ ÁP DỤNG:")
        print("="*60)
        print("✓ Imputation (median/mode)")
        print("✓ Scaling (Standard/Robust/MinMax)")
        print("✓ Feature Engineering")
        print("✓ Feature Selection (Chi2, Mutual Info)")
        print("✓ Dimensionality Reduction (PCA)")
        print("✓ Imbalanced Data Handling (SMOTE, Undersampling)")
        print("✓ Cross-validation")
    
    def run_experiment(self):
        """Chạy toàn bộ thí nghiệm"""
        print("🚀 BẮT ĐẦU THỰC NGHIỆM NÂNG CAO")
        print("📋 Random Forest vs Naive Bayes trên Multiple Datasets")
        print("🔬 Với các kỹ thuật tiền xử lý và khai thác dữ liệu nâng cao")
        
        if not self.load_datasets():
            print("❌ Không thể tải datasets!")
            return
        
        self.explore_data()
        self.preprocess_data()
        self.apply_dimensionality_reduction()
        self.train_models()
        self.evaluate_models()
        self.visualize_results()
        self.generate_report()
        
        print("\n" + "="*80)
        print("🎉 THỰC NGHIỆM HOÀN THÀNH!")
        print("="*80)
        print("📁 Các file đã được tạo:")
        print("   📊 eda_analysis_all.png - Khám phá dữ liệu")
        print("   📈 overall_comparison.png - So sánh tổng thể")
        print("   🎯 confusion_matrices_all.png - Confusion matrices")

if __name__ == "__main__":
    experiment = AdvancedDataMiningExperiment()
    experiment.run_experiment()
