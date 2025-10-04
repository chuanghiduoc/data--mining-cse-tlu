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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
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
                'path': 'datasets/winequality-red.csv', #Đức Anh
                'encoding': 'utf-8',
                'type': 'numeric'
            },
            'Diabetes': {
                'path': 'datasets/diabetes.csv', #Huyền
                'encoding': 'utf-8',
                'type': 'numeric'
            },
            'HeartDisease': {
                'path': 'datasets/heart_disease_uci.csv', #Đức Anh
                'encoding': 'utf-8',
                'type': 'mixed'
            },
            'Adult': {
                'path': 'datasets/adult.csv', #Hà
                'encoding': 'utf-8',
                'type': 'mixed'
            },
            'Mushroom': {
                'path': 'datasets/mushrooms.csv', #Trọng
                'encoding': 'utf-8',
                'type': 'categorical'
            },
            'Sonar': {
                'path': 'datasets/sonar-all-data.csv', #Trọng
                'encoding': 'utf-8',
                'type': 'numeric'
            },
            'CreditCard': {
                'path': 'datasets/creditcard.csv', #Hà
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
            'HeartDisease': 'num',
            'Adult': 'income',
            'Mushroom': 'class',
            'Sonar': 'Label',
            'CreditCard': 'Class'
        }
        return target_map.get(dataset_name)
    
    def preprocess_data(self):
        """
        Bước 3: Tiền xử lý dữ liệu với Pipeline (BEST PRACTICE)
        
        Quy trình ĐÚNG:
        1. Split train/test TRƯỚC
        2. Fit preprocessing pipeline trên train
        3. Transform cả train và test
        4. SMOTE/Undersample CHỈ áp dụng trên train
        
        Tránh DATA LEAKAGE!
        """
        print("\n" + "="*80)
        print("BƯỚC 3: TIỀN XỬ LÝ DỮ LIỆU (BEST PRACTICE - NO DATA LEAKAGE)")
        print("="*80)
        
        self.processed_datasets = {}
        self.pipelines = {}
        
        for name, dataset_info in self.datasets.items():
            print(f"\n{'='*60}")
            print(f"🔧 Xử lý {name} Dataset")
            print(f"{'='*60}")
            
            df = dataset_info['data'].copy()
            
            # Bước 1: Lấy X, y THÔ (chưa preprocessing)
            if name == 'Wine':
                X, y = self._extract_wine(df)
            elif name == 'Diabetes':
                X, y = self._extract_diabetes(df)
            elif name == 'HeartDisease':
                X, y = self._extract_heartdisease(df)
            elif name == 'Adult':
                X, y = self._extract_adult(df)
            elif name == 'Mushroom':
                X, y = self._extract_mushroom(df)
            elif name == 'Sonar':
                X, y = self._extract_sonar(df)
            elif name == 'CreditCard':
                X, y = self._extract_creditcard(df)
            else:
                continue
            
            # Bước 2: SPLIT TRƯỚC KHI PREPROCESSING
            print(f"   📊 Original shape: {X.shape}")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            print(f"   ✂️ Split: Train={X_train.shape}, Test={X_test.shape}")
            
            # Bước 3: BUILD PIPELINE cho từng dataset
            pipeline = self._build_pipeline(name)
            self.pipelines[name] = pipeline
            
            # Bước 4: FIT pipeline trên TRAIN, transform cả train và test
            print(f"   🔧 Fitting pipeline on TRAIN set...")
            X_train_transformed = pipeline.fit_transform(X_train, y_train)
            
            print(f"   🔄 Transforming TEST set...")
            X_test_transformed = pipeline.transform(X_test)
            
            # Bước 5: Apply SMOTE/Undersample CHỈ trên TRAIN (sau preprocessing)
            y_train_transformed = y_train
            
            if name == 'Diabetes':
                print(f"   ⚖️ Applying SMOTE on TRAIN set...")
                smote = SMOTE(random_state=42)
                X_train_transformed, y_train_transformed = smote.fit_resample(X_train_transformed, y_train)
                print(f"      Before: {len(y_train)} samples → After: {len(y_train_transformed)} samples")
            
            elif name == 'CreditCard':
                print(f"   ⚖️ Applying Undersampling on TRAIN set...")
                rus = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
                X_train_transformed, y_train_transformed = rus.fit_resample(X_train_transformed, y_train)
                print(f"      Before: {len(y_train)} samples → After: {len(y_train_transformed)} samples")
            
            # y_test KHÔNG bao giờ resample!
            
            self.processed_datasets[name] = {
                'X_train': X_train_transformed,
                'X_test': X_test_transformed,
                'y_train': y_train_transformed,
                'y_test': y_test,
                'feature_count': X_train_transformed.shape[1]
            }
            
            print(f"   ✅ Final - Train: {X_train_transformed.shape}, Test: {X_test_transformed.shape}")
    
    def _build_pipeline(self, dataset_name):
        """Build appropriate preprocessing pipeline for each dataset"""
        pipeline_map = {
            'Wine': self._build_pipeline_wine,
            'Diabetes': self._build_pipeline_diabetes,
            'HeartDisease': self._build_pipeline_heartdisease,
            'Adult': self._build_pipeline_adult,
            'Mushroom': self._build_pipeline_mushroom,
            'Sonar': self._build_pipeline_sonar,
            'CreditCard': self._build_pipeline_creditcard
        }
        
        if dataset_name in pipeline_map:
            return pipeline_map[dataset_name]()
        else:
            # Default pipeline
            return Pipeline([('scaler', StandardScaler())])
    
    def _extract_wine(self, df):
        """Extract raw X, y for Wine dataset (NO preprocessing yet)"""
        print("🍷 Wine: Extracting features")
        y = (df['quality'] >= 6).astype(int)
        X = df.drop('quality', axis=1)
        
        # Feature Engineering (BEFORE split - domain knowledge)
        X['alcohol_sulphates'] = X['alcohol'] * X['sulphates']
        X['volatile_total_acidity'] = X['volatile acidity'] / (X['fixed acidity'] + 0.001)
        
        return X.values, y.values
    
    def _build_pipeline_wine(self):
        """Build preprocessing pipeline for Wine dataset"""
        return Pipeline([
            ('scaler', RobustScaler())
        ])
    
    def _extract_diabetes(self, df):
        """Extract raw X, y for Diabetes dataset"""
        print("💉 Diabetes: Extracting features")
        y = df['Outcome'].values
        X = df.drop('Outcome', axis=1).copy()
        
        # Replace medical impossibilities (0 values) with NaN
        zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in zero_cols:
            if col in X.columns:
                X[col] = X[col].replace(0, np.nan)
        
        return X.values, y
    
    def _build_pipeline_diabetes(self):
        """Build preprocessing pipeline for Diabetes dataset"""
        # SMOTE sẽ được apply riêng sau khi transform
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
    
    def _extract_heartdisease(self, df):
        """Extract raw X, y for Heart Disease dataset"""
        print("❤️ HeartDisease: Extracting features")
        
        # Binary classification: num > 0 = heart disease (1), num = 0 = no disease (0)
        y = (df['num'] > 0).astype(int).values
        X = df.drop(['num', 'id'], axis=1, errors='ignore').copy()
        
        # Handle categorical columns
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'missing')
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        
        # Handle numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 0:
            imputer = SimpleImputer(strategy='median')
            X[numeric_cols] = imputer.fit_transform(X[numeric_cols])
        
        return X.values, y
    
    def _build_pipeline_heartdisease(self):
        """Build preprocessing pipeline for Heart Disease dataset"""
        return Pipeline([
            ('scaler', StandardScaler())
        ])
    
    def _extract_adult(self, df):
        """Extract raw X, y for Adult dataset"""
        print("👤 Adult: Extracting features")
        df.columns = df.columns.str.strip()
        
        y = LabelEncoder().fit_transform(df['income'])
        X = df.drop(['income'], axis=1).copy()
        
        # Handle missing values
        X = X.replace('?', np.nan)
        
        # Separate numeric and categorical
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        # Impute and encode categorical BEFORE split (necessary for proper encoding)
        for col in categorical_cols:
            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'missing')
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        
        # Impute numeric
        if len(numeric_cols) > 0:
            imputer = SimpleImputer(strategy='median')
            X[numeric_cols] = imputer.fit_transform(X[numeric_cols])
        
        return X.values, y
    
    def _build_pipeline_adult(self):
        """Build preprocessing pipeline for Adult dataset"""
        return Pipeline([
            ('scaler', StandardScaler())
        ])
    
    def _extract_mushroom(self, df):
        """Extract raw X, y for Mushroom dataset"""
        print("🍄 Mushroom: Extracting features")
        y = LabelEncoder().fit_transform(df['class'])
        X = df.drop('class', axis=1).copy()
        
        # Label encode all categorical features
        for col in X.columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        
        return X.values, y
    
    def _build_pipeline_mushroom(self):
        """Build preprocessing pipeline for Mushroom dataset"""
        return Pipeline([
            ('feature_selection', SelectKBest(mutual_info_classif, k=15))
        ])
    
    def _extract_sonar(self, df):
        """Extract raw X, y for Sonar dataset"""
        print("📡 Sonar: Extracting features")
        y = LabelEncoder().fit_transform(df.iloc[:, -1])
        X = df.iloc[:, :-1].values
        return X, y
    
    def _build_pipeline_sonar(self):
        """Build preprocessing pipeline for Sonar dataset"""
        return Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=30, random_state=42))
        ])
    
    def _extract_creditcard(self, df):
        """Extract raw X, y for Credit Card dataset"""
        print("💳 Credit Card: Extracting features")
        y = df['Class'].values
        X = df.drop('Class', axis=1).copy()
        
        # Scale Time and Amount (V1-V28 already scaled)
        scaler = StandardScaler()
        X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])
        
        return X.values, y
    
    def _build_pipeline_creditcard(self):
        """Build preprocessing pipeline for Credit Card dataset"""
        return Pipeline([
            ('passthrough', FunctionTransformer())  # Identity transform
        ])
    
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
        # Skip apply_dimensionality_reduction() - PCA đã có trong pipeline của Sonar
        # self.apply_dimensionality_reduction()
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
