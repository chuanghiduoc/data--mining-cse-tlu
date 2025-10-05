#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mục tiêu:
- Xây dựng thuật toán Random Forest từ đầu (không dùng thư viện)
- So sánh với Naive Bayes
- Áp dụng best practice: Split trước, sau đó fit/transform (tránh data leakage)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = ['DejaVu Sans']
plt.style.use('seaborn-v0_8')


# ============================================================================
# PHẦN 1: IMPLEMENT DECISION TREE TỪ ĐẦU
# ============================================================================

class DecisionTreeClassifier:
    """
    Cây quyết định (Decision Tree) được xây dựng từ đầu
    
    Thuật toán:
    1. Tìm split tốt nhất (feature, threshold) bằng Gini impurity
    2. Chia dữ liệu thành 2 nhánh (left, right) theo split
    3. Đệ quy xây dựng cây cho đến khi:
       - Đạt max_depth
       - Số mẫu quá ít (min_samples_split, min_samples_leaf)
       - Tất cả mẫu cùng class
    4. Dự đoán: Duyệt cây từ root đến leaf
    
    Cách hoạt động:
    - Mỗi node lưu: feature_idx, threshold, left_child, right_child
    - Leaf node lưu: predicted_class
    """
    
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1, max_features=None):
        """
        Khởi tạo Decision Tree
        
        Tham số:
        - max_depth: Độ sâu tối đa của cây (tránh overfitting)
        - min_samples_split: Số mẫu tối thiểu để chia node
        - min_samples_leaf: Số mẫu tối thiểu ở leaf node
        - max_features: Số features xét tại mỗi split (None = xét tất cả)
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.tree = None
    
    def gini_impurity(self, y):
        """
        Tính Gini impurity - Độ "không tinh khiết" của node
        
        Công thức: Gini = 1 - sum(p_i^2)
        trong đó p_i là tỉ lệ của class i
        
        Ví dụ:
        - Node có 100% class 0 → Gini = 0 (tinh khiết hoàn toàn)
        - Node có 50% class 0, 50% class 1 → Gini = 0.5 (không tinh khiết)
        
        Tham số:
        - y: Mảng chứa các nhãn (labels)
        
        Trả về:
        - Gini impurity (giá trị từ 0 đến 0.5 với binary classification)
        """
        if len(y) == 0:
            return 0
        
        # Đếm số lượng mỗi class
        _, counts = np.unique(y, return_counts=True)
        
        # Tính tỉ lệ (probability) của mỗi class
        probabilities = counts / len(y)
        
        # Tính Gini = 1 - sum(p^2)
        return 1 - np.sum(probabilities ** 2)
    
    def split_data(self, X, y, feature_idx, threshold):
        """
        Chia dữ liệu thành 2 phần dựa trên feature và threshold
        
        Left: X[:, feature_idx] <= threshold
        Right: X[:, feature_idx] > threshold
        
        Tham số:
        - X: Ma trận features (n_samples, n_features)
        - y: Vector labels (n_samples,)
        - feature_idx: Chỉ số của feature dùng để split
        - threshold: Ngưỡng để so sánh
        
        Trả về:
        - X_left, y_left: Dữ liệu bên trái
        - X_right, y_right: Dữ liệu bên phải
        """
        # Tạo mask cho left và right
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask  # Phủ định của left_mask
        
        return X[left_mask], y[left_mask], X[right_mask], y[right_mask]
    
    def find_best_split(self, X, y, max_features=None):
        """
        Tìm split tốt nhất (feature, threshold) để chia node
        
        Thuật toán:
        1. Random chọn m features (max_features) để xét
        2. Với mỗi feature trong m features:
           3. Với mỗi giá trị unique của feature (làm threshold):
              4. Split dữ liệu thành left và right
              5. Tính weighted Gini sau khi split
              6. Lưu lại split tốt nhất (Gini thấp nhất)
        
        Weighted Gini = (n_left * Gini_left + n_right * Gini_right) / n_total
        
        Tham số:
        - X: Ma trận features
        - y: Vector labels
        - max_features: Số features xét tại mỗi split (None = xét tất cả)
        
        Trả về:
        - best_feature: Chỉ số feature tốt nhất
        - best_threshold: Ngưỡng tốt nhất
        - best_gini: Giá trị Gini thấp nhất
        """
        best_gini = float('inf')  # Bắt đầu với giá trị vô cùng lớn
        best_feature = None
        best_threshold = None
        
        n_samples, n_features = X.shape
        
        # RANDOM FEATURE SELECTION (theo theory Random Forest)
        # Nếu không chỉ định max_features, dùng sqrt(n_features)
        if max_features is None:
            max_features = n_features
        else:
            max_features = min(max_features, n_features)
        
        # Chọn ngẫu nhiên max_features features để xét
        # Đây là điểm khác biệt giữa Random Forest và Decision Tree thông thường!
        feature_indices = np.random.choice(n_features, max_features, replace=False)
        
        # Chỉ duyệt qua SUBSET các features đã chọn (không phải tất cả)
        for feature_idx in feature_indices:
            # Lấy các giá trị unique của feature này để làm threshold
            thresholds = np.unique(X[:, feature_idx])
            
            # Thử mỗi threshold
            for threshold in thresholds:
                # Chia dữ liệu
                X_left, y_left, X_right, y_right = self.split_data(X, y, feature_idx, threshold)
                
                # Bỏ qua nếu split không hợp lệ (quá ít mẫu)
                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue
                
                # Tính weighted Gini sau split
                n_left, n_right = len(y_left), len(y_right)
                gini_left = self.gini_impurity(y_left)
                gini_right = self.gini_impurity(y_right)
                
                # Weighted average (trung bình có trọng số)
                weighted_gini = (n_left * gini_left + n_right * gini_right) / n_samples
                
                # Cập nhật best split nếu tìm được split tốt hơn
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gini
    
    def build_tree(self, X, y, depth=0, max_features=None):
        """
        Xây dựng cây quyết định bằng đệ quy
        
        Thuật toán:
        1. Kiểm tra điều kiện dừng (stopping criteria):
           - Đạt max_depth
           - Chỉ còn 1 class
           - Quá ít mẫu
           → Tạo leaf node
        
        2. Tìm best split (với random feature selection)
        
        3. Chia dữ liệu thành left và right
        
        4. Đệ quy xây dựng left_subtree và right_subtree
        
        5. Trả về node hiện tại
        
        Tham số:
        - X: Ma trận features
        - y: Vector labels
        - depth: Độ sâu hiện tại của node (bắt đầu từ 0)
        - max_features: Số features xét tại mỗi split (cho Random Forest)
        
        Trả về:
        - tree_node: Dictionary chứa thông tin node
          + Nếu là leaf: {'leaf': True, 'value': predicted_class}
          + Nếu là internal node: {'leaf': False, 'feature': idx, 'threshold': val, 
                                   'left': left_subtree, 'right': right_subtree}
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # ĐIỀU KIỆN DỪNG (stopping criteria)
        if (depth >= self.max_depth or          # Đã đạt độ sâu tối đa
            n_classes == 1 or                    # Chỉ còn 1 class (tinh khiết)
            n_samples < self.min_samples_split): # Quá ít mẫu để chia tiếp
            
            # Tạo leaf node: Chọn class xuất hiện nhiều nhất
            leaf_value = np.bincount(y.astype(int)).argmax()
            return {'leaf': True, 'value': leaf_value}
        
        # TÌM BEST SPLIT (với random feature selection)
        best_feature, best_threshold, best_gini = self.find_best_split(X, y, max_features)
        
        # Nếu không tìm được split hợp lệ, tạo leaf node
        if best_feature is None:
            leaf_value = np.bincount(y.astype(int)).argmax()
            return {'leaf': True, 'value': leaf_value}
        
        # CHIA DỮ LIỆU thành left và right
        X_left, y_left, X_right, y_right = self.split_data(X, y, best_feature, best_threshold)
        
        # ĐỆ QUY xây dựng left subtree và right subtree
        left_subtree = self.build_tree(X_left, y_left, depth + 1, max_features)
        right_subtree = self.build_tree(X_right, y_right, depth + 1, max_features)
        
        # Trả về internal node
        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def fit(self, X, y):
        """
        Train cây quyết định
        
        Tham số:
        - X: Ma trận features (n_samples, n_features)
        - y: Vector labels (n_samples,)
        """
        self.tree = self.build_tree(X, y, max_features=self.max_features)
        return self
    
    def predict_sample(self, x, tree):
        """
        Dự đoán cho 1 mẫu bằng cách duyệt cây
        
        Thuật toán:
        1. Nếu là leaf node → trả về predicted class
        2. Nếu là internal node:
           - So sánh x[feature] với threshold
           - Nếu <= threshold → duyệt left subtree
           - Nếu > threshold → duyệt right subtree
        
        Tham số:
        - x: 1 mẫu cần dự đoán (vector 1 chiều)
        - tree: Node hiện tại của cây
        
        Trả về:
        - predicted_class: Class dự đoán (0 hoặc 1)
        """
        # Nếu là leaf node, trả về giá trị dự đoán
        if tree['leaf']:
            return tree['value']
        
        # Nếu là internal node, duyệt tiếp
        if x[tree['feature']] <= tree['threshold']:
            # Đi sang bên trái
            return self.predict_sample(x, tree['left'])
        else:
            # Đi sang bên phải
            return self.predict_sample(x, tree['right'])
    
    def predict(self, X):
        """
        Dự đoán cho nhiều mẫu
        
        Tham số:
        - X: Ma trận features (n_samples, n_features)
        
        Trả về:
        - predictions: Vector chứa các dự đoán (n_samples,)
        """
        return np.array([self.predict_sample(x, self.tree) for x in X])


# ============================================================================
# PHẦN 2: IMPLEMENT RANDOM FOREST TỪ ĐẦU
# ============================================================================

class RandomForestClassifier:
    """
    Random Forest Classifier được xây dựng từ đầu
    
    Thuật toán:
    1. Tạo n_estimators cây quyết định
    2. Với mỗi cây:
       a. Bootstrap sampling: Lấy ngẫu nhiên n mẫu (có lặp lại) từ training set
       b. Xây dựng cây quyết định trên bootstrap sample
       c. Tại mỗi node, CHỈ xét ngẫu nhiên một subset các feature (feature bagging)
    3. Dự đoán: Majority voting từ tất cả các cây
    
    Ưu điểm:
    - Giảm overfitting so với single decision tree
    - Tăng tính robust (ổn định) nhờ ensemble
    - Parallel training (các cây độc lập)
    
    Cách hoạt động:
    - Bootstrap: Mỗi cây train trên subset khác nhau của data
    - Feature randomness: Mỗi split chỉ xét một phần các feature
    - Voting: Kết quả cuối = class được vote nhiều nhất
    """
    
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=2, 
                 min_samples_leaf=1, max_features='sqrt', random_state=None):
        """
        Khởi tạo Random Forest
        
        Tham số:
        - n_estimators: Số lượng cây trong forest (nhiều hơn → chính xác hơn nhưng chậm hơn)
        - max_depth: Độ sâu tối đa của mỗi cây
        - min_samples_split: Số mẫu tối thiểu để chia node
        - min_samples_leaf: Số mẫu tối thiểu ở leaf
        - max_features: Số feature xét tại mỗi split ('sqrt', 'log2', hoặc số nguyên)
        - random_state: Random seed để kết quả lặp lại được
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []  # Lưu tất cả các cây
        self.n_features_ = None  # Sẽ được set khi fit
        
        # Set random seed nếu có
        if random_state is not None:
            np.random.seed(random_state)
    
    def bootstrap_sample(self, X, y):
        """
        Tạo bootstrap sample (sampling with replacement)
        
        Bootstrap: Lấy ngẫu nhiên n mẫu TỪ training set (CÓ THỂ LẶP LẠI)
        → Một số mẫu có thể xuất hiện nhiều lần, một số không xuất hiện
        
        Ví dụ: X = [A, B, C, D, E]
        Bootstrap sample có thể là: [A, A, C, D, E] hoặc [B, B, B, D, E], v.v.
        
        Tham số:
        - X: Ma trận features gốc
        - y: Vector labels gốc
        
        Trả về:
        - X_bootstrap: Ma trận features sau bootstrap
        - y_bootstrap: Vector labels sau bootstrap
        """
        n_samples = X.shape[0]
        
        # Chọn ngẫu nhiên n_samples chỉ số (có lặp lại)
        indices = np.random.choice(n_samples, n_samples, replace=True)
        
        return X[indices], y[indices]
    
    def fit(self, X, y):
        """
        Train Random Forest
        
        Thuật toán:
        1. Lặp n_estimators lần:
           a. Tạo bootstrap sample từ (X, y)
           b. Xây dựng 1 decision tree trên bootstrap sample (với random feature selection)
           c. Lưu tree vào danh sách
        
        Tham số:
        - X: Ma trận features (n_samples, n_features)
        - y: Vector labels (n_samples,)
        """
        self.trees = []
        self.n_features_ = X.shape[1]
        
        # Xác định số features xét tại mỗi split
        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(self.n_features_))
        elif self.max_features == 'log2':
            max_features = int(np.log2(self.n_features_))
        elif isinstance(self.max_features, int):
            max_features = self.max_features
        else:
            max_features = self.n_features_
        
        print(f"   Training {self.n_estimators} trees...")
        print(f"   Max features per split: {max_features}/{self.n_features_}")
        
        # Xây dựng n_estimators cây
        for i in range(self.n_estimators):
            # Bước 1: Bootstrap sampling
            # Lấy ngẫu nhiên n mẫu (có lặp lại) từ training set
            X_bootstrap, y_bootstrap = self.bootstrap_sample(X, y)
            
            # Bước 2: Xây dựng cây quyết định
            # QUAN TRỌNG: Truyền max_features để có random feature selection
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features  # ← Random feature selection!
            )
            tree.fit(X_bootstrap, y_bootstrap)
            
            # Bước 3: Lưu cây vào danh sách
            self.trees.append(tree)
            
            # Hiển thị tiến độ
            if (i + 1) % 20 == 0:
                print(f"      Progress: {i+1}/{self.n_estimators} trees")
        
        print(f"   Completed training {len(self.trees)} trees")
        return self
    
    def predict(self, X):
        """
        Dự đoán bằng MAJORITY VOTING
        
        Thuật toán:
        1. Lấy dự đoán từ tất cả các cây
        2. Với mỗi mẫu:
           - Đếm số vote cho mỗi class
           - Chọn class có nhiều vote nhất
        
        Ví dụ: Có 5 cây dự đoán cho mẫu X:
        Tree 1 → class 0
        Tree 2 → class 1
        Tree 3 → class 1
        Tree 4 → class 1
        Tree 5 → class 0
        → Kết quả: class 1 (vì 3 > 2)
        
        Tham số:
        - X: Ma trận features cần dự đoán
        
        Trả về:
        - predictions: Vector chứa dự đoán cuối cùng
        """
        # Bước 1: Lấy dự đoán từ tất cả các cây
        # tree_predictions[i, j] = dự đoán của tree i cho mẫu j
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Bước 2: Majority voting cho mỗi mẫu
        predictions = []
        for i in range(X.shape[0]):
            # Lấy tất cả các vote cho mẫu thứ i
            votes = tree_predictions[:, i]
            
            # Đếm số vote và chọn class có nhiều vote nhất
            # np.bincount đếm số lần xuất hiện của mỗi giá trị
            # argmax trả về chỉ số của giá trị lớn nhất
            prediction = np.bincount(votes.astype(int)).argmax()
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """
        Dự đoán xác suất của mỗi class
        
        Xác suất = Tỉ lệ cây vote cho class đó
        
        Ví dụ: Có 100 cây, 65 cây vote class 1, 35 cây vote class 0
        → P(class 0) = 0.35, P(class 1) = 0.65
        
        Tham số:
        - X: Ma trận features cần dự đoán
        
        Trả về:
        - probabilities: Ma trận xác suất (n_samples, n_classes)
          probabilities[i, j] = xác suất mẫu i thuộc class j
        """
        # Lấy dự đoán từ tất cả các cây
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        
        n_samples = X.shape[0]
        n_classes = 2  # Binary classification (0 và 1)
        
        # Khởi tạo ma trận xác suất
        probabilities = np.zeros((n_samples, n_classes))
        
        # Tính xác suất cho mỗi mẫu
        for i in range(n_samples):
            # Lấy tất cả vote cho mẫu thứ i
            votes = tree_predictions[:, i]
            
            # Tính tỉ lệ vote cho mỗi class
            for cls in range(n_classes):
                probabilities[i, cls] = np.sum(votes == cls) / len(self.trees)
        
        return probabilities


# ============================================================================
# PHẦN 3: EXPERIMENT CLASS - CHẠY THÍ NGHIỆM
# ============================================================================

class WineQualityExperiment:
    """
    Class thực hiện thí nghiệm so sánh Random Forest (từ đầu) và Naive Bayes
    trên Wine Quality dataset
    
    Quy trình ĐÚNG (Best Practice):
    1. Load data
    2. Explore data (EDA)
    3. Prepare data:
       a. Extract features & target
       b. SPLIT train/test TRƯỚC
       c. Fit preprocessing (scaler) trên TRAIN
       d. Transform cả train và test
    4. Train models
    5. Evaluate models
    6. Visualize results
    7. Generate report
    
    Quan trọng: TRÁNH DATA LEAKAGE!
    - PHẢI split trước khi preprocessing
    - FIT scaler/imputer CHỈ trên train set
    - Transform test set bằng statistics từ train
    """
    
    def __init__(self):
        """Khởi tạo experiment"""
        self.results = {}  # Lưu kết quả đánh giá
        
    def load_data(self):
        """
        Bước 1: Load Wine Quality dataset
        
        Dataset: winequality-red.csv
        - 1599 mẫu rượu đỏ
        - 11 features: fixed acidity, volatile acidity, citric acid, v.v.
        - 1 target: quality (0-10)
        """
        print("="*80)
        print("STEP 1: LOAD DATA")
        print("="*80)
        
        try:
            df = pd.read_csv('datasets/winequality-red.csv')
            print(f"Loaded Wine dataset: {df.shape[0]} samples, {df.shape[1]} columns")
            self.df = df
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def explore_data(self):
        """
        Bước 2: Khám phá dữ liệu (EDA - Exploratory Data Analysis)
        
        Mục tiêu:
        - Hiểu về phân bố dữ liệu
        - Phát hiện missing values, outliers
        - Xem correlation giữa các features
        """
        print("\n" + "="*80)
        print("STEP 2: EXPLORATORY DATA ANALYSIS")
        print("="*80)
        
        print(f"\nShape: {self.df.shape}")
        print(f"\nQuality distribution:")
        print(self.df['quality'].value_counts().sort_index())
        
        # Chuyển thành binary classification
        # quality >= 6 → good wine (1)
        # quality < 6 → bad wine (0)
        binary = (self.df['quality'] >= 6).astype(int)
        print(f"\nBinary classification (quality >= 6):")
        print(f"  Bad wine (0): {(binary == 0).sum()}")
        print(f"  Good wine (1): {(binary == 1).sum()}")
    
    def prepare_data(self):
        """
        Bước 3: Chuẩn bị dữ liệu theo BEST PRACTICE
        
        QUY TRÌNH ĐÚNG:
        1. Extract X (features) và y (target)
        2. Feature engineering (nếu cần)
        3. SPLIT train/test TRƯỚC TIÊN
        4. Build preprocessing pipeline
        5. FIT pipeline trên TRAIN set
        6. TRANSFORM cả train và test
        
        TẠI SAO PHẢI SPLIT TRƯỚC?
        - Tránh data leakage: Test set KHÔNG ĐƯỢC ảnh hưởng đến training
        - Preprocessing (scaling, imputation) CHỈ học từ train set
        - Test set dùng để đánh giá KHÁCH QUAN
        
        VÍ DỤ DATA LEAKAGE:
        SAI: Scale toàn bộ data → split → train
             → Scaler đã thấy dữ liệu test → leakage!
        
        ĐÚNG: Split → fit scaler trên train → transform test
              → Scaler KHÔNG thấy dữ liệu test → OK!
        """
        print("\n" + "="*80)
        print("STEP 3: PREPARE DATA (NO DATA LEAKAGE)")
        print("="*80)
        
        # Bước 3.1: Extract features và target
        print("\nExtracting features and target...")
        
        # Target: Binary classification
        y = (self.df['quality'] >= 6).astype(int).values
        
        # Features: Tất cả các cột trừ 'quality'
        X = self.df.drop('quality', axis=1).copy()
        
        # Bước 3.2: Feature Engineering (domain knowledge)
        # Tạo thêm các feature mới từ các feature cũ
        print("Feature engineering:")
        print("  - alcohol x sulphates (tuong tac giua do con va sulphates)")
        print("  - volatile acidity / fixed acidity (ty le acid de bay hoi)")
        
        X['alcohol_sulphates'] = X['alcohol'] * X['sulphates']
        X['volatile_total_acidity'] = X['volatile acidity'] / (X['fixed acidity'] + 0.001)
        
        print(f"Features after engineering: {X.shape[1]}")
        
        # Bước 3.3: SPLIT TRƯỚC (QUAN TRỌNG!)
        # 80% train, 20% test
        # stratify=y đảm bảo tỉ lệ class giống nhau ở train và test
        print("\nSplitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Bước 3.4: Build preprocessing pipeline
        # RobustScaler: Chống outliers tốt hơn StandardScaler
        # Công thức: X_scaled = (X - median) / IQR
        print("\nBuilding preprocessing pipeline...")
        pipeline = Pipeline([
            ('scaler', RobustScaler())
        ])
        print("  Pipeline: RobustScaler (robust to outliers)")
        
        # Bước 3.5: FIT trên TRAIN set
        # Scaler học median và IQR TỪ TRAIN set
        print("\nFitting pipeline on TRAIN set...")
        X_train_transformed = pipeline.fit_transform(X_train)
        
        # Bước 3.6: TRANSFORM trên TEST set
        # Dùng median và IQR đã học từ TRAIN
        # KHÔNG fit lại trên test → Tránh leakage!
        print("Transforming TEST set (using TRAIN statistics)...")
        X_test_transformed = pipeline.transform(X_test)
        
        # Lưu dữ liệu đã xử lý
        self.X_train = X_train_transformed
        self.X_test = X_test_transformed
        self.y_train = y_train
        self.y_test = y_test
        
        print("\nData preparation completed - NO data leakage!")
    
    def train_models(self):
        """
        Bước 4: Train các model
        
        1. Random Forest (FROM SCRATCH)
        2. Naive Bayes (sklearn)
        """
        print("\n" + "="*80)
        print("STEP 4: TRAIN MODELS")
        print("="*80)
        
        self.models = {}
        
        # Model 1: Random Forest (từ đầu)
        print("\nTraining Random Forest (FROM SCRATCH)...")
        print("Parameters:")
        print("  - n_estimators: 100 (100 cay)")
        print("  - max_depth: 15 (do sau toi da 15)")
        print("  - min_samples_split: 5 (can 5 mau de chia)")
        print("  - min_samples_leaf: 2 (moi leaf can 2 mau)")
        
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        rf.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf
        print("Random Forest trained!\n")
        
        # Model 2: Naive Bayes
        print("Training Naive Bayes...")
        nb = GaussianNB()
        nb.fit(self.X_train, self.y_train)
        self.models['Naive Bayes'] = nb
        print("Naive Bayes trained!")
    
    def evaluate_models(self):
        """
        Bước 5: Đánh giá các model
        
        Metrics:
        - Accuracy: (TP + TN) / Total - Tỉ lệ dự đoán đúng
        - Precision: TP / (TP + FP) - Trong các dự đoán positive, bao nhiêu đúng?
        - Recall: TP / (TP + FN) - Trong các positive thật, bắt được bao nhiêu?
        - F1-Score: Harmonic mean của Precision và Recall
        - AUC-ROC: Khả năng phân biệt 2 class
        
        TP = True Positive, TN = True Negative
        FP = False Positive, FN = False Negative
        """
        print("\n" + "="*80)
        print("STEP 5: EVALUATE MODELS")
        print("="*80)
        
        for model_name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"{model_name}")
            print(f"{'='*60}")
            
            # Dự đoán trên test set
            y_pred = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test)[:, 1]  # Xác suất class 1
            
            # Tính các metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_proba)
            
            # Lưu kết quả
            self.results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'y_pred': y_pred,
                'y_proba': y_proba
            }
            
            # In ra kết quả
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            print(f"AUC-ROC:   {auc:.4f}")
    
    def visualize_results(self):
        """
        Bước 6: Trực quan hóa kết quả
        
        Vẽ các biểu đồ:
        1. So sánh metrics
        2. Confusion matrices
        3. ROC curves
        """
        print("\n" + "="*80)
        print("STEP 6: VISUALIZATION")
        print("="*80)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Random Forest (From Scratch) vs Naive Bayes - Wine Quality', 
                    fontsize=16, fontweight='bold')
        
        # 1. So sánh metrics
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        rf_scores = [self.results['Random Forest'][m] for m in metrics]
        nb_scores = [self.results['Naive Bayes'][m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        axes[0, 0].bar(x - width/2, rf_scores, width, label='Random Forest (Scratch)', 
                      color='darkgreen', alpha=0.8)
        axes[0, 0].bar(x + width/2, nb_scores, width, label='Naive Bayes', 
                      color='orange', alpha=0.8)
        axes[0, 0].set_xlabel('Metrics', fontsize=11)
        axes[0, 0].set_ylabel('Score', fontsize=11)
        axes[0, 0].set_title('Performance Comparison', fontsize=12, fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(metrics, fontsize=10)
        axes[0, 0].legend(fontsize=9)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 1.1])
        
        # 2. Confusion Matrix - Random Forest
        cm_rf = confusion_matrix(self.y_test, self.results['Random Forest']['y_pred'])
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[0, 1],
                   xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'])
        axes[0, 1].set_title('Confusion Matrix - Random Forest', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Actual', fontsize=11)
        axes[0, 1].set_xlabel('Predicted', fontsize=11)
        
        # 3. Confusion Matrix - Naive Bayes
        cm_nb = confusion_matrix(self.y_test, self.results['Naive Bayes']['y_pred'])
        sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Oranges', ax=axes[1, 0],
                   xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'])
        axes[1, 0].set_title('Confusion Matrix - Naive Bayes', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Actual', fontsize=11)
        axes[1, 0].set_xlabel('Predicted', fontsize=11)
        
        # 4. ROC Curves
        for model_name in ['Random Forest', 'Naive Bayes']:
            fpr, tpr, _ = roc_curve(self.y_test, self.results[model_name]['y_proba'])
            auc = self.results[model_name]['auc']
            color = 'darkgreen' if model_name == 'Random Forest' else 'orange'
            axes[1, 1].plot(fpr, tpr, label=f'{model_name} (AUC={auc:.3f})', 
                          color=color, linewidth=2)
        
        axes[1, 1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        axes[1, 1].set_xlabel('False Positive Rate', fontsize=11)
        axes[1, 1].set_ylabel('True Positive Rate', fontsize=11)
        axes[1, 1].set_title('ROC Curves', fontsize=12, fontweight='bold')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('wine_results_scratch.png', dpi=150, bbox_inches='tight')
        print("Saved: wine_results_scratch.png")
        plt.close()
    
    def generate_report(self):
        """
        Bước 7: Tạo báo cáo kết quả
        """
        print("\n" + "="*80)
        print("STEP 7: FINAL REPORT")
        print("="*80)
        
        rf_acc = self.results['Random Forest']['accuracy']
        nb_acc = self.results['Naive Bayes']['accuracy']
        
        print(f"\nRESULTS:")
        print(f"  Random Forest (FROM SCRATCH): Accuracy = {rf_acc:.4f}")
        print(f"  Naive Bayes:                   Accuracy = {nb_acc:.4f}")
        
        winner = 'Random Forest' if rf_acc > nb_acc else 'Naive Bayes'
        print(f"\nWinner: {winner}")
        
        print(f"\nKEY POINTS:")
        print("  1. Random Forest implemented FROM SCRATCH")
        print("     - Bootstrap sampling: Moi cay train tren subset ngau nhien")
        print("     - Decision trees: Dung Gini impurity de split")
        print("     - Majority voting: Du doan cuoi = vote tu tat ca cac cay")
        print("  2. Best practice applied: Split FIRST, then fit/transform")
        print("     - Tranh data leakage")
        print("     - Scaler chi hoc tu train set")
        print("  3. Random Feature Selection: Chi xet subset features tai moi split")
        print("  4. NO data leakage!")
    
    def run(self):
        """Chạy toàn bộ thí nghiệm"""
        print("WINE QUALITY CLASSIFICATION")
        print("Random Forest (FROM SCRATCH) vs Naive Bayes\n")
        
        if not self.load_data():
            return
        
        self.explore_data()
        self.prepare_data()
        self.train_models()
        self.evaluate_models()
        self.visualize_results()
        self.generate_report()
        
        print("\n" + "="*80)
        print("EXPERIMENT COMPLETED!")
        print("="*80)


if __name__ == "__main__":
    experiment = WineQualityExperiment()
    experiment.run()
