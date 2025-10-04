#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VÍ DỤ MINH HỌA RANDOM FOREST: DỰ ĐOÁN HIỆU SUẤT HỌC TẬP SINH VIÊN
Ví dụ tự tạo, không có trên mạng
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings

# Tắt warnings không cần thiết
warnings.filterwarnings('ignore', category=UserWarning)

# Tạo dataset ví dụ: Dự đoán sinh viên đậu/rớt môn Khai phá dữ liệu
data = {
    'ID': range(1, 11),
    'Gio_hoc': [20, 15, 25, 12, 30, 18, 10, 22, 28, 16],
    'Bai_tap': [8, 6, 10, 4, 9, 7, 3, 8, 10, 5],
    'Diem_danh': [85, 90, 95, 70, 88, 92, 65, 80, 96, 75],
    'Kinh_nghiem_lap_trinh': [1, 0, 1, 0, 1, 1, 0, 0, 1, 0],  # 1=Có, 0=Không
    'Ket_qua': [1, 0, 1, 0, 1, 1, 0, 0, 1, 0]  # 1=Đậu, 0=Rớt
}

df = pd.DataFrame(data)

print("="*60)
print("VÍ DỤ MINH HỌA RANDOM FOREST")
print("Bài toán: Dự đoán sinh viên đậu/rớt môn Khai phá dữ liệu")
print("="*60)

print("\n📊 DỮ LIỆU GỐC:")
print(df[['Gio_hoc', 'Bai_tap', 'Diem_danh', 'Kinh_nghiem_lap_trinh', 'Ket_qua']])

# Chuẩn bị dữ liệu
X = df[['Gio_hoc', 'Bai_tap', 'Diem_danh', 'Kinh_nghiem_lap_trinh']]
y = df['Ket_qua']

print(f"\n📈 PHÂN BỐ KẾT QUẢ:")
print(f"Đậu: {sum(y)} sinh viên ({sum(y)/len(y)*100:.1f}%)")
print(f"Rớt: {len(y)-sum(y)} sinh viên ({(len(y)-sum(y))/len(y)*100:.1f}%)")

# Tạo Random Forest với 3 cây để minh họa
print("\n🌲 XÂY DỰNG RANDOM FOREST (3 cây):")
print("-" * 40)

rf = RandomForestClassifier(
    n_estimators=3,
    max_features=2,  # Chỉ xem xét 2 features mỗi split
    max_depth=3,
    random_state=42,
    bootstrap=True
)

rf.fit(X, y)

# Hiển thị thông tin từng cây
for i, tree in enumerate(rf.estimators_):
    print(f"\n🌳 CÂY {i+1}:")
    print(f"Features được chọn: {tree.feature_importances_}")
    
    # In cấu trúc cây (đơn giản)
    tree_rules = export_text(tree, feature_names=X.columns.tolist(), max_depth=2)
    print("Cấu trúc cây (2 level đầu):")
    print(tree_rules[:300] + "..." if len(tree_rules) > 300 else tree_rules)

# Dự đoán cho sinh viên mới
print("\n🎯 DỰ ĐOÁN CHO SINH VIÊN MỚI:")
print("-" * 40)

# Sinh viên X: 24 giờ học, 8 bài tập, 87% điểm danh, có kinh nghiệm
sinh_vien_moi_data = [[24, 8, 87, 1]]
# Tạo DataFrame với feature names để tránh warning
sinh_vien_moi = pd.DataFrame(sinh_vien_moi_data, columns=X.columns)

# Dự đoán từng cây
print("Dự đoán từng cây:")
for i, tree in enumerate(rf.estimators_):
    pred = tree.predict(sinh_vien_moi)[0]
    proba = tree.predict_proba(sinh_vien_moi)[0][1]  # Probability của class 1 (Đậu)
    result = "Đậu" if pred == 1 else "Rớt"
    print(f"  Cây {i+1}: {result} (P(Đậu) = {proba:.3f})")

# Kết quả ensemble
final_pred = rf.predict(sinh_vien_moi)[0]
final_proba = rf.predict_proba(sinh_vien_moi)[0][1]
final_result = "Đậu" if final_pred == 1 else "Rớt"

print(f"\n🏆 KẾT QUẢ CUỐI CÙNG (Ensemble):")
print(f"   Dự đoán: {final_result}")
print(f"   Xác suất đậu: {final_proba:.3f} ({final_proba*100:.1f}%)")
print(f"   Độ tin cậy: {'Cao' if final_proba > 0.7 or final_proba < 0.3 else 'Trung bình'}")

# Feature importance
print(f"\n📊 MỨC ĐỘ QUAN TRỌNG CỦA CÁC YẾU TỐ:")
print("-" * 45)
feature_importance = rf.feature_importances_
feature_names = X.columns.tolist()

for i, (name, importance) in enumerate(zip(feature_names, feature_importance)):
    print(f"{i+1}. {name}: {importance:.3f} ({importance*100:.1f}%)")

# Tính toán Information Gain thủ công cho minh họa
print(f"\n🧮 TÍNH TOÁN CHI TIẾT INFORMATION GAIN:")
print("-" * 50)

def calculate_entropy(labels):
    """Tính entropy"""
    unique_labels, counts = np.unique(labels, return_counts=True)
    entropy = 0
    for count in counts:
        p = count / len(labels)
        if p > 0:
            entropy -= p * np.log2(p)
    return entropy

def calculate_information_gain(X_col, y, threshold=None):
    """Tính Information Gain cho một feature"""
    if threshold is None:
        threshold = np.median(X_col)
    
    # Entropy ban đầu
    initial_entropy = calculate_entropy(y)
    
    # Split data
    left_mask = X_col <= threshold
    right_mask = X_col > threshold
    
    if sum(left_mask) == 0 or sum(right_mask) == 0:
        return 0
    
    # Weighted entropy sau split
    left_entropy = calculate_entropy(y[left_mask])
    right_entropy = calculate_entropy(y[right_mask])
    
    weighted_entropy = (sum(left_mask)/len(y)) * left_entropy + (sum(right_mask)/len(y)) * right_entropy
    
    return initial_entropy - weighted_entropy

# Tính IG cho từng feature
print("Information Gain cho từng feature:")
for col in X.columns:
    if col == 'Kinh_nghiem_lap_trinh':
        # Binary feature
        ig = calculate_information_gain(X[col], y, threshold=0.5)
    else:
        ig = calculate_information_gain(X[col], y)
    print(f"  {col}: {ig:.4f}")

# Visualization
plt.figure(figsize=(15, 10))

# 1. Feature Importance
plt.subplot(2, 3, 1)
bars = plt.bar(feature_names, feature_importance, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
plt.title('Mức độ quan trọng các yếu tố')
plt.ylabel('Importance')
plt.xticks(rotation=45, ha='right')
for bar, imp in zip(bars, feature_importance):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{imp:.3f}', ha='center', va='bottom')

# 2. Phân bố giờ học theo kết quả
plt.subplot(2, 3, 2)
dau_gio = df[df['Ket_qua']==1]['Gio_hoc']
rot_gio = df[df['Ket_qua']==0]['Gio_hoc']
plt.hist([dau_gio, rot_gio], bins=5, alpha=0.7, label=['Đậu', 'Rớt'], color=['green', 'red'])
plt.title('Phân bố giờ học')
plt.xlabel('Giờ học')
plt.ylabel('Số sinh viên')
plt.legend()

# 3. Phân bố bài tập theo kết quả
plt.subplot(2, 3, 3)
dau_bt = df[df['Ket_qua']==1]['Bai_tap']
rot_bt = df[df['Ket_qua']==0]['Bai_tap']
plt.hist([dau_bt, rot_bt], bins=5, alpha=0.7, label=['Đậu', 'Rớt'], color=['green', 'red'])
plt.title('Phân bố số bài tập')
plt.xlabel('Số bài tập')
plt.ylabel('Số sinh viên')
plt.legend()

# 4. Tỷ lệ đậu theo kinh nghiệm lập trình
plt.subplot(2, 3, 4)
co_kn = df[df['Kinh_nghiem_lap_trinh']==1]['Ket_qua'].mean()
khong_kn = df[df['Kinh_nghiem_lap_trinh']==0]['Ket_qua'].mean()
plt.bar(['Có kinh nghiệm', 'Không kinh nghiệm'], [co_kn, khong_kn], 
        color=['darkgreen', 'darkred'], alpha=0.7)
plt.title('Tỷ lệ đậu theo kinh nghiệm')
plt.ylabel('Tỷ lệ đậu')
for i, v in enumerate([co_kn, khong_kn]):
    plt.text(i, v + 0.02, f'{v:.1%}', ha='center', va='bottom')

# 5. Scatter plot: Giờ học vs Bài tập
plt.subplot(2, 3, 5)
colors = ['green' if x == 1 else 'red' for x in df['Ket_qua']]
plt.scatter(df['Gio_hoc'], df['Bai_tap'], c=colors, alpha=0.7, s=100)
plt.xlabel('Giờ học')
plt.ylabel('Bài tập')
plt.title('Giờ học vs Bài tập')
# Thêm điểm sinh viên mới
plt.scatter(24, 8, c='blue', s=200, marker='*', label='Sinh viên mới')
plt.legend()

# 6. Prediction confidence
plt.subplot(2, 3, 6)
# Tạo dữ liệu giả cho visualization
test_students = []
predictions = []
confidences = []

for gio in range(10, 35, 5):
    for bt in range(3, 11, 2):
        for kn in [0, 1]:
            test_point_data = [[gio, bt, 85, kn]]  # Điểm danh cố định 85%
            test_point_df = pd.DataFrame(test_point_data, columns=X.columns)
            pred_proba = rf.predict_proba(test_point_df)[0][1]
            test_students.append(test_point_data[0])
            predictions.append(pred_proba > 0.5)
            confidences.append(pred_proba)

# Vẽ decision boundary (simplified)
x_range = range(10, 35)
y_range = range(3, 11)
confidence_colors = ['red' if c < 0.3 else 'yellow' if c < 0.7 else 'green' for c in confidences]

plt.title('Độ tin cậy dự đoán')
plt.xlabel('Mẫu test')
plt.ylabel('Confidence')
plt.bar(range(len(confidences)), confidences, color=confidence_colors, alpha=0.7)
plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('random_forest_demo.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n💡 NHẬN XÉT:")
print("="*40)
print("✅ Random Forest kết hợp 3 cây với các góc nhìn khác nhau")
print("✅ Mỗi cây chỉ xem xét 2/4 features → tăng tính đa dạng")
print("✅ Kết quả ensemble tin cậy hơn từng cây riêng lẻ")
print("✅ Feature importance cho thấy 'Kinh nghiệm lập trình' quan trọng nhất")
print("✅ Probability output giúp đánh giá độ tin cậy của dự đoán")

print(f"\n🎓 KẾT LUẬN:")
print("="*30)
print(f"Sinh viên với profile [24h học, 8 bài tập, 87% điểm danh, có kinh nghiệm]")
print(f"có {final_proba*100:.1f}% khả năng ĐẬU môn Khai phá dữ liệu")
print("Đây là ví dụ minh họa thuật toán Random Forest hoàn toàn tự tạo!")
