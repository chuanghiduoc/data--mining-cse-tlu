#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GIẢI BÀI TẬP RANDOM FOREST TỪNG BƯỚC CHI TIẾT
Giống như giảng toán trên bảng
"""

import pandas as pd
import numpy as np
import math

class RandomForestStepByStep:
    """Giải Random Forest từng bước như giảng toán"""
    
    def __init__(self):
        print("="*70)
        print("🎓 BÀI GIẢNG: GIẢI RANDOM FOREST TỪNG BƯỚC")
        print("="*70)
        
    def step1_prepare_data(self):
        """Bước 1: Chuẩn bị dữ liệu"""
        print("\n📚 BƯỚC 1: CHUẨN BỊ DỮ LIỆU")
        print("-" * 50)
        
        # Dữ liệu gốc
        data = {
            'ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'Gio_hoc': [20, 15, 25, 12, 30, 18, 10, 22, 28, 16],
            'Bai_tap': [8, 6, 10, 4, 9, 7, 3, 8, 10, 5],
            'Diem_danh': [85, 90, 95, 70, 88, 92, 65, 80, 96, 75],
            'Kinh_nghiem': ['Có', 'Không', 'Có', 'Không', 'Có', 'Có', 'Không', 'Không', 'Có', 'Không'],
            'Ket_qua': ['Đậu', 'Rớt', 'Đậu', 'Rớt', 'Đậu', 'Đậu', 'Rớt', 'Rớt', 'Đậu', 'Rớt']
        }
        
        self.df = pd.DataFrame(data)
        print("Dữ liệu gốc:")
        print(self.df)
        
        # Mã hóa
        print("\n🔢 Mã hóa dữ liệu:")
        self.df['Kinh_nghiem_encoded'] = self.df['Kinh_nghiem'].map({'Có': 1, 'Không': 0})
        self.df['Ket_qua_encoded'] = self.df['Ket_qua'].map({'Đậu': 1, 'Rớt': 0})
        
        print("Kinh_nghiem: 'Có' = 1, 'Không' = 0")
        print("Ket_qua: 'Đậu' = 1, 'Rớt' = 0")
        
        print("\nDữ liệu sau mã hóa:")
        encoded_df = self.df[['ID', 'Gio_hoc', 'Bai_tap', 'Diem_danh', 'Kinh_nghiem_encoded', 'Ket_qua_encoded']]
        encoded_df.columns = ['ID', 'Gio_hoc', 'Bai_tap', 'Diem_danh', 'Kinh_nghiem', 'Ket_qua']
        print(encoded_df)
        
        return encoded_df
    
    def step2_create_bootstrap_sample(self, df):
        """Bước 2: Tạo Bootstrap Sample"""
        print("\n🌲 BƯỚC 2: TẠO BOOTSTRAP SAMPLES")
        print("-" * 50)
        
        print("📖 Lý thuyết Bootstrap:")
        print("- Lấy mẫu có hoàn lại từ tập gốc")
        print("- Kích thước = tập gốc (10 mẫu)")
        print("- Mỗi mẫu có thể xuất hiện nhiều lần")
        
        # Bootstrap Sample 1 (giả sử random được)
        bootstrap_indices = [1, 3, 5, 6, 9, 2, 8, 5, 1, 10]  # indices (1-based)
        bootstrap_indices_0 = [i-1 for i in bootstrap_indices]  # convert to 0-based
        
        print(f"\n🎲 Giả sử random chọn được indices: {bootstrap_indices}")
        
        bootstrap_sample = df.iloc[bootstrap_indices_0].copy()
        bootstrap_sample.reset_index(drop=True, inplace=True)
        
        print("\nBootstrap Sample 1:")
        print(bootstrap_sample)
        
        # Phân tích phân bố
        dau_count = sum(bootstrap_sample['Ket_qua'])
        rot_count = len(bootstrap_sample) - dau_count
        
        print(f"\n📊 Phân bố trong Sample 1:")
        print(f"- Đậu (1): {dau_count} mẫu")
        print(f"- Rớt (0): {rot_count} mẫu")
        
        return bootstrap_sample
    
    def calculate_entropy(self, labels):
        """Tính entropy"""
        if len(labels) == 0:
            return 0
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        
        entropy = 0
        for p in probabilities:
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    def step3_build_tree1(self, sample):
        """Bước 3: Xây dựng cây quyết định 1"""
        print("\n📊 BƯỚC 3: XÂY DỰNG CÂY QUYẾT ĐỊNH 1")
        print("-" * 50)
        
        print("⚙️ Thiết lập tham số:")
        print("- max_features = 2: Mỗi nút chỉ xem xét 2/4 đặc trưng")
        print("- Tổng features: {Gio_hoc, Bai_tap, Diem_danh, Kinh_nghiem}")
        
        print("\n🎯 TẠI NÚT GỐC:")
        print("Random chọn 2 features: {Gio_hoc, Kinh_nghiem}")
        
        # Tính Entropy ban đầu
        print("\n📐 TÍNH ENTROPY BAN ĐẦU:")
        labels = sample['Ket_qua'].values
        initial_entropy = self.calculate_entropy(labels)
        
        total_samples = len(labels)
        dau_count = sum(labels)
        rot_count = total_samples - dau_count
        
        print(f"Tổng mẫu: {total_samples}")
        print(f"Đậu: {dau_count} mẫu → p₁ = {dau_count}/{total_samples} = {dau_count/total_samples:.3f}")
        print(f"Rớt: {rot_count} mẫu → p₂ = {rot_count}/{total_samples} = {rot_count/total_samples:.3f}")
        
        print(f"\nCông thức: Entropy(S) = -Σ pᵢ × log₂(pᵢ)")
        print(f"Entropy(S) = -({dau_count/total_samples:.3f} × log₂({dau_count/total_samples:.3f})) - ({rot_count/total_samples:.3f} × log₂({rot_count/total_samples:.3f}))")
        
        if dau_count > 0:
            term1 = (dau_count/total_samples) * math.log2(dau_count/total_samples)
        else:
            term1 = 0
            
        if rot_count > 0:
            term2 = (rot_count/total_samples) * math.log2(rot_count/total_samples)
        else:
            term2 = 0
            
        print(f"         = -({term1:.3f}) - ({term2:.3f})")
        print(f"         = {-term1:.3f} + {-term2:.3f}")
        print(f"         = {initial_entropy:.3f}")
        
        # Tính Information Gain cho Kinh_nghiem
        print("\n🧮 TÍNH INFORMATION GAIN CHO KINH_NGHIEM:")
        
        # Split theo Kinh_nghiem
        co_mask = sample['Kinh_nghiem'] == 1
        khong_mask = sample['Kinh_nghiem'] == 0
        
        co_labels = sample[co_mask]['Ket_qua'].values
        khong_labels = sample[khong_mask]['Ket_qua'].values
        
        print(f"\nKinh_nghiem = 1 (Có):")
        print(f"- Mẫu: {list(sample[co_mask]['ID'].values)} = {len(co_labels)} mẫu")
        print(f"- Đậu: {sum(co_labels)}, Rớt: {len(co_labels) - sum(co_labels)}")
        
        co_entropy = self.calculate_entropy(co_labels)
        print(f"- Entropy = {co_entropy:.3f} {'(thuần khiết)' if co_entropy == 0 else ''}")
        
        print(f"\nKinh_nghiem = 0 (Không):")
        print(f"- Mẫu: {list(sample[khong_mask]['ID'].values)} = {len(khong_labels)} mẫu")
        print(f"- Đậu: {sum(khong_labels)}, Rớt: {len(khong_labels) - sum(khong_labels)}")
        
        khong_entropy = self.calculate_entropy(khong_labels)
        print(f"- Entropy = {khong_entropy:.3f} {'(thuần khiết)' if khong_entropy == 0 else ''}")
        
        # Information Gain
        weighted_entropy = (len(co_labels)/total_samples) * co_entropy + (len(khong_labels)/total_samples) * khong_entropy
        ig_kinh_nghiem = initial_entropy - weighted_entropy
        
        print(f"\nInformation Gain:")
        print(f"IG = Entropy_ban_đầu - Σ(|Sᵥ|/|S|) × Entropy(Sᵥ)")
        print(f"   = {initial_entropy:.3f} - ({len(co_labels)}/{total_samples} × {co_entropy:.3f} + {len(khong_labels)}/{total_samples} × {khong_entropy:.3f})")
        print(f"   = {initial_entropy:.3f} - {weighted_entropy:.3f}")
        print(f"   = {ig_kinh_nghiem:.3f}")
        
        # Tính Information Gain cho Gio_hoc (đơn giản hóa)
        print("\n🧮 TÍNH INFORMATION GAIN CHO GIO_HOC:")
        
        threshold = sample['Gio_hoc'].median()
        print(f"Chọn threshold = median = {threshold}")
        
        le_mask = sample['Gio_hoc'] <= threshold
        gt_mask = sample['Gio_hoc'] > threshold
        
        le_labels = sample[le_mask]['Ket_qua'].values
        gt_labels = sample[gt_mask]['Ket_qua'].values
        
        print(f"\nGio_hoc ≤ {threshold}:")
        print(f"- Mẫu: {len(le_labels)} mẫu, Đậu: {sum(le_labels)}, Rớt: {len(le_labels) - sum(le_labels)}")
        
        le_entropy = self.calculate_entropy(le_labels)
        print(f"- Entropy = {le_entropy:.3f}")
        
        print(f"\nGio_hoc > {threshold}:")
        print(f"- Mẫu: {len(gt_labels)} mẫu, Đậu: {sum(gt_labels)}, Rớt: {len(gt_labels) - sum(gt_labels)}")
        
        gt_entropy = self.calculate_entropy(gt_labels)
        print(f"- Entropy = {gt_entropy:.3f}")
        
        weighted_entropy_gio = (len(le_labels)/total_samples) * le_entropy + (len(gt_labels)/total_samples) * gt_entropy
        ig_gio_hoc = initial_entropy - weighted_entropy_gio
        
        print(f"\nInformation Gain:")
        print(f"IG = {initial_entropy:.3f} - ({len(le_labels)}/{total_samples} × {le_entropy:.3f} + {len(gt_labels)}/{total_samples} × {gt_entropy:.3f})")
        print(f"   = {initial_entropy:.3f} - {weighted_entropy_gio:.3f}")
        print(f"   = {ig_gio_hoc:.3f}")
        
        # So sánh và chọn feature
        print(f"\n🏆 SO SÁNH VÀ CHỌN FEATURE:")
        print(f"IG(Kinh_nghiem) = {ig_kinh_nghiem:.3f}")
        print(f"IG(Gio_hoc) = {ig_gio_hoc:.3f}")
        
        if ig_kinh_nghiem > ig_gio_hoc:
            print(f"\n✅ Chọn Kinh_nghiem làm root (IG cao hơn)")
            chosen_feature = "Kinh_nghiem"
        else:
            print(f"\n✅ Chọn Gio_hoc làm root (IG cao hơn)")
            chosen_feature = "Gio_hoc"
        
        # Vẽ cây
        print(f"\n🌳 CÂY QUYẾT ĐỊNH 1 HOÀN CHỈNH:")
        print("```")
        if chosen_feature == "Kinh_nghiem":
            print("                Kinh_nghiem")
            print("                    /    \\")
            print("                 Có           Không")
            print("                /               \\")
            if co_entropy == 0:
                co_result = "Đậu" if sum(co_labels) > 0 else "Rớt"
                print(f"           {co_result} ({sum(co_labels)}/{len(co_labels)})         ", end="")
            if khong_entropy == 0:
                khong_result = "Đậu" if sum(khong_labels) > 0 else "Rớt"
                print(f"{khong_result} ({sum(khong_labels)}/{len(khong_labels)})")
            
            print(f"        P(Đậu) = {sum(co_labels)/len(co_labels):.1f}      P(Đậu) = {sum(khong_labels)/len(khong_labels):.1f}")
        
        print("```")
        
        return {
            'feature': chosen_feature,
            'co_prob': sum(co_labels)/len(co_labels) if len(co_labels) > 0 else 0,
            'khong_prob': sum(khong_labels)/len(khong_labels) if len(khong_labels) > 0 else 0
        }
    
    def step4_predict_new_student(self, tree1_info):
        """Bước 4: Dự đoán sinh viên mới"""
        print("\n🎯 BƯỚC 4: DỰ ĐOÁN SINH VIÊN MỚI")
        print("-" * 50)
        
        print("👤 Profile sinh viên X:")
        student_x = {
            'Gio_hoc': 24,
            'Bai_tap': 8,
            'Diem_danh': 87,
            'Kinh_nghiem': 1  # Có
        }
        
        for key, value in student_x.items():
            print(f"- {key}: {value}")
        
        print("\n🌳 DỰ ĐOÁN TỪNG CÂY:")
        
        # Cây 1
        print("\n📊 Cây 1:")
        if tree1_info['feature'] == 'Kinh_nghiem':
            if student_x['Kinh_nghiem'] == 1:
                prob1 = tree1_info['co_prob']
                result1 = "Đậu" if prob1 > 0.5 else "Rớt"
                print(f"Kinh_nghiem = 1 (Có) → đi nhánh trái → {result1}")
                print(f"P(Đậu) = {prob1:.3f}")
            else:
                prob1 = tree1_info['khong_prob']
                result1 = "Đậu" if prob1 > 0.5 else "Rớt"
                print(f"Kinh_nghiem = 0 (Không) → đi nhánh phải → {result1}")
                print(f"P(Đậu) = {prob1:.3f}")
        
        # Cây 2 (giả sử)
        print("\n📊 Cây 2 (giả sử):")
        print("Bai_tap = 8 ≥ 7 → đi nhánh trái")
        print("Diem_danh = 87% < 90% → đi nhánh phải → Rớt")
        prob2 = 0.25
        print(f"P(Đậu) = {prob2:.3f}")
        
        # Cây 3 (giả sử)
        print("\n📊 Cây 3 (giả sử):")
        print("Gio_hoc = 24 ≥ 18 → đi nhánh trái → Đậu")
        prob3 = 0.83
        print(f"P(Đậu) = {prob3:.3f}")
        
        # Ensemble
        print("\n🎯 KẾT HỢP KẾT QUẢ (ENSEMBLE):")
        
        print("\n📊 Phương pháp 1: Majority Voting")
        votes = []
        if prob1 > 0.5:
            votes.append("Đậu")
        else:
            votes.append("Rớt")
            
        if prob2 > 0.5:
            votes.append("Đậu") 
        else:
            votes.append("Rớt")
            
        if prob3 > 0.5:
            votes.append("Đậu")
        else:
            votes.append("Rớt")
        
        print(f"Cây 1: {votes[0]}")
        print(f"Cây 2: {votes[1]}")
        print(f"Cây 3: {votes[2]}")
        
        dau_votes = votes.count("Đậu")
        rot_votes = votes.count("Rớt")
        
        if dau_votes > rot_votes:
            majority_result = "Đậu"
        else:
            majority_result = "Rớt"
            
        print(f"\nKết quả: {majority_result} ({dau_votes}/3 phiếu)")
        
        print("\n📊 Phương pháp 2: Probability Averaging")
        avg_prob = (prob1 + prob2 + prob3) / 3
        print(f"P(Đậu) = ({prob1:.3f} + {prob2:.3f} + {prob3:.3f}) / 3 = {avg_prob:.3f}")
        print(f"→ {avg_prob*100:.1f}% khả năng đậu")
        
        return avg_prob
    
    def step5_conclusion(self, final_prob):
        """Bước 5: Kết luận"""
        print("\n📝 BƯỚC 5: KẾT LUẬN")
        print("-" * 50)
        
        print("🏆 ĐÁP ÁN CUỐI CÙNG:")
        print(f"Sinh viên X có {final_prob*100:.1f}% khả năng ĐẬU môn Khai phá dữ liệu")
        
        print("\n💡 GIẢI THÍCH KẾT QUẢ:")
        print("- Cây 1 dự đoán Đậu vì sinh viên có kinh nghiệm lập trình")
        print("- Cây 2 dự đoán Rớt vì điểm danh thấp (87% < 90%)")
        print("- Cây 3 dự đoán Đậu vì học nhiều giờ (24 ≥ 18)")
        
        print("\n✨ ưu điểm Random Forest thể hiện:")
        print("1. Đa dạng: 3 cây nhìn từ góc độ khác nhau")
        print("2. Robust: Không phụ thuộc vào 1 cây duy nhất")
        print("3. Confidence: Cho ra xác suất thay vì chỉ yes/no")
        
    def solve_complete_problem(self):
        """Giải toàn bộ bài toán"""
        print("🚀 BẮT ĐẦU GIẢI BÀI TOÁN RANDOM FOREST")
        
        # Bước 1: Chuẩn bị dữ liệu
        df = self.step1_prepare_data()
        
        # Bước 2: Tạo bootstrap sample
        bootstrap_sample = self.step2_create_bootstrap_sample(df)
        
        # Bước 3: Xây dựng cây 1
        tree1_info = self.step3_build_tree1(bootstrap_sample)
        
        # Bước 4: Dự đoán sinh viên mới
        final_prob = self.step4_predict_new_student(tree1_info)
        
        # Bước 5: Kết luận
        self.step5_conclusion(final_prob)
        
        print("\n🎉 HOÀN THÀNH BÀI GIẢNG!")
        print("📚 Hy vọng các bạn đã hiểu rõ cách giải Random Forest!")

if __name__ == "__main__":
    # Khởi tạo và giải bài toán
    teacher = RandomForestStepByStep()
    teacher.solve_complete_problem()

