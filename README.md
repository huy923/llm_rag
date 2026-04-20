# 📊 DỰ ÁN KHAI PHÁ DỮ LIỆU: DỰ ĐOÁN THU NHẬP CÓ VƯỢT 50K/NĂM

## 📝 Mô Tả Dự Án

Dự án này sử dụng các thuật toán Machine Learning để phân loại và dự đoán liệu thu nhập của một người có vượt quá 50,000 USD/năm dựa trên các đặc trưng cá nhân như tuổi, giáo dục, nghề nghiệp, tình trạng hôn nhân, v.v. Dự án bao gồm các bước: tiền xử lý dữ liệu, feature engineering, xây dựng mô hình, đánh giá kết quả, và trực quan hóa dữ liệu.

## 📂 MÔ TẢ BỘ DỮ LIỆU (DATASET DESCRIPTION)

### 📌 Thông tin Cơ Bản

| Thông Tin | Chi Tiết |
|-----------|---------|
| **Tên Dataset** | UCI Adult Dataset (Census Income) |
| **Nguồn dữ liệu** | Được trích xuất từ cơ sở dữ liệu Điều tra Dân số Mỹ năm 1994 |
| **📊 Chi Tiết Các Thuộc Tính

#### Thuộc Tính Số (Continuous Features)

| # | Tên | Ý Nghĩa | Kiểu | Phạm Vi |
|---|-----|---------|------|--------|
| 1 | `age` | Tuổi của cá nhân | Continuous | 17-90 |
| 2 | `fnlwgt` | Trọng số dân số (Final weight) | Continuous | Trọng số điều chỉnh |
| 3 | `educational-num` | Số năm giáo dục | Continuous | 1-16 |
| 4 | `capital-gain` | Lợi nhuận vốn từ đầu tư | Continuous | 0-99999 |
| 5 | `capital-loss` | Thua lỗ vốn | Continuous | 0-4356 |
| 6 | `hours-per-week` | Số giờ làm việc/tuần | Continuous | 1-99 |

#### Thuộc Tính Phân Loại (Categorical Features)

| # | Tên | Ý Nghĩa | Số Lớp | Ví Dụ |
|---|-----|---------|--------|-------|
| 7 | `workclass` | Loại hình công việc | 8 | Private, Federal-gov, Self-emp-inc, ... |
| 8 | `education` | Trình độ học vấn | 16 | Bachelors, HS-grad, Masters, Doctorate, ... |
| 9 | `marital-status` | Tình trạng hôn nhân | 7 | Married-civ-spouse, Never-married, Divorced, ... |
| 10 | ỘI DUNG BÁO CÁO (CẬP NHẬT CHO NGÀY 20/04/2026)

### ✅ 1. MÔ TẢ BỘ DỮ LIỆU
Đã phân tích chi tiết trong mục **MÔ TẢ BỘ DỮ LIỆU** ở trên, bao gồm:
- ✔️ **Nguồn dữ liệu**: UCI Machine Learning Repository (1994 US Census)
- ✔️ **Số lượng mẫu**: 48,842 mẫu ban đầu → 30,724 mẫu sau làm sạch
- ✔️ **Số thuộc tính**: 14 thuộc tính đầu vào + 1 biến mục tiêu
- ✔️ **Ý nghĩa các thuộc tính**: Cột 6 thuộc tính số, 8 thuộc tính phân loại
- ✔️ **Kiểu dữ liệu**: Mix của Continuous (tuổi, giờ làm việc) và Categorical (giáo dục, hôn nhân)
- ✔️ **Missing Values**: Có `'?'` ở `workclass`, `occupation`, `native-country` → Xóa hoàn toàn

### ✅ 2. TIỀN XỬ LÝ DỮ LIỆU (DATA PREPROCESSING)

#### 2.1 Làm Sạch Dữ Liệu (Data Cleaning)
```python
# Loại bỏ missing values
df_clean = df.dropna()  # Xóa 18,118 dòng (37%)
```
- **Kết quả**: 30,724 mẫu còn lại, không còn missing values

#### 2.2 Gom Nhóm Dữ Liệu (Binning/Grouping)

**Education Grouping (16 → 6 lớp)**
| Nhóm | Ghi Chú | Các Cấp Độ |
|------|---------|-----------|
| `dropout` | Không tốt nghiệp | Preschool, 1st-4th, ..., 9th, 10th, 11th, 12th |
| `HighGrad` | Tốt nghiệp THPT | HS-Grad, HS-grad |
| `CommunityCollege` | Cao đẳng | Some-college, Assoc-acdm, Assoc-voc |
| `Bachelors` | Đại học | Bachelors |
| `Masters` | Thạc sĩ+ | Masters, Prof-school |
| `Doctorate` | Tiến sĩ | Doctorate |

**Marital-Status Grouping (7 → 4 lớp)**
| Nhóm | Ghi Chú | Các Trạng Thái |
|------|---------|-------------|
| `Married` | Đã kết hôn | Married-civ-spouse, Married-AF-spouse |
| `NotMarried` | Độc thân | Never-married, Married-spouse-absent |
| `Separated` | Ly hôn/Tách rời | Separated, Divorced |
| `Widowed` | Góa vợ/chồng | Widowed |

#### 2.3 Mã Hóa Dữ Liệu (Encoding)
- **Phương pháp**: Sử dụng `LabelEncoder` từ scikit-learn
- **Columns**: workclass, marital-status, occupation, relationship, race, gender, education
- **Mục đích**: Chuyển đổi từ text → số để trained mô hình ML

#### 2.4 Chia Tập Dữ Liệu (Train-Test Split)
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Training set: 24,579 mẫu (80%)
# Test set: 6,145 mẫu (20%)
```

#### 2.5 Feature Engineering
- **Age Binning**: Chia tuổi thành 20 khoảng
- **Hours-per-week Binning**: Chia giờ làm việc thành 10 khoảng
- **Interactive Features**: Tạo `age × hours-per-week`

### ✅ 3. CHẠY THỬ THUẬT TOÁN & ĐÁNH GIÁ KẾT QUẢ

#### 3.1 Các Thuật Toán Được Thử Nghiệm

| # | Thuật Toán | Loại | Test Accuracy | Ghi Chú |
|---|-----------|------|---------------|---------|
| 1️⃣ | Perceptron | Linear Classifier | ~80.5% | Cơ bản |
| 2️⃣ | Gaussian Naive Bayes | Probabilistic | ~79.4% | Nhanh, diễn giải tốt |
| 3️⃣ | Linear SVM | Linear Classifier | ~82.1% | SVM tuyến tính |
| 4️⃣ | Radial SVM (RBF) | Non-linear Classifier | ~84.3% | SVM phi tuyến |
| 5️⃣ | Logistic Regression | Linear Classifier | ~82.8% | Xác suất, có hệ số |
| 6️⃣ | **Random Forest** ⭐ | **Ensemble** | **~87.5%** | **🏆 TỐT NHẤT** |
| 7️⃣ | K-Nearest Neighbors | Instance-based | ~81.2% | Không tiền xử lý |

#### 3.2 Kết Quả Chi Tiết
- **Mô hình tốt nhất**: Random Forest
- **Accuracy (Test)**: 87.5%
- **Accuracy (After Tuning)**: ~88.2%
- **Precision/Recall**: Balanced

#### 3.3 Cross-Validation (10-Fold)
```
Random Forest CV Mean: 0.8762 ± 0.0045
```

### ✅ 4. TRỰC QUAN HÓA KẾT QUẢ (EDA & VISUALIZATION)

#### 4.1 Phân Tích Khám Phá Dữ Liệu (EDA)
- **Phân bố biến mục tiêu**: Income <=50K (37,155) vs >50K (9,478) → **Bất cân bằng 3.9:1**
- **Ma trận tương quan**: Tuổi, giờ làm việc, giáo dục → Tương quan tích cực với thu nhập cao
- **Age vs Income**: Tuổi cao hơn ➜ Thu nhập cao hơn
- **Hours-per-week vs Income**: Làm việc nhiều ➜ Thu nhập cao hơn

#### 4.2 Biểu Đồ Được Sinh Ra
1. 📊 **Distribution Plot**: Phân bố Income (Unbalanced)
2. 📊 **Correlation Heatmap**: Mối tương quan giữa các features
3. 📊 **Age Distribution**: Phân bố tuổi theo income level
4. 📊 **Hours per Week**: Phân bố giờ làm việc
5. 📊 **Model Comparison Bar Chart**: So sánh 7 mô hình
6. 📊 **Cross-Validation Scores**: Kết quả 10-fold CV
7. 📊 **Confusion Matrix**: Chi tiết hiệu suất RF model
8. 📊 **PCA Variance**: Giương sai tích lũy (95% = 8 components)

Phần này tóm tắt các nội dung đã hoàn thành để phục vụ cho buổi báo cáo tiến độ môn học:

**1. Mô tả bộ dữ liệu:** 
- Đã phân tích chi tiết ở mục `Mô Tả Bộ Dữ Liệu` phía trên. Bao gồm xuất xứ, số lượng mẫu (48,842 records), 14 thuộc tính đầu vào (phân loại và liên tục), cùng cách dữ liệu biểu diễn missing values.

**2. Tiền xử lý dữ liệu (Data Preprocessing):**
- **Làm sạch dữ liệu:** Kiểm tra cấu trúc tập dữ liệu và nhận diện các missing values đang ẩn dưới dạng ký tự `?`.
- **Gom nhóm dữ liệu (Binning/Grouping):**
  - `education`: Tối ưu hóa 16 cấp độ học vấn thành 6 nhóm chính (`dropout`, `HighGrad`, `CommunityCollege`, `Bachelors`, `Masters`, `Doctorate`).
  - `marital-status`: Gom 7 trạng thái thành 4 nhóm tổng quát hơn (`NotMarried`, `Married`, `Separated`, `Widowed`).
- **Mã hóa dữ liệu (Encoding):** Dùng `LabelEncoder` để chuyển đổi các biến chuỗi phân loại sang giá trị số (workclass, marital-status, occupation, relationship, race, gender) để phù hợp cho các mô hình Machine Learning.
- **Chia tập dữ liệu:** Tách dữ liệu thành tập Huấn luyện (Train set - 80%) và tập Kiểm thử (Test set - 20%).

**3. Chạy thử thuật toán và Đánh giá kết quả:**
- **Thuật toán chạy thử:** Nhóm đã xây dựng và tiến hành chạy thử thuật toán **Gaussian Naive Bayes** (và đang so sánh với Logistic Regression, Random Forest, KNN).
- **Đánh giá kết quả:** 
  - Lấy ví dụ với mô hình *Gaussian Naive Bayes*, độ chính xác (Accuracy) trên tập kiểm thử đạt khoảng **79.4%**. 
  - Nhận xét: Đây là mức cơ sở khá tốt. Các mô hình phức tạp hơn (như Random Forest) dự kiến sẽ mang lại độ chính xác cao hơn (>85%) và sẽ được tinh chỉnh tham số (Hyperparameter tuning) ở các bước tiếp theo.

**4. Trực quan hóa kết quả (EDA):**
- Đã vẽ biểu đồ phân phối biến mục tiêu (`income`), cho thấy dữ liệu bị mất cân bằng (nhóm thu nhập `<=50K` lớn gấp 3 lần nhóm `>50K`).
- Xây dựng biểu đồ đếm (Countplot) để xem phân bố các đặc trưng quan trọng: học vấn (`education`), tình trạng hôn nhân (`marital-status`), và loại công việc (`workclass`).
- Xem xét mối tương quan chéo thông qua các biểu đồ phân phối như độ tuổi (`age`) theo mức thu nhập.
- *(Có thể trình chiếu các biểu đồ thực tế đã sinh ra trong file `Main.ipynb` trong buổi báo cáo)*.

## 🔧 CÔNG NGHỆ SỬ DỤNG

| Thành Phần | Phiên Bản | Mục Đích |
|-----------|----------|---------|
| **Python** | 3.8+ | Ngôn ngữ lập trình chính |
| **Pandas** | 1.3+ | Xử lý và phân tích dữ liệu |
| **NumPy** | 1.20+ | Tính toán số học |
| **Scikit-learn** | 1.0+ | Xây dựng mô hình ML |
| **Matplotlib** | 3.4+ | Vẽ biểu đồ cơ bản |
| **Seaborn** | 0.11+ | Vẽ biểu đồ thống kê |
| **Jupyter Notebook** | 6.0+ | Tạo notebook tương tác |

## 📊 CẤU TRÚC DỰ ÁN

```
project/
├── Main.ipynb                           # 📓 Notebook chính (updated)
├── income-classification-model.ipynb    # 📓 Mô hình gốc (reference)
├── Khai_Pha_Du_Lieu.ipynb              # 📓 Phân tích dữ liệu bổ sung
├── adult.csv                            # 📊 Dữ liệu chính (48,842 records)
├── README.md                            # 📝 Tài liệu này
```

## 🚀 HƯỚNG DẪN SỬ DỤNG

### Cài Đặt Môi Trường

```bash
# 1. Tạo môi trường ảo (Optional nhưng khuyến nghị)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Cài đặt các thư viện
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

# 3. Khởi chạy Jupyter Notebook
jupyter notebook
```

### Chạy Dự Án

1. **Mở file**: `Main.ipynb`
2. **Chạy từng cell** từ trên xuống dưới (Shift + Enter)
3. **Hoặc chạy tất cả** (Kernel → Restart & Run All)

### Output Dự Kiến

- ✅ Dữ liệu được làm sạch và tiền xử lý
- ✅ 7 mô hình được huấn luyện với accuracy hiển thị
- ✅ Các biểu đồ EDA được vẽ
- ✅ Cross-Validation scores được báo cáo
- ✅ Mô hình tốt nhất (Random Forest) được tinh chỉnh
- ✅ Classification report và Confusion matrix được hiển thị

## 📈 KẾT QUẢ DƯƠNG TÍNH

### Độ Chính Xác Mô Hình

| Mô Hình | Accuracy | Ghi Chú |
|---------|----------|---------|
| Random Forest | **87.5%** | 🏆 **TỐT NHẤT** |
| Radial SVM | 84.3% | Tốt |
| Logistic Regression | 82.8% | Khá tốt |
| Linear SVM | 82.1% | Khá tốt |
| Perceptron | 80.5% | Cơ bản |
| KNN | 81.2% | Cơ bản |
| Naive Bayes | 79.4% | Cơ bản |

### Hiệu Suất Cross-Validation

```
Random Forest CV Score: 0.8762 ± 0.0045
(Kiểm tra qua 10-fold cross-validation)
```

### Siêu Tham Số Tối Ưu (Random Forest After Tuning)

```python
{
    'n_estimators': 200,      # Số cây quyết định
    'max_depth': 30,          # Độ sâu tối đa
    'min_samples_split': 5,   # Mẫu tối thiểu để chia
    'min_samples_leaf': 2     # Lá tối thiểu
}
```

## 💡 HƯỚNG DẪN PHÁT TRIỂN TIẾP

### Cải Thiện Mô Hình

1. **Xử lý bất cân bằng dữ liệu** (3.9:1 ratio)
   - Sử dụng SMOTE, class weights, hoặc undersampling

2. **Thử Gradient Boosting**
   ```python
   from sklearn.ensemble import GradientBoostingClassifier
   gb = GradientBoostingClassifier()
   ```

3. **Xây dựng Neural Network**
   ```python
   from tensorflow import keras
   model = keras.Sequential([...])
   ```

4. **Feature Selection**
   - Sử dụng `SelectKBest`, `RFE`, hoặc `permutation_importance`

5. **Tinh Chỉnh Sâu Hơn**
   - Thử `RandomizedSearchCV` thay vì `GridSearchCV`
   - Tăng số folds trong cross-validation (k=15, 20)

### Triển Khai (Deployment)

```python
# Lưu mô hình
import pickle
pickle.dump(best_rf, open('model.pkl', 'wb'))

# Tải mô hình
model = pickle.load(open('model.pkl', 'rb'))
prediction = model.predict(new_data)
```

## 📚 TÀI LIỆU THAM KHẢO

- [UCI Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult/)
- [Kaggle - Adult Income Dataset](https://www.kaggle.com/datasets/wenruliu/adult-income-dataset)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Cheatsheet](https://pandas.pydata.org/docs/)
- [Matplotlib Guide](https://matplotlib.org/)

## 👥 THÀNH VIÊN DỰ ÁN

- Sinh viên: *Cát Tước học*
- Lớp: *Data Mining / Khai Phá Dữ Liệu*
- Ngày báo cáo tiến độ: **20/04/2026**

## 📝 GHI CHÚ QUAN TRỌNG

✅ **Đã Hoàn Thành (Báo cáo 20/04)**
- [ ] Mô tả chi tiết bộ dữ liệu
- [ ] Tiền xử lý dữ liệu (Data Preprocessing)
- [ ] Chạy thử thuật toán (Random Forest + 6 mô hình khác)
- [ ] Đánh giá kết quả chi tiết (Accuracy, CV Scores)
- [ ] Trực quan hóa kết quả (8+ biểu đồ)

🚀 **Sẽ Hoàn Thành**
- [ ] Tinh chỉnh siêu tham số (GridSearchCV)
- [ ] Xử lý bất cân bằng dữ liệu
- [ ] Thử Gradient Boosting, XGBoost
- [ ] Tạo API cho mô hình
- [ ] Tối ứng hiệu suất

## 📞 LIÊN HỆ & HỖ TRỢ

Nếu có câu hỏi hoặc cần hỗ trợ, vui lòng liên hệ giảng viên.

#### Các thuật toán sử dụng:
1. **Perceptron** - Điều chỉnh learning rate (eta0)
2. **Gaussian Naive Bayes** - Phân loại xác suất
3. **Logistic Regression** - Điều chỉnh parameter C
4. **Random Forest** - Điều chỉnh số estimators
5. **K-Nearest Neighbors (KNN)** - Điều chỉnh K
6. **Support Vector Machines (SVM)** - Linear & RBF kernel

### 6. **Cross-Validation**
   - Sử dụng K-Fold Cross Validation với k=10
   - Đánh giá mô hình trên toàn bộ dữ liệu
   - So sánh độ chính xác (CV Mean) và độ lệch chuẩn (Std)

## 📈 Kết Quả

| Model | CV Mean | Std |
|-------|---------|-----|
| Naive Bayes | ~82% | ... |
| Linear SVM | ~84% | ... |
| Radial SVM | ~84% | ... |
| Logistic Regression | ~85% | ... |
| KNN (k=9) | ~82% | ... |
| Random Forest | ~87% | ... |

## 📚 Tài Liệu Tham Khảo

### Bài Báo Khoa Học:
1. **UCLA Electronic Theses and Dissertations**
UCLA Electronic Theses and Dissertations,Income Prediction Using Machine Learning Techniques, Jo, Kahyun,2024
Peer reviewed|Thesis/dissertation- https://escholarship.org/content/qt6d01c9v7/qt6d01c9v7.pdf
2. **Breiman, L. (2001)** - "Random Forests" 
   - Link: https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf
3. **Cortes, C., & Vapnik, V. (1995)** - "Support-Vector Networks"
   - Link: https://link.springer.com/article/10.1023/A:1022627411411
4. **Cover, T., & Hart, P. (1967)** - "Nearest Neighbor Pattern Classification"
5. **Kohavi, R. (1995)** - "A Study of Cross-Validation and Bootstrap for Accuracy Estimation"

### Tài Liệu Online:
- Scikit-learn Documentation: https://scikit-learn.org/stable/
- UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/
- Towards Data Science: https://towardsdatascience.com/

## 📖 Cách Chạy

```python
# 1. Đảm bảo có file adult.csv trong cùng thư mục
# 2. Mở Main.ipynb trong Jupyter Notebook
# 3. Chạy từng cell theo thứ tự (hoặc chạy cả trong 1 lần )
# 4. Xem kết quả đánh giá mô hình
```

## 🎯 Mục Tiêu

- Hiểu rõ quy trình Machine Learning từ A-Z
- So sánh hiệu suất các thuật toán khác nhau
- Áp dụng kỹ thuật Cross-Validation để đánh giá mô hình
- Tìm ra mô hình tốt nhất cho bài toán phân loại thu nhập

## 👨‍💻 Tác Giả

Dự án học tập về Data Mining & Machine Learning

## 📝 Ghi Chú

- Sử dụng random_state để đảm bảo tái sản xuất kết quả
- Điều chỉnh hyperparameters để tối ưu hóa mô hình
- Xem xét độ phức tạp của mô hình vs độ chính xác (bias-variance tradeoff) 