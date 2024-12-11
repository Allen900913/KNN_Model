import csv
import math

class KNearestNeighbors:
    def __init__(self, k):
        self.k = k
        self.train_data = []
        self.train_labels = []
        self.mean_std = {}

    def calculate_mean_std(self):
        """計算所有數值特徵的均值和標準差"""
        num_features = len(self.train_data[0])
        for i in range(num_features):
            column = [row[i] for row in self.train_data]
            if column:
                mean_val = sum(column) / len(column)
                variance = sum((x - mean_val) ** 2 for x in column) / len(column)
                std_dev = math.sqrt(variance)
                self.mean_std[i] = (mean_val, std_dev)

    def normalize(self, row):
        """對所有數值特徵進行標準化"""
        num_features = len(row)
        for i in range(num_features):
            if i in self.mean_std:
                mean_val, std_dev = self.mean_std[i]
                if std_dev > 0:
                    row[i] = (row[i] - mean_val) / std_dev
                else:
                    row[i] = 0  # 如果標準差為 0，則該特徵值歸零
        return row

    def load_data(self, data_file, label_file):
        """載入數據並執行預處理和標準化"""
        with open(data_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                processed_row = self.preprocess(row)
                self.train_data.append(processed_row)

        # 計算特徵的均值和標準差
        self.calculate_mean_std()

        # 標準化數值特徵
        for i in range(len(self.train_data)):
            self.train_data[i] = self.normalize(self.train_data[i])

        # 載入標籤資料
        with open(label_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.train_labels.append(row[0])


    def preprocess(self, row):
        """對原始數據進行預處理"""
        processed_row = []
        
        # 性別
        processed_row.append(1 if row[0] == "Male" else 0)
        
        # 二元特徵
        binary_indices = [1, 2, 3, 5, 15]
        for i in binary_indices:
            processed_row.append(1 if row[i] == "Yes" else 0)
        
        # 數值特徵：tenure
        processed_row.append(float(row[4]))

        # 特殊二元特徵: MultipleLines
        if row[6] == "Yes":
            processed_row.append(2)
        elif row[6] == "No":
            processed_row.append(1)
        else:
            processed_row.append(0)
           
        # 類別特徵：網路服務
        internet_service = row[7]
        if internet_service == "Fiber optic":
            processed_row.append(2)
        elif internet_service == "DSL":
            processed_row.append(1)
        else:
            processed_row.append(0)
        
        # 網路相關服務：處理 No internet service
        internet_related_indices = [8, 9, 10, 11, 12, 13]
        for i in internet_related_indices:
            if row[i] == "Yes":
                processed_row.append(2)
            elif row[i] == "No":
                processed_row.append(1)
            else:  # "No internet service"
                processed_row.append(0)
        
        # 類別特徵：合約
        contract_type = row[14]
        if contract_type == "Month-to-month":
            processed_row.append(2)
        elif contract_type == "One year":
            processed_row.append(1)
        else:
            processed_row.append(0)
        
        # 類別特徵：付款方式
        payment_method = row[16]
        if payment_method == "Electronic check":
            processed_row.append(3)
        elif payment_method == "Mailed check":
            processed_row.append(2)
        elif payment_method == "Bank transfer (automatic)":
            processed_row.append(1)
        else:
            processed_row.append(0)
        
        # 數值特徵：每月和總費用
        processed_row.append(float(row[17]))
        processed_row.append(float(row[18]))
        
        return processed_row

    def euclidean_distance(self, point1, point2):
        return sum((a - b) ** 2 for a, b in zip(point1, point2))

    def predict(self, test_point):
        distances = []
        for i, train_point in enumerate(self.train_data):
            distance = self.euclidean_distance(test_point, train_point)
            distances.append((distance, self.train_labels[i]))
        # 找出最近的 k 個鄰居
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:self.k]
        
        # 多數決
        labels = [label for _, label in neighbors]
        prediction = max(set(labels), key=labels.count)
        return prediction

    def predict_all(self, test_data):
        predictions = []
        for test_point in test_data:
            prediction = self.predict(test_point)
            predictions.append(prediction)
        return predictions


# 主程序
if __name__ == "__main__":
    knn = KNearestNeighbors(k=29)
    knn.load_data("train.csv", "train_gt.csv")
    
    #載入test_data
    with open('test.csv', mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)  # 假設第一行是標題
        test_data = [row for row in reader]
    
    # 預處理測試資料
    test_data = [knn.preprocess(row) for row in test_data]

    # 標準化測試資料
    for i in range(len(test_data)):
        test_data[i] = knn.normalize(test_data[i])

    predictions = knn.predict_all(test_data)
    
    # 將預測結果寫入 CSV 檔案
    with open("test_pred.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Churn"])
        for prediction in predictions:
            writer.writerow([prediction])

