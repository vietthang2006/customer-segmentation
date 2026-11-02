# Dữ liệu

Bộ dữ liệu sử dụng cho dự án là: Online Retail Dataset

- Nguồn gốc: UCI Machine Learning Repository
- Link trực tiếp: https://archive.ics.uci.edu/dataset/352/online+retail

## Mô tả
Bộ dữ liệu Online Retail chứa thông tin giao dịch mua bán trực tuyến của một cửa hàng bán lẻ châu Âu trong khoảng thời gian nhất định. Mỗi dòng tương ứng với một giao dịch, bao gồm các thông tin như:
- mã khách hàng
- mã sản phẩm
- số lượng
- đơn giá
- ngày mua hàng
- quốc gia

Dữ liệu này thường được dùng cho các bài toán phân tích hành vi khách hàng, RFM, phân khúc thị trường, và dự đoán khách hàng tiềm năng.

## Cách tải dữ liệu
Người dùng có thể truy cập trực tiếp vào link nguồn để tải về bộ dữ liệu gốc:

https://archive.ics.uci.edu/dataset/352/online+retail

Sau khi tải xuống, hãy lưu file dữ liệu vào thư mục này để dự án có thể đọc dữ liệu từ `data/`.

## Dữ liệu đã làm sạch (`cleaned data`)
Sau khi chạy notebook xử lý dữ liệu, dự án sinh ra file dữ liệu giao dịch sạch tại:

`data/cleaned/retail_online.parquet`

File này đã được xử lý các vấn đề sau:
- Loại bỏ các dòng thiếu `CustomerID`
- Loại bỏ `UnitPrice <= 0`
- Loại bỏ các `StockCode` không phải mã sản phẩm hợp lệ
- Loại bỏ các ghi chú / mã dịch vụ không phải giao dịch mua hàng thực tế
- Giữ lại dữ liệu phù hợp cho phân tích RFM và phân khúc khách hàng

### Cách tạo dữ liệu đã làm sạch
Để tạo file cleaned data, mở notebook:

`notebook/cleaning.ipynb`

và chạy toàn bộ cell từ đầu đến cuối. Dữ liệu sạch sẽ được lưu tự động vào `data/cleaned/retail_online.parquet`.

## Dữ liệu phục vụ phân cụm và xây dựng chân dung khách hàng (`clustered data`)
Sau khi hoàn tất bước khám phá dữ liệu (EDA), kỹ thuật đặc trưng và chuẩn hóa trong:

`notebook/eda.ipynb`

Dự án xuất ra 2 tệp dữ liệu độc lập tại thư mục `data/clustered/` phục vụ hai mục đích chuyên biệt:

### 1. Tệp phục vụ phân cụm (`customer_scaled.parquet`)
Đường dẫn: `data/clustered/customer_scaled.parquet`

Tệp gồm khóa `CustomerID` và 8 đặc trưng đã biến đổi qua `np.log1p` và chuẩn hóa về thang `StandardScaler` ($\mu = 0, \sigma = 1$), sẵn sàng đưa vào các thuật toán K-Means, GMM, Hierarchical Clustering:
- `CustomerID`: Định danh duy nhất của khách hàng (dùng làm khóa nối)
- `Recency_scaled`: Số ngày từ lần mua gần nhất đến mốc phân tích
- `Frequency_scaled`: Số lượng đơn hàng mua thành công
- `NetSpending_scaled`: Doanh thu ròng sau khi trừ hàng hoàn trả
- `AvgOrderValue_scaled`: Giá trị đơn hàng trung bình (AOV)
- `NumDistinctProducts_scaled`: Số lượng mã sản phẩm (SKU) khác nhau đã mua
- `CustomerTenureDays_scaled`: Tổng thời gian gắn bó của khách hàng tính bằng ngày
- `PurchaseVelocity_Monthly_scaled`: Tần suất đặt hàng bình quân theo tháng
- `ReturnRate_scaled`: Tỷ lệ giá trị trả hàng trên tổng chi tiêu

### 2. Tệp phục vụ phân tích chân dung khách hàng (`customer_raw_profiling.parquet`)
Đường dẫn: `data/clustered/customer_raw_profiling.parquet`

Tệp gồm khóa `CustomerID` và 12 đặc trưng giữ nguyên giá trị đo lường thực tế (ngày thật, số đơn thật, tiền thật bằng bảng Anh, tỷ lệ % thật). Tệp này dùng để gán nhãn cụm sau khi mô hình hóa, giúp tính trung bình/trung vị thực tế và vẽ chân dung khách hàng (Customer Profiling) phục vụ nghiệp vụ kinh doanh:
- `CustomerID`: Khóa nối với kết quả phân cụm
- `Recency`: Số ngày kể từ lần mua cuối cùng
- `Frequency`: Tổng số đơn hàng mua thành công
- `NetSpending`: Doanh thu ròng thực tế
- `AvgOrderValue`: Giá trị đơn hàng trung bình
- `NumDistinctProducts`: Số lượng SKU khác nhau đã mua
- `CustomerTenureDays`: Thời gian gắn bó tính bằng ngày
- `PurchaseVelocity_Monthly`: Số đơn trung bình mỗi tháng
- `ReturnRate`: Tỷ lệ tiền hàng bị trả lại (từ 0.0 đến 1.0)
- `GrossSpending`: Tổng tiền hàng đặt mua trước khi trừ trả hàng
- `ReturnedAmount`: Tổng tiền hàng đã trả lại
- `CancelFrequency`: Số lần phát sinh giao dịch hủy/trả hàng
- `IsUK`: Biến chỉ báo khách hàng nội địa UK (1) hay quốc tế (0)

### 3. Tệp phân khúc khách hàng hoàn chỉnh (`customer_segmented.parquet`)
Đường dẫn: `data/clustered/customer_segmented.parquet`

Tệp kết quả cuối cùng sau khi phân cụm (4,320 dòng x 16 cột), bao gồm toàn bộ 12 đặc trưng đo lường gốc của `customer_raw_profiling.parquet`, kết hợp cùng nhãn phân khúc K-Means ($K=4$) và cờ dị biệt DBSCAN:
- `CustomerID`: Khóa định danh khách hàng
- 12 đặc trưng hành vi gốc (`Recency`, `Frequency`, `NetSpending`, `AvgOrderValue`, `NumDistinctProducts`, `CustomerTenureDays`, `PurchaseVelocity_Monthly`, `ReturnRate`, `GrossSpending`, `ReturnedAmount`, `CancelFrequency`, `IsUK`)
- `Cluster`: Nhãn cụm K-Means số nguyên (`0, 1, 2, 3`)
- `Cluster_Name`: Tên định danh chân dung kinh doanh (`VIP / Champions`, `Core Loyal / At-Risk Frequency`, `Newly Acquired / Promising Trials`, `Churned / Low-Value`)
- `Is_Anomaly`: Cờ chỉ báo khách hàng dị biệt từ DBSCAN (`1` nếu là điểm ngoại lai/noise, `0` nếu thuộc phân phối thông thường)

### Cách tạo các tệp trên
- 2 tệp `customer_scaled.parquet` và `customer_raw_profiling.parquet`: Chạy toàn bộ notebook `notebook/eda.ipynb`.
- Tệp `customer_segmented.parquet`: Chạy pipeline phân cụm và mapping qua module `src.profiling` trong `notebook/clustering.ipynb`.
