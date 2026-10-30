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
Sau khi chạy notebook xử lý dữ liệu, dự án sẽ sinh ra file dữ liệu giao dịch sạch tại:

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

và chạy toàn bộ cell từ đầu đến cuối. Khi chạy xong, file dữ liệu sạch sẽ được lưu tự động vào `data/cleaned/retail_online.parquet`.

## Dữ liệu phục vụ phân cụm (`clustered data`)
Sau khi hoàn tất bước khám phá dữ liệu (EDA) và tạo đặc trưng khách hàng trong:

`notebook/eda.ipynb`

Dữ liệu được tổng hợp ở cấp độ khách hàng (customer-level), tính toán các chỉ số RFM mở rộng, chuẩn hóa thang đo qua phép biến đổi `log1p`, loại bỏ các biến đa cộng tuyến và lưu ma trận đặc trưng cuối cùng vào:

`data/clustered/retail_online.parquet` (hoặc `data/clustered/online_retail.parquet`)

Tệp này gồm 8 đặc trưng cốt lõi sẵn sàng đưa vào các thuật toán phân cụm (K-Means, GMM, Hierarchical Clustering):
- `CustomerID`: Định danh duy nhất của từng khách hàng
- `Recency_log1p`: Số ngày từ lần mua gần nhất đến mốc phân tích (thang log)
- `Frequency_log1p`: Số lượng đơn hàng mua thành công (thang log)
- `NetSpending_log1p`: Doanh thu ròng sau khi trừ hàng hoàn trả (thang log)
- `AvgOrderValue_log1p`: Giá trị đơn hàng trung bình AOV (thang log)
- `NumDistinctProducts_log1p`: Số lượng mã sản phẩm (SKU) khác nhau đã mua (thang log)
- `CustomerTenureDays_log1p`: Tổng thời gian gắn bó của khách hàng tính bằng ngày (thang log)
- `PurchaseVelocity_Monthly_log1p`: Tần suất đặt hàng bình quân theo tháng (thang log)
- `ReturnRate_log1p`: Tỷ lệ giá trị trả hàng trên tổng chi tiêu (thang log)

### Cách tạo dữ liệu phân cụm
Chạy toàn bộ notebook `notebook/eda.ipynb` từ đầu đến cuối. Quá trình xử lý sẽ tự động tạo thư mục `data/clustered/` và xuất file `retail_online.parquet`.
