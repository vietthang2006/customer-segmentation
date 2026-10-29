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
Sau khi chạy notebook xử lý dữ liệu, dự án sẽ sinh ra một file dữ liệu đã làm sạch tại:

`data/cleaned/cleaned_retail_online.xlsx`

File này đã được xử lý các vấn đề sau:
- loại bỏ các dòng thiếu `CustomerID`
- loại bỏ `UnitPrice <= 0`
- loại bỏ các `StockCode` không phải mã sản phẩm hợp lệ
- loại bỏ các ghi chú / mã dịch vụ không phải giao dịch mua hàng thực tế
- giữ lại dữ liệu phù hợp cho phân tích RFM và phân khúc khách hàng

## Cách tạo dữ liệu đã làm sạch
Để tạo file cleaned data, hãy mở notebook sau:

`notebook/cleaning.ipynb`

và chạy toàn bộ cell từ đầu đến cuối. Khi notebook chạy xong, file dữ liệu sạch sẽ được lưu tự động vào:

```text
data/cleaned/cleaned_retail_online.xlsx