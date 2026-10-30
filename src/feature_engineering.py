from typing import Optional
import numpy as np
import pandas as pd


def extract_customer_features(
    df: pd.DataFrame,
    reference_date: Optional[pd.Timestamp] = None,
    filter_negative_net: bool = True,
    days_per_month: float = 30.4375,
) -> pd.DataFrame:
    
    # 0. Copy and ensure proper types
    data = df.copy()
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    data['CustomerID'] = data['CustomerID'].astype(str)
    
    # Calculate TotalPrice if not present
    if 'TotalPrice' not in data.columns:
        data['TotalPrice'] = data['Quantity'] * data['UnitPrice']

    # 1. Determine reference date (1 day after max date in dataset)
    if reference_date is None:
        reference_date = data['InvoiceDate'].max() + pd.Timedelta(days=1)
    else:
        reference_date = pd.to_datetime(reference_date)

    # 2. Separate successful purchases and cancellations
    is_cancelled = data['InvoiceNo'].astype(str).str.startswith('C') | (data['Quantity'] < 0) #type: ignore
    purchases = data[~is_cancelled]
    returns = data[is_cancelled]

    # 3. Aggregate successful purchases
    purchases_agg = purchases.groupby('CustomerID').agg(
        FirstPurchaseDate=('InvoiceDate', 'min'),
        LastPurchaseDate=('InvoiceDate', 'max'),
        Frequency=('InvoiceNo', 'nunique'),
        GrossSpending=('TotalPrice', 'sum'),
        TotalQuantity=('Quantity', 'sum'),
        NumDistinctProducts=('StockCode', 'nunique'),
        Country=('Country', 'first')
    )

    # 4. Aggregate returns / cancellations
    if not returns.empty:
        returns_agg = returns.groupby('CustomerID').agg(
            ReturnedAmount=('TotalPrice', lambda x: abs(x.sum())),
            CancelFrequency=('InvoiceNo', 'nunique')
        )
    else:
        returns_agg = pd.DataFrame(
            columns=['ReturnedAmount', 'CancelFrequency'],
            index=purchases_agg.index
        )

    # 5. Merge purchases and returns
    features = purchases_agg.join(returns_agg, how='left')
    features['ReturnedAmount'] = features['ReturnedAmount'].fillna(0.0)
    features['CancelFrequency'] = features['CancelFrequency'].fillna(0).astype(int)

    # 6. Compute Core Features
    # Recency: days from last purchase to reference date
    features['Recency'] = (reference_date - features['LastPurchaseDate']).dt.days

    # Customer Tenure: days from first purchase to reference date (clipped at minimum 1 day)
    features['CustomerTenureDays'] = (
        (reference_date - features['FirstPurchaseDate']).dt.days
    ).clip(lower=1)

    # Net Spending 
    features['NetSpending'] = (features['GrossSpending'] - features['ReturnedAmount']).round(2)

    # Return Rate (ReturnedAmount / GrossSpending)
    features['ReturnRate'] = np.where(
        features['GrossSpending'] > 0,
        (features['ReturnedAmount'] / features['GrossSpending']).clip(0.0, 1.0),
        0.0
    )

    # Average Order Value (AOV: NetSpending / Frequency)
    features['AvgOrderValue'] = np.where(
        features['Frequency'] > 0,
        features['NetSpending'] / features['Frequency'],
        0.0
    )

    # Average Items Per Order
    features['AvgItemsPerOrder'] = np.where(
        features['Frequency'] > 0,
        features['TotalQuantity'] / features['Frequency'],
        0.0
    )

    # Purchase Velocity based on Poisson Rate parameter (lambda)
    # lambda_monthly = Frequency / (Tenure in months)
    # Represents the expected number of orders a customer places per month.
    tenure_in_months = (features['CustomerTenureDays'] / days_per_month).clip(lower=1.0)
    features['PurchaseVelocity_Monthly'] = features['Frequency'] / tenure_in_months

    # Binary indicator for UK
    features['IsUK'] = (features['Country'] == 'United Kingdom').astype(int)

    # 7. Select & Order unified columns
    ordered_cols = [
        'Recency',
        'Frequency',
        'NetSpending',
        'GrossSpending',
        'ReturnedAmount',
        'ReturnRate',
        'CancelFrequency',
        'NumDistinctProducts',
        'AvgOrderValue',
        'AvgItemsPerOrder',
        'CustomerTenureDays',
        'PurchaseVelocity_Monthly',
        'IsUK',
    ]
    features = features[ordered_cols]

    # 8. Filter out negative or zero net spending if requested
    if filter_negative_net:
        features = features[features['NetSpending'] > 0].copy()

    return features


def get_feature_descriptions() -> pd.DataFrame:
    """
    Return a reference table explaining the definitions and roles of all engineered features.
    """
    metadata = [
        {
            "Feature": "Recency",
            "Formula / Concept": "Reference Date - Last Purchase Date (days)",
            "Business Meaning": "Số ngày kể từ lần mua hàng thành công cuối cùng đến mốc phân tích.",
        },
        {
            "Feature": "Frequency",
            "Formula / Concept": "Count of distinct positive InvoiceNo",
            "Business Meaning": "Tổng số đơn hàng mua thành công (không tính đơn hủy).",
        },
        {
            "Feature": "NetSpending",
            "Formula / Concept": "GrossSpending - ReturnedAmount",
            "Business Meaning": "Doanh thu ròng thực tế thu được từ khách sau khi trừ giá trị hoàn trả.",
        },
        {
            "Feature": "GrossSpending",
            "Formula / Concept": "Sum(Quantity * UnitPrice) for Quantity > 0",
            "Business Meaning": "Tổng tiền hàng khách đã đặt mua trước khi hoàn trả.",
        },
        {
            "Feature": "ReturnedAmount",
            "Formula / Concept": "|Sum(Quantity * UnitPrice)| for cancellations",
            "Business Meaning": "Tổng giá trị tiền hàng đã bị hoàn trả/hủy.",
        },
        {
            "Feature": "ReturnRate",
            "Formula / Concept": "ReturnedAmount / GrossSpending",
            "Business Meaning": "Tỷ lệ tiền hoàn trả trên tổng tiền mua (đo lường độ rủi ro/phiền toái hoàn hàng).",
        },
        {
            "Feature": "CancelFrequency",
            "Formula / Concept": "Count of distinct cancellation InvoiceNo ('C')",
            "Business Meaning": "Số lần khách phát sinh yêu cầu hủy đơn hoặc trả hàng.",
        },
        {
            "Feature": "NumDistinctProducts",
            "Formula / Concept": "Count of unique StockCode",
            "Business Meaning": "Độ đa dạng danh mục sản phẩm khách hàng đã chọn mua.",
        },
        {
            "Feature": "AvgOrderValue",
            "Formula / Concept": "NetSpending / Frequency",
            "Business Meaning": "Giá trị trung bình của một đơn hàng (AOV) - thống nhất từ các ý trùng lặp.",
        },
        {
            "Feature": "AvgItemsPerOrder",
            "Formula / Concept": "TotalQuantity / Frequency",
            "Business Meaning": "Số lượng món đồ trung bình trong mỗi đơn hàng.",
        },
        {
            "Feature": "CustomerTenureDays",
            "Formula / Concept": "Reference Date - First Purchase Date (days)",
            "Business Meaning": "Vòng đời quan sát của khách hàng (thời gian gắn bó từ đơn đầu tiên).",
        },
        {
            "Feature": "PurchaseVelocity_Monthly",
            "Formula / Concept": "Frequency / (CustomerTenureDays / 30.4375)",
            "Business Meaning": "Tốc độ mua hàng (ước lượng tham số lambda của phân phối Poisson: số đơn/tháng).",
        },
        {
            "Feature": "IsUK",
            "Formula / Concept": "1 if Country == 'United Kingdom' else 0",
            "Business Meaning": "Biến chỉ báo khách hàng nội địa (UK) vs quốc tế (Non-UK).",
        },
    ]
    return pd.DataFrame(metadata)

