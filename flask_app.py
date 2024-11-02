# Import các thư viện cần thiết
from flask import Flask, render_template, request, send_file  # Flask framework để tạo web app
import pandas as pd  # Thư viện xử lý dữ liệu dạng bảng
import matplotlib.pyplot as plt  # Thư viện vẽ đồ thị
import io  # Thư viện xử lý input/output
import base64  # Thư viện mã hóa dữ liệu
from prophet import Prophet  # Thư viện dự báo dữ liệu chuỗi thời gian
import matplotlib.dates as mdates  # Xử lý định dạng ngày tháng trong đồ thị
from plotly.subplots import make_subplots  # Tạo nhiều đồ thị con
import plotly.graph_objects as go  # Thư viện vẽ đồ thị tương tác
import json  # Xử lý dữ liệu JSON
from statsmodels.tsa.arima.model import ARIMA  # Mô hình dự báo ARIMA
from scipy.optimize import curve_fit  # Tối ưu hóa hàm
import numpy as np  # Thư viện tính toán số học
import plotly  # Thư viện vẽ đồ thị tương tác

# Khởi tạo Flask app
app = Flask(__name__)

# Hàm tính tỷ lệ tăng trưởng
def tinh_ti_le_tang_truong(series):
    # Chuyển đổi dữ liệu sang dạng số
    series = pd.to_numeric(series, errors='coerce')
    # Tính tỷ lệ tăng trưởng = (giá trị hiện tại - giá trị trước) / giá trị trước * 100
    return (series.pct_change() * 100).fillna(0)

# Hàm tạo biểu đồ hoạt động kinh doanh
def create_plot_business_activity(data):
    # Tạo figure mới
    fig = go.Figure()
    
    # Thêm cột bar cho tổng số doanh nghiệp
    fig.add_trace(go.Bar(
        x=data['Năm'],  # Trục x là năm
        y=data['Tổng số doanh nghiệp hoạt động'],  # Trục y là số lượng DN
        name='Tổng số doanh nghiệp',
        marker_color='skyblue'  # Màu cột
    ))
    
    # Thêm đường cho từng loại hình kinh doanh
    for column, color in [
        ('Doanh nghiệp có hình thức kinh doanh trên MXH', 'green'),
        ('Doanh nghiệp có hoạt động kinh doanh trên sàn TMĐT', 'orange')
    ]:
        values = data[column]
        # Tính tỷ lệ tăng trưởng cho mỗi năm
        growth_rates = tinh_ti_le_tang_truong(values)
        
        # Thêm đường cho từng loại hình
        fig.add_trace(go.Scatter(
            x=data['Năm'],
            y=values,
            name=column,
            line=dict(color=color),
            mode='lines+markers',  # Hiển thị cả đường và điểm
            text=[f'Tăng trưởng: {rate:.1f}%' for rate in growth_rates],
            hovertemplate='Năm: %{x}<br>Số lượng: %{y}<br>%{text}<extra></extra>'  # Template hiển thị khi hover
        ))
    
    # Cấu hình layout cho biểu đồ
    fig.update_layout(
        title='Biểu đồ hoạt động của doanh nghiệp theo năm',
        xaxis_title='Năm',
        yaxis_title='Số lượng doanh nghiệp (Triệu DN)',
        showlegend=True,  # Hiển thị chú thích
        hovermode='x unified',  # Chế độ hover
        plot_bgcolor='white',  # Màu nền
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        )
    )
    
    # Chuyển đổi biểu đồ sang JSON để có thể truyền vào template
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_plot_payment_methods(data):
    fig = go.Figure()
    
    # Định nghĩa các phương thức thanh toán và màu sắc tương ứng
    payment_methods = [
        'Doanh nghiệp sử dụng phương thức thanh toán tiền mặt',
        'Doanh nghiệp sử dụng phương thức thanh toán internet banking',
        'Doanh nghiệp sử dụng phương thức thanh toán Ví điện tử'
    ]
    colors = ['red', 'blue', 'purple']
    
    # Tạo đường cho từng phương thức thanh toán
    for method, color in zip(payment_methods, colors):
        # Chuyển đổi dữ liệu sang dạng số
        values = pd.to_numeric(data[method], errors='coerce')
        # Tính tỷ lệ tăng trưởng
        growth_rates = tinh_ti_le_tang_truong(values)
        
        # Thêm đường vào biểu đồ
        fig.add_trace(go.Scatter(
            x=data['Năm'],
            y=values,
            # Lấy tên ngắn gọn bằng cách bỏ phần "Doanh nghiệp sử dụng phương thức thanh toán"
            name=method.split('Doanh nghiệp sử dụng phương thức thanh toán ')[-1],
            line=dict(color=color),
            mode='lines+markers',
            text=[f'Tăng trưởng: {rate:.1f}%' for rate in growth_rates],
            hovertemplate='Năm: %{x}<br>Số lượng: %{y}<br>%{text}<extra></extra>'
        ))
    
    # Cấu hình layout cho biểu đồ
    fig.update_layout(
        title='Biểu đồ phương thức thanh toán của doanh nghiệp',
        xaxis_title='Năm',
        yaxis_title='Số lượng doanh nghiệp',
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray')
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_plot_population_internet(data):
    fig = go.Figure()
    
    # Định nghĩa các cột dữ liệu cần vẽ và màu sắc tương ứng
    population_columns = [
        'Dân số Việt Nam (Triệu người)',
        'Số người dân sử dụng Internet (Triệu người)',
        'Số liệu người tiêu dùng mua sắp trực tuyến (triệu người)'
    ]
    colors = ['lightblue', 'lightgreen', 'cyan']
    
    # Tạo đường cho từng loại dữ liệu
    for column, color in zip(population_columns, colors):
        values = data[column]
        # Tính phần trăm so với dân số nếu không phải là cột dân số
        if column != 'Dân số Việt Nam (Triệu người)':
            percentages = [(value / data['Dân số Việt Nam (Triệu người)'][i]) * 100 
                         for i, value in enumerate(values)]
            hover_text = [f'Tỷ lệ: {pct:.1f}%' for pct in percentages]
        else:
            hover_text = [f'Dân số: {val:.1f}M' for val in values]
            
        # Thêm đường vào biểu đồ với fill để tạo hiệu ứng diện tích
        fig.add_trace(go.Scatter(
            x=data['Năm'],
            y=values,
            name=column,
            fill='tonexty',  # Tô màu diện tích dưới đường
            line=dict(color=color),
            text=hover_text,
            hovertemplate='Năm: %{x}<br>Số lượng: %{y}<br>%{text}<extra></extra>'
        ))
    
    # Cấu hình layout
    fig.update_layout(
        title='Biểu đồ người dùng internet và mua sắm trực tuyến so với tổng dân số',
        xaxis_title='Năm',
        yaxis_title='Số lượng (Triệu người)',
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray')
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_plot_b2c_revenue(data):
    fig = go.Figure()
    
    # Thêm cột bar cho doanh thu B2C
    fig.add_trace(go.Bar(
        x=data['Năm'],
        y=data['Doanh thu B2C (Tỷ đô)'],
        name='Doanh thu B2C',
        marker_color='skyblue',
        text=data['Doanh thu B2C (Tỷ đô)'],  # Hiển thị giá trị trên mỗi cột
        textposition='auto',
        hovertemplate='Năm: %{x}<br>Doanh thu: %{y} tỷ đô<extra></extra>'
    ))
    
    # Thêm đường cho tỷ lệ tăng trưởng (trục y thứ 2)
    fig.add_trace(go.Scatter(
        x=data['Năm'],
        y=data['Tỷ lệ tăng trưởng doanh thu B2C (%)'],
        name='Tỷ lệ tăng trưởng',
        line=dict(color='red'),
        mode='lines+markers',
        yaxis='y2',  # Sử dụng trục y thứ 2
        text=[f'{val:.1f}%' for val in data['Tỷ lệ tăng trưởng doanh thu B2C (%)']],
        hovertemplate='Năm: %{x}<br>Tăng trưởng: %{text}<extra></extra>'
    ))
    
    # Cấu hình layout với 2 trục y
    fig.update_layout(
        title='Doanh thu B2C và Tỷ lệ tăng trưởng theo năm',
        xaxis_title='Năm',
        yaxis_title='Doanh thu B2C (Tỷ đô)',
        yaxis2=dict(  # Cấu hình trục y thứ 2
            title='Tỷ lệ tăng trưởng (%)',
            overlaying='y',
            side='right',
            range=[0, 100],
            showgrid=False
        ),
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray')
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def logistic_function(x, L, k, x0):
    # L: giá trị tối đa (carrying capacity)
    # k: tốc độ tăng trưởng
    # x0: điểm giữa của đường cong
    return L / (1 + np.exp(-k * (x - x0)))

def create_forecast_b2c_revenue(data):
    # Xử lý và chuẩn bị dữ liệu
    df = data.copy()
    df['Doanh thu B2C (Tỷ đô)'] = pd.to_numeric(df['Doanh thu B2C (Tỷ đô)'], errors='coerce')
    df = df.dropna(subset=['Doanh thu B2C (Tỷ đô)'])
    
    # Kiểm tra đủ dữ liệu để dự báo (tối thiểu 3 điểm)
    if len(df) < 3:
        return json.dumps({
            'data': [],
            'layout': {'title': 'Không đủ dữ liệu để dự báo'}
        }, cls=plotly.utils.PlotlyJSONEncoder)
    
    try:
        # Áp dụng mô hình ARIMA
        y = df['Doanh thu B2C (Tỷ đô)'].values
        model = ARIMA(y, order=(1,1,1))  # Mô hình ARIMA(1,1,1)
        results = model.fit()
        
        # Dự báo 3 năm tiếp theo
        forecast = results.forecast(steps=3)
        
        # Tạo trục thời gian cho dự báo
        last_year = df['Năm'].iloc[-1]
        forecast_years = np.array([last_year + i for i in range(1, 4)])
        
        # Vẽ biểu đồ kết hợp dữ liệu thực tế và dự báo
        fig = go.Figure()
        
        # Dữ liệu thực tế
        fig.add_trace(go.Scatter(
            x=df['Năm'],
            y=df['Doanh thu B2C (Tỷ đô)'],
            name='Dữ liệu thực tế',
            mode='lines+markers'
        ))
        
        # Dữ liệu dự báo
        fig.add_trace(go.Scatter(
            x=forecast_years,
            y=forecast,
            name='Dự báo',
            mode='lines+markers',
            line=dict(dash='dash')  # Đường đứt khúc cho dự báo
        ))
        
        fig.update_layout(
            title='Dự báo doanh thu B2C (ARIMA)',
            xaxis_title='Năm',
            yaxis_title='Doanh thu B2C (Tỷ đô)',
            showlegend=True
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    except Exception as e:
        print(f"Lỗi khi dự báo doanh thu: {str(e)}")
        return json.dumps({
            'data': [],
            'layout': {'title': 'Lỗi khi tạo dự báo'}
        }, cls=plotly.utils.PlotlyJSONEncoder)

def create_forecast_online_shoppers(data):
    # Xử lý và chuẩn bị dữ liệu
    df = data.copy()
    df['Số liệu người tiêu dùng mua sắp trực tuyến (triệu người)'] = pd.to_numeric(
        df['Số liệu người tiêu dùng mua sắp trực tuyến (triệu người)'], 
        errors='coerce'
    )
    df = df.dropna(subset=['Số liệu người tiêu dùng mua sắp trực tuyến (triệu người)'])
    
    if len(df) < 3:
        return json.dumps({
            'data': [],
            'layout': {'title': 'Không đủ dữ liệu để dự báo'}
        }, cls=plotly.utils.PlotlyJSONEncoder)
    
    try:
        # Chuẩn bị dữ liệu cho mô hình logistic
        years = np.array(range(len(df['Năm'])))
        y = df['Số liệu người tiêu dùng mua sắp trực tuyến (triệu người)'].values
        
        # Fit mô hình logistic
        popt, _ = curve_fit(logistic_function, years, y, p0=[100, 0.5, 5], maxfev=10000)
        
        # Dự báo 3 năm tiếp theo
        future_years = np.array(range(len(years) + 3))
        y_pred = logistic_function(future_years, *popt)
        
        # Tạo trục thời gian đầy đủ
        all_years = np.concatenate([
            df['Năm'], 
            np.array([df['Năm'].iloc[-1] + i for i in range(1, 4)])
        ])
        
        # Vẽ biểu đồ
        fig = go.Figure()
        
        # Dữ liệu thực tế
        fig.add_trace(go.Scatter(
            x=df['Năm'],
            y=y,
            name='Dữ liệu thực tế',
            mode='lines+markers'
        ))
        
        # Dữ liệu dự báo
        fig.add_trace(go.Scatter(
            x=all_years,
            y=y_pred,
            name='Dự báo',
            mode='lines+markers',
            line=dict(dash='dash')
        ))
        
        fig.update_layout(
            title='Dự báo số người mua sắm trực tuyến (Logistic)',
            xaxis_title='Năm',
            yaxis_title='Số người (Triệu)',
            showlegend=True
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    except Exception as e:
        print(f"Lỗi khi dự báo số người mua sắm: {str(e)}")
        return json.dumps({
            'data': [],
            'layout': {'title': 'Lỗi khi tạo dự báo'}
        }, cls=plotly.utils.PlotlyJSONEncoder)

# Routes - Định nghĩa các endpoint của web app
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Kiểm tra file upload
        if 'file' not in request.files:
            return 'No file uploaded', 400

        file = request.files['file']
        if file.filename == '':
            return 'No file selected', 400

        # Đọc file Excel
        data = pd.read_excel(file)

        # Tạo các biểu đồ
        plot1 = create_plot_business_activity(data)
        plot2 = create_plot_payment_methods(data)
        plot3 = create_plot_population_internet(data)
        plot4 = create_plot_b2c_revenue(data)
        forecast1 = create_forecast_b2c_revenue(data)
        forecast2 = create_forecast_online_shoppers(data)

        # Trả về template với các biểu đồ
        return render_template('results.html',
                             plot1=plot1,
                             plot2=plot2,
                             plot3=plot3,
                             plot4=plot4,
                             forecast1=forecast1,
                             forecast2=forecast2
                             )

    # Nếu là GET request, hiển thị trang upload
    return render_template('index.html')

# Chạy ứng dụng trong môi trường debug
if __name__ == '__main__':
    app.run(debug=True)
