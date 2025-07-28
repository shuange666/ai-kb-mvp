FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# 先升级pip
RUN pip install --upgrade pip

COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 创建持久化存储目录
RUN mkdir -p /app/chrom_db

EXPOSE 10000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
