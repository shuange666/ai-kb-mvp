FROM python:3.10

WORKDIR /app

# 先升级 pip 避免旧版 pip 的依赖解析问题
RUN pip install --upgrade pip

COPY requirements.txt .

# 使用 --no-cache-dir 和 --upgrade 确保干净安装
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . .

# 修正命令拼写 (CHD -> CMD) 并添加正确的双短横线
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
