
FROM python:3.10-slim

RUN apt-get update && apt-get install -y git ffmpeg
COPY requirements.txt /app/
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app

EXPOSE 8050

# The .cache directory will be optionally bound to /root/.cache/
VOLUME ["/root/.cache/"]

# The working directory can be mounted to /app/workdir
VOLUME ["/app/workdir"]

CMD ["python", "webui.py"]