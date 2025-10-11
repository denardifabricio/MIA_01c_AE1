FROM python:3.11-slim

WORKDIR /app

COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .


EXPOSE 8505

# Usa la variable de entorno COURSE_VIEWER_PORT para definir el puerto de Streamlit
CMD ["/bin/sh", "-c", "streamlit run main.py --server.port ${COURSE_VIEWER_PORT:-8505}"]
