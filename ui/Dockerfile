FROM python:3.11-slim

WORKDIR /app
COPY ui/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY ui /app/ui
COPY scripts /app/scripts

ENV PYTHONUNBUFFERED=1
EXPOSE 8780
CMD ["uvicorn", "ui.app:app", "--host", "0.0.0.0", "--port", "8780"]
