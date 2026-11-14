FROM python:3.12-slim

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

RUN pip install --no-cache-dir "streamlit>=1.40" "pandas>=2.2" \
    "llm-training-data-refinery @ https://github.com/adwitiyashukla/llm-training-data-refinery/archive/refs/heads/main.zip"

COPY --chown=user . .
USER user

EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
