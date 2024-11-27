FROM python:3.10-slim-buster
USER root
RUN mkdir /app
COPY . /app/
WORKDIR /app/
#RUN pip3 install -r requirements.txt
RUN pip install -r requirements.txt

# # Install system dependencies
# RUN apt update -y && \
#     apt install -y --no-install-recommends \
#     build-essential \
#     libre2-dev \
#     gcc \
#     libffi-dev \
#     libssl-dev \
#     libpq-dev

# # Upgrade pip
# RUN pip install --upgrade pip

# # Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt



ENV AIRFLOW_HOME="/app/airflow"
ENV AIRFLOW_CORE_DAGBAG_IMPORT_TIMEOUT=1000
ENV AIRFLOW_CORE_ENABLE_XCOM_PICKLING=TRUE
ENV PYTHONPATH="/app"

RUN airflow db init
RUN airflow users create -e anant4830@gmail.com -f anant -l mohan -p admin -r Admin -u admin
RUN chmod 777 start.sh
RUN apt update -y
ENTRYPOINT [ "/bin/sh"]
CMD ["start.sh"]