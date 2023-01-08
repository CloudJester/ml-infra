FROM apache/airflow
COPY airflow/dags/ /opt/airflow/dags/
COPY requirements.txt /
COPY train /opt/airflow/dags/

RUN pip install --no-cache-dir -r /requirements.txt