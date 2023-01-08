from datetime import timedelta, datetime

# The DAG object
from airflow import DAG

# Operators
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator

from train.titanic.train import train_titanic

import logging


# initializing the default arguments
default_args = {
		'owner': 'Ranga',
		'start_date': datetime(2023, 1, 1),
		'retries': 3,
		'retry_delay': timedelta(minutes=5)
}

# Instantiate a DAG object
hello_world_dag = DAG('hello_world_dag',
		default_args=default_args,
		description='Hello World DAG',
		schedule_interval='* * * * *', 
		catchup=False,
		tags=['example, helloworld']
)

# python callable function
def titanic():
	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    logging.info('Training Model...')
    classifler = train_titanic()
    logging.info('Exporting Model...')
    joblib.dump(classifler, 'classifier.pkl')
    logging.info('Done')

# Creating first task
hello_world_task = PythonOperator(task_id='hello_world_task', python_callable=print_hello, dag=hello_world_dag)

# Set the order of execution of tasks. 
hello_world_task