"""
an Aiflow DAG task to upload the whole local working directory to the AutoDL remote resource.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash_operator import BashOperator

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'scp_file_transfer',
    default_args=default_args,
    description='Transfer file to remote host via SCP',
    schedule_interval=timedelta(days=1),  # 每天执行一次
    catchup=False,
)

# 本地文件路径
local_file_path = '/path/to/local/file.txt'
# 远程主机信息
remote_user = 'username'
remote_host = 'remote.server.com'
remote_port = '22'  # 默认SSH端口
remote_path = '/path/to/remote/destination/'
local_private_key_dir = '/Users/orbithe/.ssh/id_rsa'

# 使用BashOperator执行SCP命令
transfer_file = BashOperator(
    task_id='scp_file_transfer',
    bash_command=f'scp -r -P {remote_port} -i {local_private_key_dir}  {local_file_path} {remote_user}@{remote_host}:{remote_path}',
    dag=dag,
)


transfer_file
