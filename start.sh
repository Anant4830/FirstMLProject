#!bin/sh
nohup airflow scheduler &
airflow webserver

# # Start the Airflow scheduler in the background
# nohup airflow scheduler &

# # Start the Airflow webserver and keep it in the foreground
# exec airflow webserver -p 8080
