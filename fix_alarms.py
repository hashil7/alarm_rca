import mysql.connector
import logging
import configparser
from mysql.connector import Error

# Load database credentials from config.ini
config = configparser.ConfigParser()
config.read('config.ini')
mysql_config = config['mysql']

# Configure logging to a new log file for alarm history actions
logging.basicConfig(
    filename='fix_alarms.log',
    filemode='a',
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO
)

def connect_to_database():
    try:
        print("Connecting to the database...")
        connection = mysql.connector.connect(
            host=mysql_config['host'],
            database=mysql_config['database'],
            user=mysql_config['user'],
            password=mysql_config['password']
        )
        if connection.is_connected():
            logging.info("Successfully connected to the database.")
            print("Successfully connected to the database.")
    except Error as e:
        logging.error("Error connecting to the database: " + str(e))
        print("Error connecting to the database:", str(e))
    finally:
        if connection.is_connected():
            connection.close()
            logging.info("MySQL connection is closed.")
            print("MySQL connection is closed.")

def move_alarms_to_history(alarm_ids):
    try:
        print("Connecting to the database...")
        connection = mysql.connector.connect(
            host=mysql_config['host'],
            database=mysql_config['database'],
            user=mysql_config['user'],
            password=mysql_config['password']
        )
        if connection.is_connected():
            cursor = connection.cursor(dictionary=True)
            print("Fetching alarms for the given IDs...")
            
            # Fetch rows from alarm table for the given alarm_ids
            format_strings = ','.join(['%s'] * len(alarm_ids))
            cursor.execute(f"SELECT * FROM alarm WHERE ID IN ({format_strings})", tuple(alarm_ids))
            alarms = cursor.fetchall()
            print(f"Fetched {len(alarms)} alarms.")
            
            if alarms:
                move_count = 0
                for alarm in alarms:
                    # Set remarks to "NAS Alarm clearance"
                    alarm['REMARKS'] = "NAS Alarm clearance"
                    
                    # Map columns from alarm (source) to alarm_hist (target),
                    # filling extra columns with dummy data.
                    alarm_hist = {
                        'ID':              alarm['ID'],
                        'NOTIF_ID':        alarm['NOTIF_ID'],
                        'NE_TIME':         alarm['NE_TIME'],
                        'CLEARED_TIME':    alarm['NE_TIME'],            # dummy: same as NE_TIME
                        'OBJ_NAME':        alarm['OBJ_NAME'],
                        'OBJ_TYPE':        alarm['OBJ_TYPE'],
                        'RES_NAME':        alarm['RES_NAME'],
                        'EMS_TIME':        alarm['EMS_TIME'],
                        'CLEARED_EMS_TIME':alarm['EMS_TIME'],           # dummy: same as EMS_TIME
                        'CURR_LOG_TIME':   alarm['LOG_TIME'],           # dummy: from alarm's LOG_TIME
                        'HIST_LOG_TIME':   alarm['LOG_TIME'],           # dummy: same as LOG_TIME
                        'PROB_CAUSE':      alarm['PROB_CAUSE'],
                        'PERC_SEVERITY':   alarm['PERC_SEVERITY'],
                        'NMS_SEVERITY':    alarm['NMS_SEVERITY'],
                        'ADDI_INFO':       alarm['ADDI_INFO'],
                        'RCA_INDICATOR':   alarm['RCA_INDICATOR'],
                        'RCA_ID':          alarm['RCA_ID'],
                        'VISIBLE':         alarm['VISIBLE'],
                        'TT_ID':           alarm['TT_ID'],
                        'LOCATION_ID':     alarm['LOCATION_ID'],
                        'REMARKS':         alarm['REMARKS'],
                        'CLEARED_NOTIF_ID':'' ,                     # dummy data
                        'PROCESS_FLAG':    alarm['PROCESS_FLAG'],
                        'CATEGORY':        alarm['CATEGORY'],
                        'interface':       alarm['interface'],
                        'RFO':             alarm['RFO'],
                        'FIBER_DETAILS':   alarm['FIBER_DETAILS'],
                        'LATEST_RFO':      alarm['LATEST_RFO'],
                        'PREVIOUS_TT_ID':  alarm['PREVIOUS_TT_ID'],
                        'PREVIOUS_RCA_ID': alarm['PREVIOUS_RCA_ID'],
                        'UPS_BATTERY_PERCENT': alarm['UPS_BATTERY_PERCENT']
                    }
                    
                    try:
                        # Build and execute the INSERT query for alarm_hist.
                        columns = ', '.join(alarm_hist.keys())
                        placeholders = ', '.join(['%s'] * len(alarm_hist))
                        sql = f"INSERT INTO alarm_hist_dummy ({columns}) VALUES ({placeholders})"
                        cursor.execute(sql, list(alarm_hist.values()))
                        
                        # Only delete if insert was successful
                        cursor.execute("DELETE FROM alarm WHERE ID = %s", (alarm['ID'],))
                        move_count += 1
                        print(f"Moved alarm ID {alarm['ID']} to alarm_hist.")
                    except Error as insert_error:
                        logging.error(f"Failed to move alarm ID {alarm['ID']}: {str(insert_error)}")
                        print(f"Failed to move alarm ID {alarm['ID']}: {str(insert_error)}")
                        continue

                connection.commit()
                logging.info(f"Successfully moved {move_count} alarm record(s) to alarm_hist table and removed from alarm table.")
                print(f"Successfully moved {move_count} alarm record(s) to alarm_hist table and removed from alarm table.")
            else:
                logging.info("No alarms found for the provided alarm IDs.")
                print("No alarms found for the provided alarm IDs.")
                
    except Error as e:
        logging.error("Error moving alarms to alarm_hist: " + str(e))
        print("Error moving alarms to alarm_hist:", str(e))
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            logging.info("MySQL connection is closed.")
            print("MySQL connection is closed.")

if __name__ == "__main__":
    alarm_ids = ['LIN_20250210095444_59b3668b8965','LIN_20250210095445_80372af1251e','DUMMY_5dbada7d-34f1-4a63-a135-109c0c3145a1','DUMMY_fe165912-1ff9-4e71-9d63-8e58f43fee2b']  # Example list of alarm IDs
    move_alarms_to_history(alarm_ids)


