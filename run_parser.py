import os
import sys
import json
import logging
import glob
import pprint
import datetime
import time
import random
import mysql.connector
from mysql.connector import Error

current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.dirname(current_dir)
sys.path.insert(0, module_dir)


from resume_parser import ResumeParser

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

#Change the database details according to the google bucket 
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'ats_db2'
}

def get_db_connection():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            return connection
    except Error as e:
        logging.error(f"Error connecting to MySQL: {e}")
        return None

def insert_into_db(connection, filename, data):
    cursor = connection.cursor()
    
    fullname = data.get('name', '')
    email = data.get('email', '')
    mobile = data.get('mobile_number', '')
    total_exp = data.get('total_experience', 0)
    
    skills_list = data.get('skills', [])
    parsed_skills_str = ", ".join(skills_list) if isinstance(skills_list, list) else str(skills_list)

    education_list = data.get('education', [])
    education_str = json.dumps(education_list)
    
    full_json_data = json.dumps(data)
    updated_at = datetime.datetime.now()

    #change to candidate id and also add job order id to remove duplicate entries intead of this
    candidate_id = int(time.time()) + random.randint(1, 100)
    job_id = 1 

    sql_query = """
    INSERT INTO candidate_parsed_data 
    (candidate_id, job_id, fullname, email, mobile_number, total_experience, parsed_skills, education, json_data, updated_at) 
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    values = (
        candidate_id, 
        job_id, 
        fullname, 
        email, 
        mobile, 
        total_exp, 
        parsed_skills_str, 
        education_str, 
        full_json_data, 
        updated_at
    )

    try:
        cursor.execute(sql_query, values)
        connection.commit()
        logging.info(f"SUCCESS: Inserted record for {filename} (ID: {candidate_id})")
    except Error as e:
        logging.error(f"Failed to insert data for {filename}. Error: {e}")
    finally:
        cursor.close()

def main():
    #get resumes from the google bucket intead of this  
    test_files_directory = r"C:\wamp64\www\python_parser_repo\ResumeParser\resume_parser\resume_parse\test_files"
    skills_file_path = os.path.join(current_dir, 'skills.txt')

    if not os.path.exists(test_files_directory):
        logging.error(f"Error: Directory not found {test_files_directory}")
        sys.exit(1)

    logging.info(f"Scanning for resume files in: {test_files_directory}")
    
    allowed_extensions = ['*.pdf', '*.doc', '*.docx']
    resume_files = []

    for ext in allowed_extensions:
        resume_files.extend(glob.glob(os.path.join(test_files_directory, ext)))

    if not resume_files:
        logging.warning("No resume files found.")
        sys.exit(0)

    db_conn = get_db_connection()
    if not db_conn:
        sys.exit(1)

    successful_uploads_count = 0

    for resume_file_path in resume_files:
        filename = os.path.basename(resume_file_path)
        try:
            parser = ResumeParser(resume_file_path, skills_file=skills_file_path)
            extracted_data = parser.get_extracted_data()

            if extracted_data:
                
                insert_into_db(db_conn, filename, extracted_data)
                
                successful_uploads_count += 1
            else:
                logging.warning(f"No meaningful data extracted for {filename}. Skipping database insertion.")
            
        except Exception as e:
            logging.error(f"Error processing {filename}: {e}")

    if db_conn.is_connected():
        db_conn.close()
        logging.info(f"Database connection closed. Total {successful_uploads_count} resumes processed and attempted to upload.")
        if successful_uploads_count > 0:
            logging.info("Data upload process completed.")
        else:
            logging.info("No data was successfully uploaded during this run.")

if __name__ == '__main__':
    main()