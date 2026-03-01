
import sys
import os
import spacy
import pprint
from spacy.matcher import Matcher
import multiprocessing as mp
import PyPDF2
import textract
import docx2txt
import io 
import logging 

from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

import utils


class ResumeParser(object):
    def __init__(self, resume, skills_file=None):
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            logging.info("Downloading spacy model 'en_core_web_sm'...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load('en_core_web_sm')

        self.__matcher = Matcher(self.nlp.vocab)

        self.__details = {
            'name'              : None,
            'email'             : None,
            'mobile_number'     : None,
            'skills'            : None,
            'education'         : None,
            'experience'        : None,
            'competencies'      : None,
            'measurable_results': None,
            'total_experience'  : 0.0,
            'no_of_pages'       : 0,
            'extracted_text'    : '' 
        }
        self.__resume      = resume
        self.__skills_file = skills_file
        
        self.__text_raw, self.__details['no_of_pages'] = self._extract_text_and_pages()
        self.__details['extracted_text'] = self.__text_raw
        self.__text        = ' '.join(self.__text_raw.split())
        
        if not self.__text:
            logging.warning(f"No text extracted from {self.__resume}. Skipping NLP processing.")
            self.__nlp_doc = None
            self.__noun_chunks = []
        else:
            self.__nlp_doc     = self.nlp(self.__text)
            self.__noun_chunks = list(self.__nlp_doc.noun_chunks)

        self.__get_basic_details()

    def _extract_text_and_pages(self):
        """
        Extracts text and page count from the resume, handling different file types.
        """
        text = ''
        num_pages = 0
        extension = os.path.splitext(self.__resume)[1].lower()

        if extension == '.pdf':
            try:
                with open(self.__resume, 'rb') as pdf_file:
                    reader = PyPDF2.PdfReader(pdf_file)
                    num_pages = len(reader.pages)
                    for page_num in range(num_pages):
                        page = reader.pages[page_num]
                        text += page.extract_text() or ''
                logging.debug(f"Extracted {num_pages} pages and text from PDF: {self.__resume}")
            except Exception as e:
                logging.error(f"Error extracting text from PDF {self.__resume}: {e}")
                num_pages = 0
        elif extension in ('.doc', '.docx'): # Use textract for both .doc and .docx
            try:
               
                extracted_bytes = textract.process(self.__resume)
                text = extracted_bytes.decode('utf-8', errors='ignore')
                num_pages = 1
                logging.debug(f"Extracted text from {extension.upper()}: {self.__resume}")
            except Exception as e:
                logging.error(f"Error extracting text from {extension.upper()} {self.__resume}: {e}. Make sure required system tools (e.g., antiword for .doc) are installed and in PATH.")
                num_pages = 0
        else:
            logging.warning(f"Unsupported file type: {extension} for {self.__resume}")
            num_pages = 0

        return text, num_pages

    def get_extracted_data(self):
        return self.__details

    def __get_basic_details(self):
        if not self.__nlp_doc:
            logging.warning(f"Skipping detail extraction for {self.__resume} due to no text.")
            return

        name = utils.extract_name(self.__nlp_doc, matcher=self.__matcher)
        email      = utils.extract_email(self.__text)
        mobile     = utils.extract_mobile_number(self.__text)
        
        skills     = utils.extract_skills(self.__nlp_doc, self.__noun_chunks, self.__skills_file)
        
        edu = utils.extract_education([sent.text.strip() for sent in self.__nlp_doc.sents])
        
        experience = utils.extract_experience(self.__text)
        
        total_experience = utils.extract_total_experience(experience)
        
        entities   = utils.extract_entity_sections(self.__text_raw)
        
        self.__details['name'] = name
        self.__details['email'] = email
        self.__details['mobile_number'] = mobile
        self.__details['skills'] = skills
        self.__details['education'] = edu
        self.__details['experience'] = experience
        self.__details['total_experience'] = total_experience

        try:
            if 'experience' in entities and entities['experience']:
         
                self.__details['competencies'] = utils.extract_competencies(self.__text_raw) 
                self.__details['measurable_results'] = utils.extract_measurable_results(self.__text_raw) 
            else:
                self.__details['competencies'] = []
                self.__details['measurable_results'] = []
        except KeyError:
            self.__details['competencies'] = []
            self.__details['measurable_results'] = []
        return

def resume_result_wrapper(resume, skills_file=None):
    parser = ResumeParser(resume, skills_file=skills_file)
    return parser.get_extracted_data()

if __name__ == '__main__':
    pool = mp.Pool(mp.cpu_count())

    resumes = []
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    test_files_dir_for_main = os.path.join(current_script_dir, 'resume_parse', 'test_files')

    default_skills_file = os.path.join(current_script_dir, 'resume_parse', 'skills.txt')


    for root, directories, filenames in os.walk(test_files_dir_for_main):
        for filename in filenames:
            if filename.lower().endswith(('.pdf', '.doc', '.docx')):
                file_path = os.path.join(root, filename)
                resumes.append(file_path)

    results = [pool.apply_async(resume_result_wrapper, args=(x, default_skills_file)) for x in resumes]

    results = [p.get() for p in results]

    pprint.pprint(results)