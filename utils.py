import io
import os
import re
import nltk
import spacy
import pandas as pd # Not currently used, but keep if planned for future
import docx2txt
import constants as cs
from spacy.matcher import Matcher
import PyPDF2 # New: Used for more robust PDF text extraction and page count (though ResumeParser now handles page count)
# from pdfminer.converter import TextConverter # Commenting out PDFMiner related imports
# from pdfminer.pdfinterp import PDFPageInterpreter
# from pdfminer.pdfinterp import PDFResourceManager
# from pdfminer.layout import LAParams
# from pdfminer.pdfpage import PDFPage
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from datetime import datetime, date # Import date for date.today()
from dateutil.relativedelta import relativedelta # For accurate date difference
import logging

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
# --- NLTK Downloads ---
# Ensure these are only downloaded if not found
for dependency in cs.NLTK_DEPENDENCIES:
    try:
        nltk.data.find(f'tokenizers/{dependency}') # For punkt
    except LookupError:
        try:
            nltk.data.find(f'corpora/{dependency}') # For wordnet, stopwords
        except LookupError:
            try:
                nltk.data.find(f'taggers/{dependency}') # For averaged_perceptron_tagger
            except LookupError:
                nltk.download(dependency)
                print(f"Downloaded NLTK '{dependency}' corpus.")
# --- End NLTK Downloads ---


# --- Text Extraction Functions ---
# NOTE: If ResumeParser's _extract_text_and_pages_from_pdf is the primary,
# this extract_text_from_pdf is redundant.
# I'll keep it simple here, but recommend centralizing PDF extraction in ResumeParser.
def extract_text_from_pdf(pdf_path):
    text_content = ""
    try:
        with open(pdf_path, 'rb') as fh:
            reader = PyPDF2.PdfReader(fh)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text_content += page.extract_text() or ''
    except Exception as e:
        print(f"Error extracting text from PDF with PyPDF2 in utils.py: {e}")
    return text_content.strip()


def extract_text_from_doc(doc_path):
    try:
        temp = docx2txt.process(doc_path)
        text = [line.replace('\t', ' ') for line in temp.split('\n') if line]
        return ' '.join(text).strip()
    except Exception as e:
        print(f"Error extracting text from DOC/DOCX: {e}")
        return ""

# This `extract_text` function might also be superseded if ResumeParser manages file type dispatch.
def extract_text(file_path, extension):
    text = ''
    if extension == '.pdf':
        text = extract_text_from_pdf(file_path)
    elif extension == '.docx' or extension == '.doc':
        text = extract_text_from_doc(file_path)
    return text

def extract_entity_sections(text):
    text_split = [line.strip() for line in text.split('\n') if line.strip()]
    entities = {}
    current_section = None

    for phrase in text_split:
        phrase_lower = phrase.lower()
        
        found_section_keyword = False
        for section_name in cs.RESUME_SECTIONS:
            # Check if the phrase starts with a section name or contains it strongly
            # Using re.search with word boundaries and a more flexible pattern
            if re.search(r'^(?:' + re.escape(section_name.lower()) + r')\s*$', phrase_lower) or \
               (re.search(r'\b' + re.escape(section_name.lower()) + r'\b', phrase_lower) and len(phrase.split()) < 6): # Heuristic for short section titles
                current_section = section_name
                entities[current_section] = []
                found_section_keyword = True
                break
        
        if not found_section_keyword and current_section:
            entities[current_section].append(phrase)
    
    return entities

def extract_email(text):
    email = re.findall(cs.EMAIL_REGEX, text) # Use regex from constants
    if email:
        try:
            return email[0]
        except IndexError:
            return None
    return None

def extract_name(nlp_text, matcher):
    patterns = []
    if isinstance(cs.NAME_PATTERN, list) and all(isinstance(p, dict) for p in cs.NAME_PATTERN):
        patterns.append(cs.NAME_PATTERN)
    elif isinstance(cs.NAME_PATTERN, list) and all(isinstance(p, list) for p in cs.NAME_PATTERN):
        patterns = cs.NAME_PATTERN
    else:
        print("Warning: cs.NAME_PATTERN not in expected format for Spacy Matcher.")
        return None

    # Clear existing patterns for 'NAME' to avoid re-adding
    if matcher.has_key('NAME'):
        matcher.remove('NAME') 
    matcher.add('NAME', patterns)
    
    matches = matcher(nlp_text)
    
    for match_id, start, end in matches:
        span = nlp_text[start:end]
        if span.text.strip():
            # Basic filtering for common non-name terms if needed
            if len(span.text.split()) > 1 and not re.search(r'^(summary|objective|skills|experience|education)$', span.text.lower()):
                 return span.text.strip()
    return None

def extract_mobile_number(text):
    phone_match = re.search(cs.PHONE_NUMBER_PATTERN, text)
    if phone_match:
        extracted_number = phone_match.group(0).strip()
        # Basic cleanup: remove spaces, hyphens, parentheses if it's a raw extract
        extracted_number = re.sub(r'[\s\-\(\)]', '', extracted_number)
        return extracted_number
    return None

def extract_skills(nlp_text, noun_chunks, skills_file):
    found_skills = set()
    skills_list = []
    if skills_file and os.path.exists(skills_file):
        with open(skills_file, 'r', encoding='utf-8', errors='ignore') as f: # Added errors='ignore'
            raw_lines = f.readlines()
            for line in raw_lines:
                for skill_part in line.strip().split(','):
                    clean_skill = skill_part.strip().lower()
                    if clean_skill:
                        skills_list.append(clean_skill)
        skills_list = list(set(skills_list)) # Ensure unique skills in list
            
    # Match against noun chunks (phrases)
    for chunk in noun_chunks:
        chunk_text = chunk.text.lower().strip()
        if chunk_text in skills_list:
            found_skills.add(chunk_text)
            
    # Match against individual tokens (single words)
    for token in nlp_text:
        token_text = token.text.lower().strip()
        if token_text in skills_list:
            found_skills.add(token_text)
    
    return list(found_skills)

# In utils.py

def extract_education(sentence_list):
    """
    Extracts education entries from a list of sentences, capturing the FULL degree name
    (e.g., "Bachelor of Engineering in Computer Science") while avoiding the university/date.
    """
    logging.debug(f"Entering extract_education. Sentence list count: {len(sentence_list)}")
    education_entries = set()
    
    # 1. Build the Degree Regex Pattern
    degree_keywords = r'(?:' + '|'.join(cs.EDUCATION_DEGREES_REGEX) + r')'
    
    # This regex looks for:
    # 1. The degree keyword (e.g., "Master", "B.Tech")
    # 2. An optional connector (e.g., "of", "in", "major in")
    # 3. The specialization text, keeping going UNTIL it hits a stop word like "from", "at", or a year.
    education_regex = re.compile(
        r'(?P<degree>'
            # Match the base degree (e.g., "Master", "Bachelor of Technology")
            + degree_keywords + 
            r'(?:'
                r'\s+' # Space after degree
                r'(?:in|of|with|major in|specialization in)?' # Optional connector
                r'\s+'
                # Lookahead: Ensure we don't start the specialization with "from" or "at"
                r'(?!from\b|at\b)'
                # Capture the Major/Specialization text (words, dots, parens)
                # STOP capturing if we hit: " from ", " at ", " - ", or a 4-digit Year
                r'(?:(?!(\s+from\b|\s+at\b|\s+[\-–]\s+|\s\(?\d{4})).)+'
            r')?'
        r')'
        
        # Capture University (Optional)
        r'(?:(?:\s+from|\s+at|[\-–])\s+(?P<university>[\w\s&.,\-]+(?:university|institute|college|school|academy)))?'
        
        # Capture Year (Optional)
        r'(?:[\s,\-|\(\)]+(?P<year_info>' + cs.YEAR_FOUR_DIGIT + r'(?:\s*[\-–]\s*(?:' + cs.YEAR_FOUR_DIGIT + r'|present|current|till date))?))?' 
        , re.IGNORECASE
    )

    for i, sentence in enumerate(sentence_list):
        sentence_lower = sentence.lower()
        # Optimization: Skip sentences without education keywords
        if not any(re.search(r'\b' + re.escape(kw) + r'\b', sentence_lower) for kw in cs.EDUCATION_KEYWORDS_LOWER):
            continue 

        match = education_regex.search(sentence)
        if match:
            # Extract groups
            degree = match.group('degree').strip()
            university = match.group('university')
            year_info = match.group('year_info')
            
            # Formatting: Ensure clean output
            parts = []
            if degree:
                # Remove trailing separators if the regex grabbed them (edge case cleanup)
                degree = re.sub(r'[,.\-]$', '', degree).strip()
                parts.append(degree)
                
            if university:
                university = university.strip()
                # Only add university if it's not already inside the degree string
                if not degree or university.lower() not in degree.lower():
                    parts.append(university)
                    
            if year_info:
                # Clean up year formatting
                year_info = re.sub(r'[()]', '', year_info).strip()
                parts.append(f"({year_info})")

            if parts:
                full_entry = ", ".join(parts)
                education_entries.add(full_entry)
                logging.debug(f"Structured education found: {full_entry}")

    logging.debug(f"Final extracted education: {list(education_entries)}")
    return list(education_entries)

# --- REVISED parse_experience_dates ---
def parse_experience_dates(experience_string):
    """
    Parses start and end dates from an experience string and calculates duration using relativedelta.
    Handles 'Present', 'Current', 'Till Date' as end dates.
    Returns start_date, end_date (datetime.date objects), total_duration_months.
    """
    logging.debug(f"Attempting to parse dates from: {experience_string}")

    start_date = None
    end_date = None
    total_duration_months = 0

    # Combined pattern to find a date range or single dates
    # This regex needs to capture the actual date strings correctly.
    date_range_pattern = re.compile(
        rf'(?P<start_date_str>'
            rf'(?:(?:{cs.MONTH}\s+)?{cs.YEAR_FOUR_DIGIT}|{cs.YEAR_FOUR_DIGIT})' # Month Year, or Year only
        r')'
        r'\s*[\-–]\s*' # Separator
        rf'(?P<end_date_str>'
            rf'(?:(?:{cs.MONTH}\s+)?{cs.YEAR_FOUR_DIGIT}|{cs.YEAR_FOUR_DIGIT})' # Month Year, or Year only
            r'|present|current|till date'
        r')',
        re.IGNORECASE
    )

    match = date_range_pattern.search(experience_string)
    if match:
        start_date_str = match.group('start_date_str').strip()
        end_date_str = match.group('end_date_str').strip()
        
        # Helper to parse a single date string (Month Year or Year)
        def parse_single_date(date_str, is_end_date=False):
            month = 1 # Default to January
            year = None
            
            month_match = re.search(cs.MONTH, date_str, re.IGNORECASE)
            year_match = re.search(cs.YEAR_FOUR_DIGIT, date_str, re.IGNORECASE)

            if year_match:
                year = int(year_match.group(0))
                if month_match:
                    month_name = month_match.group(0).lower()
                    try:
                        # Find index in either short or full months tuple
                        if month_name in cs.MONTHS_SHORT:
                            month = cs.MONTHS_SHORT.index(month_name) + 1
                        elif month_name in cs.MONTHS_LONG:
                            month = cs.MONTHS_LONG.index(month_name) + 1
                    except ValueError:
                        logging.warning(f"Could not parse month from '{month_name}' in '{date_str}'")
                
                try:
                    dt = datetime(year, month, 1).date()
                    if is_end_date:
                        # For end dates, consider the end of the month for duration calculation
                        dt += relativedelta(months=1, days=-1)
                    return dt
                except ValueError as e:
                    logging.error(f"Error creating date from '{date_str}': {e}")
            return None

        # Parse start date
        start_date = parse_single_date(start_date_str, is_end_date=False)
        
        # Parse end date
        if end_date_str.lower() in ['present', 'current', 'till date']:
            end_date = date.today()
        else:
            end_date = parse_single_date(end_date_str, is_end_date=True)

    if start_date and end_date and end_date >= start_date:
        duration = relativedelta(end_date, start_date)
        # Calculate months more accurately, considering partial months
        total_duration_months = duration.years * 12 + duration.months
        # Add a month if there's significant day difference (e.g., if end_date is more than 15 days into the month after start_date's month)
        # This is a heuristic. A simpler approach might just be to round.
        if duration.days >= 15: # If more than half a month's difference
            total_duration_months += 1
        
        logging.debug(f"Dates found: {start_date} to {end_date}. Duration: {total_duration_months} months.")
        return start_date, end_date, total_duration_months
    
    logging.debug(f"No valid date range found for: {experience_string}")
    return None, None, 0


# --- REVISED extract_experience ---
def extract_experience(resume_text):
    '''
    Extracts experience entries, including dates, from resume text.
    Focuses on identifying sections and then lines within those sections.
    '''
    extracted_experiences = []
    text_lower = resume_text.lower()
    
    # Identify the start of the experience section
    experience_section_start = -1
    for section in cs.EXPERIENCE_KEYWORDS + ['experience']: # Use a broader set of keywords
        start_match = re.search(r'\b' + re.escape(section.lower()) + r'\b', text_lower)
        if start_match:
            experience_section_start = start_match.end()
            logging.debug(f"Found experience section start: '{section}' at {experience_section_start}")
            break
    
    if experience_section_start == -1:
        logging.debug("Could not find an explicit 'Experience' section header.")
        return []

    # Identify the end of the experience section (start of the next major section)
    experience_section_end = len(text_lower) # Default to end of document
    
    # Look for the next section after experience
    # Prioritize common sections that usually follow experience
    following_sections = ['education', 'projects', 'skills', 'awards', 'publications', 'interests']
    
    for section_name in following_sections:
        # Search from after the experience section start
        end_match = re.search(r'\b' + re.escape(section_name.lower()) + r'\b', text_lower[experience_section_start:])
        if end_match:
            experience_section_end = experience_section_start + end_match.start()
            logging.debug(f"Found experience section end: '{section_name}' at {experience_section_end}")
            break
    
    experience_section_text = resume_text[experience_section_start:experience_section_end].strip()
    logging.debug(f"Experience Section Text identified (first 500 chars):\n{experience_section_text[:500]}...")

    # Now, parse lines within this identified experience section
    # Looking for lines that likely contain a job title/company AND a date range
    
    # This pattern is for identifying a date range within a line, to help filter
    date_range_in_line_pattern = re.compile(
        rf'(?:(?:{cs.MONTH}\s+)?{cs.YEAR_FOUR_DIGIT}|{cs.YEAR_FOUR_DIGIT})'  # Start Date (Month Year or Year)
        r'\s*[\-–]\s*' # Separator
        rf'(?:(?:{cs.MONTH}\s+)?{cs.YEAR_FOUR_DIGIT}|{cs.YEAR_FOUR_DIGIT}|present|current|till date)', # End Date (Month Year, Year, or keywords)
        re.IGNORECASE
    )

    for line in experience_section_text.split('\n'):
        line_stripped = line.strip()
        if not line_stripped:
            continue
        
        # Heuristic: A job line usually has multiple capitalized words (job title/company)
        # and definitely contains a date range.
        # Add filtering for lines that are too short to be an experience entry or clearly bullet points
        if len(line_stripped.split()) < 4 or line_stripped.startswith(('-', '•', '–')): # Filter out bullet points that are not full entries
            continue
            
        # Check if the line contains a date range
        if date_range_in_line_pattern.search(line_stripped):
            # Check for at least one capitalized word (for job title/company)
            if re.search(r'[A-Z][a-z]+', line_stripped):
                extracted_experiences.append(line_stripped)
                logging.debug(f"Extracted experience line: {line_stripped}")
    
    if not extracted_experiences:
        logging.warning("No structured experience entries found using specific line patterns. Trying a broader approach.")
        # Fallback: if no structured lines found, just grab any line that has a date range
        # This can be noisy but might catch some formats.
        for line in experience_section_text.split('\n'):
            line_stripped = line.strip()
            if date_range_in_line_pattern.search(line_stripped):
                extracted_experiences.append(line_stripped)

    # Filter out empty strings and return unique entries
    return list(set([exp for exp in extracted_experiences if exp]))


def extract_total_experience(experience_list):
    '''
    Calculates total experience from a list of experience strings.
    '''
    total_months_sum = 0
    logging.debug(f"Entering extract_total_experience. Experience list count: {len(experience_list)}")

    for exp_text in experience_list:
        logging.debug(f"Processing experience text for total duration: {exp_text}")
        _, _, duration_months = parse_experience_dates(exp_text)
        total_months_sum += duration_months

    total_years = round(total_months_sum / 12, 2)
    logging.debug(f"Total experience calculated: {total_years:.2f} years")
    return total_years

def extract_competencies(text): # Removed unused experience_list parameter
    competency_dict = {}
    for competency_type in cs.COMPETENCIES.keys():
        competency_dict[competency_type] = []
        for item in cs.COMPETENCIES[competency_type]:
            if string_found(item, text):
                competency_dict[competency_type].append(item)
    return competency_dict

def extract_measurable_results(text): # Removed unused experience_list parameter
    mr_dict = {}
    for mr_type in cs.MEASURABLE_RESULTS.keys():
        mr_dict[mr_type] = []
        for item in cs.MEASURABLE_RESULTS[mr_type]:
            if string_found(item, text):
                mr_dict[mr_type].append(item)
    return mr_dict

def string_found(string1, string2):
    if re.search(r"\b" + re.escape(string1) + r"\b", string2, re.IGNORECASE):
        return True
    return False