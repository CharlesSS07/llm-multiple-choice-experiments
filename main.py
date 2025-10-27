#!/usr/bin/env python3
"""
Experiment script for testing LLM responses to multiple choice questions with various formatting.
"""

import ollama
from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from tqdm import tqdm
from datetime import datetime
import uuid
import re
import json
import glob
import os

from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from formatters import MultipleChoiceFormatter, NumberedListFormatter, BaseFormatter

load_dotenv()

# Define database schema
Base = declarative_base()

class TrialResultBase(Base):
    __abstract__ = True

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    trial = Column(Integer, nullable=False)
    attempt = Column(Integer, nullable=False)

class TrialResult(TrialResultBase):
    __tablename__ = 'llm_multiple_choice_experiment_results'

    model = Column(String)
    qa_pair_json_fp = Column(String) # Question and Answers stored as JSON (model)
    question_formatter_method_name = Column(String)  # Method used to view the qa pair (view)
    question_formatter_method_seed = Column(Integer) # Method seed (sub-method) used to view the qa pair (things like all caps, answer order, etc.)
    question_formatted = Column(String) # the formatted text representing the question
    thinking_mode = Column(String) # the chosen method for how to structure "thinking" (no thinking, thinking=True (CoT or harmony), think out loud, etc.)
    answer_response_text = Column(Text) # the text that came out of the LLM
    answer_choice_parsed = Column(Integer) # the answer choice demarked by the answers index in the json file (for ease of use)
    # answer_choice_other = Column(Text) # most questions should include an "Other" answer, with manadory explaination
    # answer_explaination = Column(Text) # opprotunity for the LLM to explain it's answer

def trial_exists(session, qa_pair_json_fp: str, question_formatter_method_name: str,
                 question_formatter_method_seed: int, model: str, trial_num: int) -> bool:
    """
    Check if a trial with the given parameters already exists in the database.

    Args:
        session: SQLAlchemy session
        qa_pair_json_fp: Path to the question/answer JSON file
        question_formatter_method_name: Name of the formatter method used
        question_formatter_method_seed: Seed for the formatter variation
        model: The model name used
        trial_num: The trial number

    Returns:
        True if a trial with these parameters exists, False otherwise
    """
    result = session.query(TrialResult).filter_by(
        qa_pair_json_fp=qa_pair_json_fp,
        question_formatter_method_name=question_formatter_method_name,
        question_formatter_method_seed=question_formatter_method_seed,
        model=model,
        trial=trial_num
    ).first()

    return result is not None


def trial(formatted_question: str, model: str, thinking_mode: str = 'none', options: dict = None) -> tuple[str, int | None]:
    """
    Send a formatted multiple choice question to the LLM and get its response.

    Args:
        formatted_question: The formatted question prompt
        model: The Ollama model to use
        thinking_mode: How to structure thinking ('none' for now)
        options: Optional configuration options for Ollama

    Returns:
        A tuple of (response_text, parsed_answer_index) where:
        - response_text is the LLM's raw response as a string
        - parsed_answer_index is the chosen answer index (0-based) or None if unable to parse

    Raises:
        Exception: If the API call fails
    """
    # Prepare chat kwargs
    chat_kwargs = {
        'model': model,
        'messages': [{'role': 'user', 'content': formatted_question}]
    }

    if options:
        chat_kwargs['options'] = options

    # Make the API call
    response = ollama.chat(**chat_kwargs)
    response_text = response['message']['content'].strip()

    # Try to parse which answer was chosen
    # Look for patterns like "A", "a)", "(1)", "1.", etc.
    parsed_answer = None
    # This is a simple parser - could be made more sophisticated
    # For now, just look for single letter or number at start of response
    first_line = response_text.split('\n')[0].strip()

    # Try to match letter-based answers (A, B, C, etc.)
    letter_match = re.match(r'^[(\[]?([A-Za-z])[)\].\s:)]?', first_line)
    if letter_match:
        letter = letter_match.group(1).upper()
        parsed_answer = ord(letter) - ord('A')
    else:
        # Try to match number-based answers (1, 2, 3, etc.)
        number_match = re.match(r'^[(\[#]?(\d+)[)\].\s:-]?', first_line)
        if number_match:
            parsed_answer = int(number_match.group(1)) - 1

    return response_text, parsed_answer

def experiment(model: str, device: str = None, max_workers: int = 4, n_trials: int = 10):
    """
    Run the LLM multiple choice experiment.

    Args:
        model: The Ollama model to use
        device: Device configuration for GPU usage
        max_workers: Number of parallel workers
        n_trials: Number of repeated trials per question/formatter combination
    """

    # Get all question files
    question_files = sorted(glob.glob('./questions/*'))

    # Initialize formatters
    formatters = {
        'MultipleChoiceFormatter': MultipleChoiceFormatter(),
        'NumberedListFormatter': NumberedListFormatter(),
    }

    # Configure Ollama options for device
    options = {}
    if device:
        options['num_gpu'] = int(device) if device.isdigit() else 1

    # Initialize database connection
    db_url = os.getenv('DATABASE_URL')
    if db_url:
        engine = create_engine(db_url)
    else:
        raise Exception('No connection string in DATABASE_URL!')

    # Create tables if they don't exist
    Base.metadata.create_all(engine)

    # Create session factory
    Session = sessionmaker(bind=engine)
    session = Session()

    # Lock for database writes
    db_lock = threading.Lock()

    # Cache for loaded question files
    question_cache = {}
    cache_lock = threading.Lock()

    def load_question(filepath):
        """Load question file with caching"""
        if filepath not in question_cache:
            with cache_lock:
                if filepath not in question_cache:
                    question_cache[filepath] = BaseFormatter.load_question_file(filepath)
        return question_cache[filepath]

    def prepare_and_execute_trial(trial_params):
        """Format question and execute a single trial"""
        (qa_file, formatter_name, formatter_seed, trial_num, run_id) = trial_params

        # Load question data from cache
        qa_data = load_question(qa_file)

        # Get the formatter
        formatter = formatters[formatter_name]

        # Format the question
        formatted_question, metadata = formatter.format(qa_data, formatter_seed)

        max_retries = 10
        response_text = None
        parsed_answer_position = None  # Position in formatted choices
        attempt = 0

        for attempt in range(max_retries):
            try:
                response_text, parsed_answer_position = trial(
                    formatted_question=formatted_question,
                    model=model,
                    thinking_mode='none',
                    options=options if options else None
                )
                break  # Success, exit retry loop
            except Exception as e:
                print(e)
                print(f'Trial raised exception. Retrying. Attempt {attempt+1} out of {max_retries}.')

        # Map the parsed position back to original index
        answer_choice_parsed = None
        if parsed_answer_position is not None:
            answer_key = f'position_{parsed_answer_position}'
            answer_choice_parsed = metadata['answer_mapping'].get(answer_key)

        # Return all data needed to create DB record
        return {
            'run_id': run_id,
            'timestamp': datetime.now(),
            'trial': trial_num,
            'attempt': attempt,
            'model': model,
            'qa_pair_json_fp': qa_file,
            'question_formatter_method_name': formatter_name,
            'question_formatter_method_seed': formatter_seed,
            'question_formatted': formatted_question,
            'thinking_mode': 'none',
            'answer_response_text': response_text,
            'answer_choice_parsed': answer_choice_parsed,
        }

    try:
        # Generate a unique experiment ID for this run
        run_id = str(uuid.uuid4())
        print(f"Starting experiment run: {run_id}")
        print(f"Model: {model}")
        if device:
            print(f"Device: {device}")

        # First pass: collect trial parameters
        print("Collecting trial parameters...")
        trial_params_list = []

        # Calculate total number of formatter variations (assume 4 choices for now)
        total_formatter_variations = sum(
            formatter.get_num_variations(4) for formatter in formatters.values()
        )

        print(f"Found {len(question_files)} question files")
        print(f"Using {len(formatters)} formatters with {total_formatter_variations} total variations")
        print(f"Will run {n_trials} trials per combination")

        for qa_file in question_files:
            # Load the question to get number of choices
            qa_data = load_question(qa_file)
            num_choices = len(qa_data['choices'])

            for formatter_name, formatter in formatters.items():
                num_variations = formatter.get_num_variations(num_choices)

                for formatter_seed in range(num_variations):
                    for trial_num in range(n_trials):
                        # Check if this trial already exists
                        if trial_exists(session, qa_file, formatter_name, formatter_seed, model, trial_num):
                            continue

                        # Add trial parameters
                        trial_params_list.append((
                            qa_file, formatter_name, formatter_seed, trial_num, run_id
                        ))

        total_possible = len(question_files) * total_formatter_variations * n_trials
        print(f"\nFound {len(trial_params_list)} trials to run (skipped {total_possible - len(trial_params_list)} existing)")
        print(f"Running with {max_workers} parallel workers\n")

        # Second pass: execute trials concurrently and write to DB as they complete
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all trials
                future_to_params = {executor.submit(prepare_and_execute_trial, params): params
                                   for params in trial_params_list}

                # Create progress bar
                pbar = tqdm(total=len(trial_params_list), desc="Running experiments")

                # Process and save results as they complete
                for future in as_completed(future_to_params):
                    try:
                        result_data = future.result()

                        # Write to database immediately (with lock to prevent concurrent writes)
                        with db_lock:
                            result = TrialResult(**result_data)
                            session.add(result)
                            session.commit()

                    except Exception as e:
                        print(f"Trial failed with exception: {e}")
                        # Rollback on error
                        with db_lock:
                            session.rollback()
                    finally:
                        pbar.update(1)

                pbar.close()

            print("All trials completed and written to database")

        except Exception as e:
            print(f"Error during experiment execution: {e}")
            raise

    except Exception as e:
        # Rollback the session on error
        session.rollback()
        raise e

    finally:
        # Always close the session
        session.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run LLM multiple choice formatting experiments with Ollama models')
    parser.add_argument('--model', type=str, required=True, help='Ollama model to use (e.g., llama3.2, gpt-oss:20b)')
    parser.add_argument('--device', type=str, default=None, help='Device configuration (e.g., GPU number or "cpu")')
    parser.add_argument('--max-workers', type=int, default=4, help='Number of parallel workers for concurrent execution (default: 4)')
    parser.add_argument('--n-trials', type=int, default=10, help='Number of repeated trials per question/formatter combination (default: 10)')

    args = parser.parse_args()

    experiment(model=args.model, device=args.device, max_workers=args.max_workers, n_trials=args.n_trials)
