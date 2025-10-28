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
import socket

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
    
    assigned_to_nodename = Column(String, nullable=False) # The machine this experiment was run on
    status = Column(String, nullable=False) # pending/queued/completed/failed
    
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

def setup_experiments(models: list[str], n_trials: int = 10):
    """
    Populate the database with all experiment combinations as pending tasks.

    Args:
        models: List of model names to create experiments for
        n_trials: Number of repeated trials per question/formatter combination
    """
    # Get all question files
    question_files = sorted(glob.glob('./questions/*'))

    # Initialize formatters
    formatters = {
        'MultipleChoiceFormatter': MultipleChoiceFormatter(),
        'NumberedListFormatter': NumberedListFormatter(),
    }

    # Initialize database connection
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        raise Exception('No connection string in DATABASE_URL!')

    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        run_id = str(uuid.uuid4())
        print(f"Setting up experiment run: {run_id}")
        print(f"Models: {', '.join(models)}")
        print(f"Found {len(question_files)} question files")

        # Calculate total variations
        total_formatter_variations = sum(
            formatter.get_num_variations(4) for formatter in formatters.values()
        )
        print(f"Using {len(formatters)} formatters with {total_formatter_variations} total variations")
        print(f"Creating {n_trials} trials per combination per model\n")

        created_count = 0

        for model in models:
            for qa_file in tqdm(question_files, desc=f"Processing questions for {model}"):
                # Load question to get number of choices
                qa_data = BaseFormatter.load_question_file(qa_file)
                num_choices = len(qa_data['choices'])

                for formatter_name, formatter in formatters.items():
                    num_variations = formatter.get_num_variations(num_choices)

                    for formatter_seed in range(num_variations):
                        # Format the question once to get the formatted text
                        formatted_question, metadata = formatter.format(qa_data, formatter_seed)

                        for trial_num in range(n_trials):
                            # Create pending trial entry
                            trial_entry = TrialResult(
                                run_id=run_id,
                                timestamp=datetime.now(),
                                trial=trial_num,
                                attempt=0,
                                assigned_to_nodename='unassigned',
                                status='pending',
                                model=model,
                                qa_pair_json_fp=qa_file,
                                question_formatter_method_name=formatter_name,
                                question_formatter_method_seed=formatter_seed,
                                question_formatted=formatted_question,
                                thinking_mode='none',
                                answer_response_text=None,
                                answer_choice_parsed=None,
                            )
                            session.add(trial_entry)
                            created_count += 1

        session.commit()
        print(f"\n✓ Created {created_count} pending experiment entries")
        print(f"Run ID: {run_id}")

    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def run_experiments(model: str, device: str = None, max_workers: int = 4):
    """
    Worker function to pull and execute pending experiments from the database.

    Args:
        model: The Ollama model to use
        device: Device configuration for GPU usage
        max_workers: Number of parallel workers
    """
    # Get hostname for this worker
    hostname = socket.gethostname()

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
    if not db_url:
        raise Exception('No connection string in DATABASE_URL!')

    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    # Lock for database operations
    db_lock = threading.Lock()

    def execute_trial_from_db(trial_id):
        """Execute a single trial by its database ID"""
        # Create a new session for this thread
        session = Session()

        try:
            # Fetch the trial entry
            trial_entry = session.query(TrialResult).filter_by(id=trial_id).first()

            if not trial_entry:
                return

            # Get the formatter
            formatter = formatters[trial_entry.question_formatter_method_name]

            # The question is already formatted and stored
            formatted_question = trial_entry.question_formatted

            # Load question data to get metadata for answer mapping
            qa_data = BaseFormatter.load_question_file(trial_entry.qa_pair_json_fp)
            _, metadata = formatter.format(qa_data, trial_entry.question_formatter_method_seed)

            max_retries = 10
            response_text = None
            parsed_answer_position = None
            final_attempt = 0

            for attempt in range(max_retries):
                try:
                    response_text, parsed_answer_position = trial(
                        formatted_question=formatted_question,
                        model=model,
                        thinking_mode='none',
                        options=options if options else None
                    )
                    final_attempt = attempt
                    break  # Success, exit retry loop
                except Exception as e:
                    final_attempt = attempt
                    if attempt == max_retries - 1:
                        # Last attempt failed
                        print(f"Trial {trial_id} failed after {max_retries} attempts: {e}")

            # Map the parsed position back to original index
            answer_choice_parsed = None
            if parsed_answer_position is not None:
                answer_key = f'position_{parsed_answer_position}'
                answer_choice_parsed = metadata['answer_mapping'].get(answer_key)

            # Update the trial entry with results
            trial_entry.model = model
            trial_entry.attempt = final_attempt
            trial_entry.answer_response_text = response_text
            trial_entry.answer_choice_parsed = answer_choice_parsed
            trial_entry.timestamp = datetime.now()

            if response_text is not None:
                trial_entry.status = 'completed'
            else:
                trial_entry.status = 'failed'

            session.commit()

        except Exception as e:
            print(f"Error executing trial {trial_id}: {e}")
            session.rollback()
            # Mark as failed
            try:
                trial_entry = session.query(TrialResult).filter_by(id=trial_id).first()
                if trial_entry:
                    trial_entry.status = 'failed'
                    session.commit()
            except:
                pass
        finally:
            session.close()

    try:
        print(f"Worker starting on host: {hostname}")
        print(f"Model: {model}")
        if device:
            print(f"Device: {device}")

        # Get pending trials from database and claim them
        session = Session()

        with db_lock:
            # Find pending trials that are unassigned and match this worker's model
            pending_trials = session.query(TrialResult).filter_by(
                status='pending',
                assigned_to_nodename='unassigned',
                model=model
            ).all()

            print(f"\nFound {len(pending_trials)} pending trials for model '{model}'")

            if len(pending_trials) == 0:
                print("No pending trials to run for this model")
                return

            # Claim these trials for this worker
            trial_ids = []
            for trial_entry in pending_trials:
                trial_entry.assigned_to_nodename = hostname
                trial_entry.status = 'running'
                trial_ids.append(trial_entry.id)

            session.commit()
            print(f"Claimed {len(trial_ids)} trials for execution")

        session.close()

        print(f"Running with {max_workers} parallel workers\n")

        # Execute trials concurrently
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all trials
            futures = {executor.submit(execute_trial_from_db, trial_id): trial_id
                      for trial_id in trial_ids}

            # Create progress bar
            pbar = tqdm(total=len(trial_ids), desc="Running experiments")

            # Wait for completion
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Trial execution failed: {e}")
                finally:
                    pbar.update(1)

            pbar.close()

        print("\n✓ All claimed trials processed")

    except Exception as e:
        print(f"Error during experiment execution: {e}")
        raise

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='LLM multiple choice formatting experiments with Ollama models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup experiments (run this once to populate the database)
  python main.py setup --models qwen3:30b,llama3.2 --n-trials 10
  python main.py setup --models "qwen3:30b llama3.2" --n-trials 10

  # Run experiments as a worker (on different machines)
  python main.py run --model qwen3:30b --max-workers 2 --device 0
  python main.py run --model llama3.2 --max-workers 4
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup experiments by populating database with pending trials')
    setup_parser.add_argument('--models', type=str, required=True,
                             help='Comma or space-separated list of models (e.g., "qwen3:30b,llama3.2")')
    setup_parser.add_argument('--n-trials', type=int, default=10,
                             help='Number of repeated trials per question/formatter combination (default: 10)')

    # Run command
    run_parser = subparsers.add_parser('run', help='Run experiments as a worker')
    run_parser.add_argument('--model', type=str, required=True,
                           help='Ollama model to use (e.g., llama3.2, qwen3:30b)')
    run_parser.add_argument('--device', type=str, default=None,
                           help='Device configuration (e.g., GPU number or "cpu")')
    run_parser.add_argument('--max-workers', type=int, default=4,
                           help='Number of parallel workers for concurrent execution (default: 4)')

    args = parser.parse_args()

    if args.command == 'setup':
        # Parse models - support both comma and space separated
        models = [m.strip() for m in args.models.replace(',', ' ').split() if m.strip()]
        setup_experiments(models=models, n_trials=args.n_trials)
    elif args.command == 'run':
        run_experiments(model=args.model, device=args.device, max_workers=args.max_workers)
    else:
        parser.print_help()
