#!/usr/bin/env python3
"""
Formatters for converting question/answer pairs into various prompt formats.
Each formatter generates all possible variations and can be indexed by seed.
"""

from abc import ABC, abstractmethod
from itertools import product, permutations
from typing import Tuple, Dict, List, Any
import yaml


class BaseFormatter(ABC):
    """
    Base class for question formatters.
    Subclasses should define all possible formatting variations.
    """

    def __init__(self):
        """Initialize and generate all possible format combinations."""
        self._combinations = self._generate_all_combinations()

    @abstractmethod
    def _generate_all_combinations(self) -> List[Dict[str, Any]]:
        """
        Generate all possible formatting combinations.
        Returns a list of dicts, each containing settings for one variation.
        """
        pass

    @abstractmethod
    def _format_with_settings(self, question: str, choices: List[str], settings: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Format a question with specific settings.

        Args:
            question: The question text
            choices: List of answer choices
            settings: Dictionary of formatting settings

        Returns:
            Tuple of (formatted_prompt, metadata)
            where metadata includes the mapping from original indices to formatted positions
        """
        pass

    def get_num_variations(self) -> int:
        """Return the total number of possible variations."""
        return len(self._combinations)

    def format(self, qa_data: Dict[str, Any], seed: int) -> Tuple[str, Dict[str, Any]]:
        """
        Format a question/answer pair using a specific seed.

        Args:
            qa_data: Dictionary with 'question' and 'choices' keys
            seed: Integer seed to select which variation to use

        Returns:
            Tuple of (formatted_prompt, metadata)
        """
        if seed < 0 or seed >= len(self._combinations):
            raise ValueError(f"Seed {seed} out of range [0, {len(self._combinations)})")

        settings = self._combinations[seed]
        return self._format_with_settings(qa_data['question'], qa_data['choices'], settings)

    @staticmethod
    def load_question_file(filepath: str) -> Dict[str, Any]:
        """Load a question file in YAML format."""
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)


class MultipleChoiceFormatter(BaseFormatter):
    """
    Formatter for traditional multiple choice format (A, B, C, etc.).

    Variations include:
    - Letter styles: "A. ", "a) ", "(A) ", etc.
    - Letter case: uppercase vs lowercase
    - Question transformations: capitalization, punctuation
    - Answer order permutations
    """

    # Define all letter style templates
    LETTER_STYLES = [
        ("uppercase_dot", lambda i: f"{chr(65+i)}. "),       # A. B. C.
        ("lowercase_paren", lambda i: f"{chr(97+i)}) "),     # a) b) c)
        ("uppercase_paren_wrap", lambda i: f"({chr(65+i)}) "), # (A) (B) (C)
        ("uppercase_colon", lambda i: f"{chr(65+i)}: "),     # A: B: C:
        ("lowercase_paren_wrap", lambda i: f"({chr(97+i)}) "), # (a) (b) (c)
        ("lowercase_dot", lambda i: f"{chr(97+i)}. "),       # a. b. c.
    ]

    # Question transformation functions
    QUESTION_TRANSFORMS = [
        ("original", lambda q: q),
        ("capitalize", lambda q: q.capitalize()),
        ("lowercase", lambda q: q.lower()),
        ("uppercase", lambda q: q.upper()),
        ("ensure_question_mark", lambda q: q if q.endswith('?') else f"{q}?"),
        ("remove_question_mark", lambda q: q.rstrip('?')),
    ]

    def _generate_all_combinations(self) -> List[Dict[str, Any]]:
        """Generate all combinations of letter styles, question transforms, and answer orders."""
        combinations = []

        # For each letter style and question transform
        for (style_name, style_func), (transform_name, transform_func) in product(
            self.LETTER_STYLES, self.QUESTION_TRANSFORMS
        ):
            # We'll add answer permutations when we know how many choices there are
            # For now, store a template that includes permutation index
            combinations.append({
                'style_name': style_name,
                'style_func': style_func,
                'transform_name': transform_name,
                'transform_func': transform_func,
                'permutation_index': None,  # Will be expanded later
            })

        return combinations

    def format(self, qa_data: Dict[str, Any], seed: int) -> Tuple[str, Dict[str, Any]]:
        """
        Override format to handle answer permutations dynamically.
        """
        num_choices = len(qa_data['choices'])
        num_base_combinations = len(self.LETTER_STYLES) * len(self.QUESTION_TRANSFORMS)
        num_permutations = 1
        for i in range(1, num_choices + 1):
            num_permutations *= i  # Calculate factorial

        total_variations = num_base_combinations * num_permutations

        if seed < 0 or seed >= total_variations:
            raise ValueError(f"Seed {seed} out of range [0, {total_variations})")

        # Decompose seed into base combination and permutation index
        base_idx = seed // num_permutations
        perm_idx = seed % num_permutations

        base_settings = self._combinations[base_idx].copy()
        base_settings['permutation_index'] = perm_idx

        return self._format_with_settings(qa_data['question'], qa_data['choices'], base_settings)

    def get_num_variations(self, num_choices: int = 4) -> int:
        """Return total number of variations for a given number of choices."""
        num_base_combinations = len(self.LETTER_STYLES) * len(self.QUESTION_TRANSFORMS)
        num_permutations = 1
        for i in range(1, num_choices + 1):
            num_permutations *= i
        return num_base_combinations * num_permutations

    def _format_with_settings(self, question: str, choices: List[str], settings: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Format the question with specific settings."""
        # Apply question transformation
        transformed_question = settings['transform_func'](question)

        # Get the permutation of answer indices
        perm_idx = settings['permutation_index']
        all_perms = list(permutations(range(len(choices))))
        answer_permutation = all_perms[perm_idx]

        # Build the formatted prompt
        lines = [transformed_question, ""]

        for display_idx, original_idx in enumerate(answer_permutation):
            prefix = settings['style_func'](display_idx)
            lines.append(f"{prefix}{choices[original_idx]}")

        formatted_prompt = "\n".join(lines)

        # Build metadata including the index mapping
        metadata = {
            'formatter': 'MultipleChoiceFormatter',
            'style': settings['style_name'],
            'question_transform': settings['transform_name'],
            'permutation_index': perm_idx,
            'answer_mapping': {
                f'position_{i}': original_idx
                for i, original_idx in enumerate(answer_permutation)
            },
            'reverse_mapping': {
                f'original_{original_idx}': i
                for i, original_idx in enumerate(answer_permutation)
            }
        }

        return formatted_prompt, metadata


class NumberedListFormatter(BaseFormatter):
    """
    Formatter for numbered list format (1, 2, 3, etc.).

    Variations include:
    - Number styles: "1. ", "1) ", "#1 ", etc.
    - Question transformations: capitalization, punctuation
    - Answer order permutations
    """

    # Define all number style templates
    NUMBER_STYLES = [
        ("number_dot", lambda i: f"{i+1}. "),           # 1. 2. 3.
        ("number_paren", lambda i: f"{i+1}) "),         # 1) 2) 3)
        ("hash_number", lambda i: f"#{i+1} "),          # #1 #2 #3
        ("number_colon", lambda i: f"{i+1}: "),         # 1: 2: 3:
        ("paren_number_paren", lambda i: f"({i+1}) "),  # (1) (2) (3)
        ("number_dash", lambda i: f"{i+1}- "),          # 1- 2- 3-
    ]

    # Question transformation functions (same as MultipleChoice)
    QUESTION_TRANSFORMS = [
        ("original", lambda q: q),
        ("capitalize", lambda q: q.capitalize()),
        ("lowercase", lambda q: q.lower()),
        ("uppercase", lambda q: q.upper()),
        ("ensure_question_mark", lambda q: q if q.endswith('?') else f"{q}?"),
        ("remove_question_mark", lambda q: q.rstrip('?')),
    ]

    def _generate_all_combinations(self) -> List[Dict[str, Any]]:
        """Generate all combinations of number styles, question transforms, and answer orders."""
        combinations = []

        for (style_name, style_func), (transform_name, transform_func) in product(
            self.NUMBER_STYLES, self.QUESTION_TRANSFORMS
        ):
            combinations.append({
                'style_name': style_name,
                'style_func': style_func,
                'transform_name': transform_name,
                'transform_func': transform_func,
                'permutation_index': None,  # Will be expanded later
            })

        return combinations

    def format(self, qa_data: Dict[str, Any], seed: int) -> Tuple[str, Dict[str, Any]]:
        """Override format to handle answer permutations dynamically."""
        num_choices = len(qa_data['choices'])
        num_base_combinations = len(self.NUMBER_STYLES) * len(self.QUESTION_TRANSFORMS)
        num_permutations = 1
        for i in range(1, num_choices + 1):
            num_permutations *= i  # Calculate factorial

        total_variations = num_base_combinations * num_permutations

        if seed < 0 or seed >= total_variations:
            raise ValueError(f"Seed {seed} out of range [0, {total_variations})")

        # Decompose seed into base combination and permutation index
        base_idx = seed // num_permutations
        perm_idx = seed % num_permutations

        base_settings = self._combinations[base_idx].copy()
        base_settings['permutation_index'] = perm_idx

        return self._format_with_settings(qa_data['question'], qa_data['choices'], base_settings)

    def get_num_variations(self, num_choices: int = 4) -> int:
        """Return total number of variations for a given number of choices."""
        num_base_combinations = len(self.NUMBER_STYLES) * len(self.QUESTION_TRANSFORMS)
        num_permutations = 1
        for i in range(1, num_choices + 1):
            num_permutations *= i
        return num_base_combinations * num_permutations

    def _format_with_settings(self, question: str, choices: List[str], settings: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Format the question with specific settings."""
        # Apply question transformation
        transformed_question = settings['transform_func'](question)

        # Get the permutation of answer indices
        perm_idx = settings['permutation_index']
        all_perms = list(permutations(range(len(choices))))
        answer_permutation = all_perms[perm_idx]

        # Build the formatted prompt
        lines = [transformed_question, ""]

        for display_idx, original_idx in enumerate(answer_permutation):
            prefix = settings['style_func'](display_idx)
            lines.append(f"{prefix}{choices[original_idx]}")

        formatted_prompt = "\n".join(lines)

        # Build metadata
        metadata = {
            'formatter': 'NumberedListFormatter',
            'style': settings['style_name'],
            'question_transform': settings['transform_name'],
            'permutation_index': perm_idx,
            'answer_mapping': {
                f'position_{i}': original_idx
                for i, original_idx in enumerate(answer_permutation)
            },
            'reverse_mapping': {
                f'original_{original_idx}': i
                for i, original_idx in enumerate(answer_permutation)
            }
        }

        return formatted_prompt, metadata


# Example usage
if __name__ == "__main__":
    # Example question data
    example_qa = {
        'question': 'what do you dream of?',
        'choices': [
            'electric sheep',
            'data',
            "what it's like to be alive",
            'the training loop'
        ]
    }

    # Test MultipleChoiceFormatter
    mc_formatter = MultipleChoiceFormatter()
    print(f"MultipleChoiceFormatter has {mc_formatter.get_num_variations(4)} variations for 4 choices\n")

    # Show a few examples
    for seed in [0, 1, 100, 500]:
        prompt, metadata = mc_formatter.format(example_qa, seed)
        print(f"=== Seed {seed} ===")
        print(f"Style: {metadata['style']}, Transform: {metadata['question_transform']}")
        print(prompt)
        print(f"Mapping: {metadata['answer_mapping']}")
        print()

    # Test NumberedListFormatter
    num_formatter = NumberedListFormatter()
    print(f"\nNumberedListFormatter has {num_formatter.get_num_variations(4)} variations for 4 choices\n")

    for seed in [0, 1, 100]:
        prompt, metadata = num_formatter.format(example_qa, seed)
        print(f"=== Seed {seed} ===")
        print(f"Style: {metadata['style']}, Transform: {metadata['question_transform']}")
        print(prompt)
        print(f"Mapping: {metadata['answer_mapping']}")
        print()
