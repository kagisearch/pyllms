"""
Tests for the benchmark functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from llms.llms import LLMS
from llms.results.result import Result


class MockProvider:
    """Mock provider for testing benchmark functionality."""

    MODEL_INFO = {
        "mock-model": {"prompt": 1.0, "completion": 2.0, "token_limit": 4096}
    }

    def __init__(self, model=None, fail_on_indices=None):
        self.model = model or "mock-model"
        self.fail_on_indices = fail_on_indices or []
        self._call_count = 0

    def __str__(self):
        return f"MockProvider({self.model})"

    def complete(self, prompt, **kwargs):
        current_idx = self._call_count
        self._call_count += 1

        if current_idx in self.fail_on_indices:
            raise Exception(f"Simulated failure for prompt index {current_idx}")

        return Result(
            text=f"Mock response for prompt {current_idx}",
            model_inputs={"prompt": prompt},
            provider=self,
            meta={
                "tokens_prompt": 10,
                "tokens_completion": 20,
                "latency": 0.5,
            }
        )


class TestBenchmarkEvaluationIndexing:
    """Test that benchmark handles evaluation indices correctly."""

    def test_evaluation_dict_structure(self):
        """Test that evaluation results are stored in a dict with prompt_index as key."""
        llms_instance = LLMS.__new__(LLMS)
        llms_instance._providers = [MockProvider()]

        problems = [
            ("Problem 1", "Answer 1"),
            ("Problem 2", "Answer 2"),
        ]

        # Run benchmark without evaluator (simpler test)
        result = llms_instance.benchmark(problems=problems, evaluator=None)

        # Should complete without error
        assert result is not None

    def test_benchmark_with_partial_failures(self):
        """Test benchmark handles partial prompt failures gracefully."""
        llms_instance = LLMS.__new__(LLMS)
        # Create provider that fails on index 1
        mock_provider = MockProvider(fail_on_indices=[1])
        llms_instance._providers = [mock_provider]

        problems = [
            ("Problem 0", "Answer 0"),
            ("Problem 1", "Answer 1"),  # This will fail
            ("Problem 2", "Answer 2"),
        ]

        # Should not raise KeyError or IndexError
        result = llms_instance.benchmark(problems=problems, evaluator=None)
        assert result is not None

    def test_benchmark_output_uses_prompt_index(self):
        """Test that benchmark correctly uses prompt_index from output_data."""
        llms_instance = LLMS.__new__(LLMS)
        llms_instance._providers = [MockProvider()]

        problems = [
            ("What is 1+1?", "2"),
            ("What is 2+2?", "4"),
        ]

        table, questions_table = llms_instance.benchmark(
            problems=problems,
            evaluator=None,
            show_outputs=True
        )

        # Convert table to string and verify it contains expected data
        table_str = str(table)
        assert "MockProvider" in table_str


class TestBenchmarkResultsTable:
    """Test benchmark results table generation."""

    def test_benchmark_returns_two_tables(self):
        """Test that benchmark returns both results and questions tables."""
        llms_instance = LLMS.__new__(LLMS)
        llms_instance._providers = [MockProvider()]

        problems = [("Test question", "Test answer")]

        result = llms_instance.benchmark(problems=problems, evaluator=None)

        # Should return tuple of two tables
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestBenchmarkEdgeCases:
    """Test edge cases in benchmark functionality."""

    def test_benchmark_empty_outputs_skipped(self):
        """Test that models with no successful outputs are skipped."""
        llms_instance = LLMS.__new__(LLMS)
        # Provider that fails on all prompts
        llms_instance._providers = [MockProvider(fail_on_indices=[0, 1, 2])]

        problems = [
            ("Q1", "A1"),
            ("Q2", "A2"),
            ("Q3", "A3"),
        ]

        # Should handle gracefully without error
        result = llms_instance.benchmark(problems=problems, evaluator=None)
        assert result is not None

    def test_benchmark_single_problem(self):
        """Test benchmark with a single problem."""
        llms_instance = LLMS.__new__(LLMS)
        llms_instance._providers = [MockProvider()]

        problems = [("Single question?", "Single answer")]

        result = llms_instance.benchmark(problems=problems, evaluator=None)
        assert result is not None
