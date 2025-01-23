import unittest
import numpy as np
import threading
from parameterized import parameterized

import ring_reduce_impl_socket as ring_reduce_impl
from ring_reduce_impl_socket import (
    init_mailboxes,
    ring_allreduce
)


def run_allreduce_across_threads(world_size: int, local_data_arrays):
    """
    Utility function:
    Given a list of local arrays (length = WORLD_SIZE),
    launch threads that each call ring_allreduce on the corresponding array.
    Return a list of outputs, one per rank.
    """
    # Make sure local_data_arrays has exactly WORLD_SIZE entries
    assert len(local_data_arrays) == world_size

    # Reset global mailboxes and counters
    init_mailboxes(world_size)

    # Storage for results
    results = [None] * world_size

    # Worker function to be run in each thread
    def worker(rank):
        out = ring_allreduce(rank, local_data_arrays[rank])
        results[rank] = out

    # Launch threads
    threads = []
    for r in range(world_size):
        t = threading.Thread(target=worker, args=(r,))
        threads.append(t)
        t.start()

    # Wait for all threads to finish
    for t in threads:
        t.join()

    return results


def get_world_sizes_to_test():
    return [2, 3, 4, 6]


def get_array_lengths_to_test():
    return [100, 128, 131, 213, 256, 512, 513]


class TestRingAllReduce(unittest.TestCase):
    """
    A collection of test cases verifying the correctness of the ring_allreduce
    reference implementation under various scenarios.
    """

    @parameterized.expand(get_world_sizes_to_test())
    def test_small_fixed_data(self, world_size: int):
        """
        Test a very small array (length=3) with some fixed data.
        """
        # Prepare deterministic data for each rank
        # Example: rank 0: [1.0, 2.0, 3.0], rank 1: [4.0, 5.0, 6.0], ...
        local_data_arrays = []
        for i in range(world_size):
            local_data_arrays.append(np.array([i + j for j in range(3)], dtype=float))

        # Expected output is the sum of all local arrays
        expected = np.sum(local_data_arrays, axis=0)

        results = run_allreduce_across_threads(world_size, local_data_arrays)

        for r in range(world_size):
            self.assertTrue(
                np.allclose(results[r], expected),
                f"Rank {r} output {results[r]} != expected {expected}",
            )

    @parameterized.expand(get_world_sizes_to_test())
    def test_random_data_length_5(self, world_size: int):
        """
        Test an array of length=5, filled with random data for each rank.
        """
        np.random.seed(0)
        local_data_arrays = [np.random.randn(5) for _ in range(world_size)]
        expected = np.sum(local_data_arrays, axis=0)  # Sum along the ranks

        results = run_allreduce_across_threads(world_size, local_data_arrays)

        for r in range(world_size):
            self.assertTrue(
                np.allclose(results[r], expected),
                f"Rank {r} output mismatch with expected sum"
            )

    @parameterized.expand(get_world_sizes_to_test())
    def test_zero_length_array(self, world_size: int):
        """
        Edge case: array of length=0 for all ranks.
        Check that it doesn't break the ring logic and yields an empty result.
        """
        local_data_arrays = [np.array([], dtype=float) for _ in range(world_size)]
        expected = np.array([], dtype=float)

        results = run_allreduce_across_threads(world_size, local_data_arrays)

        for r in range(world_size):
            self.assertEqual(results[r].shape, (0,))
            self.assertTrue(np.allclose(results[r], expected))

    @parameterized.expand(get_world_sizes_to_test())
    def test_larger_random_array(self, world_size: int):
        """
        Stress test with a larger array (length=100) of random data.
        """
        np.random.seed(123)
        local_data_arrays = [np.random.randn(100) for _ in range(world_size)]
        expected = np.sum(local_data_arrays, axis=0)

        results = run_allreduce_across_threads(world_size, local_data_arrays)

        for r in range(world_size):
            self.assertTrue(
                np.allclose(results[r], expected),
                f"Rank {r} output mismatch with expected sum"
            )

    # cartesian product of world sizes and array lengths
    @parameterized.expand([
        (world_size, length)
        for world_size in get_world_sizes_to_test()
        for length in get_array_lengths_to_test()
    ])
    def test_larger_random_array(self, world_size: int, length: int):
        """
        Stress test with a larger array (length=100) of random data.
        """
        np.random.seed(123)
        local_data_arrays = [np.random.randn(length) for _ in range(world_size)]
        expected = np.sum(local_data_arrays, axis=0)

        results = run_allreduce_across_threads(world_size, local_data_arrays)

        for r in range(world_size):
            self.assertTrue(
                np.allclose(results[r], expected),
                f"Rank {r} output mismatch with expected sum"
            )

    @parameterized.expand(get_world_sizes_to_test())
    def test_uneven_chunk_distribution(self, world_size: int):
        """
        Test an array length that is NOT divisible by WORLD_SIZE.
        For example, if WORLD_SIZE=3 and length=10, chunk boundaries might be:
          - rank 0: 4 elements
          - rank 1: 3 elements
          - rank 2: 3 elements
        """
        np.random.seed(42)
        array_length = 10
        local_data_arrays = [np.random.randn(array_length) for _ in range(world_size)]
        # Expected is just the element-wise sum across the rank dimension
        expected = np.sum(local_data_arrays, axis=0)

        results = run_allreduce_across_threads(world_size, local_data_arrays)
        for r in range(world_size):
            self.assertTrue(
                np.allclose(results[r], expected),
                f"Rank {r} output mismatch. \nGot: {results[r]}\nExpected: {expected}"
            )

    @parameterized.expand(get_world_sizes_to_test())
    def test_insufficient_chunks(self, world_size: int):
        """
        Test case where the total array length is less than WORLD_SIZE.
        E.g. if WORLD_SIZE=3 and length=2, the chunk boundaries might be:
          - rank 0: 1 element
          - rank 1: 1 element
          - rank 2: 0 elements
        """
        np.random.seed(123)
        array_length = world_size - 1
        local_data_arrays = [np.random.randn(array_length) for _ in range(world_size)]
        # Expected is the element-wise sum across the rank dimension
        expected = np.sum(local_data_arrays, axis=0)

        results = run_allreduce_across_threads(world_size, local_data_arrays)
        for r in range(world_size):
            self.assertTrue(
                np.allclose(results[r], expected),
                f"Rank {r} output mismatch. \nGot: {results[r]}\nExpected: {expected}"
            )

    @parameterized.expand(get_world_sizes_to_test())
    def test_multiple_runs_in_one_test(self, world_size: int):
        """
        Confirm that re-initializing the mailboxes allows multiple
        sequential calls to ring_allreduce without leftover state contamination.
        """
        # First run
        arrs_1 = [np.ones(3) * i for i in range(world_size)]
        expected_1 = np.sum(arrs_1, axis=0)
        results_1 = run_allreduce_across_threads(world_size, arrs_1)
        for r in range(world_size):
            self.assertTrue(np.allclose(results_1[r], expected_1))

        # Second run with different data
        arrs_2 = [np.arange(3) * (i + 1) for i in range(world_size)]
        expected_2 = np.sum(arrs_2, axis=0)
        results_2 = run_allreduce_across_threads(world_size, arrs_2)
        for r in range(world_size):
            self.assertTrue(np.allclose(results_2[r], expected_2))

    @parameterized.expand(get_world_sizes_to_test())
    def test_byte_counters(self, world_size: int):
        """
        Check that bytes_sent and bytes_received are incremented
        after a normal run. The exact values may vary, but we expect them
        to be > 0 if there's actual communication.
        """
        np.random.seed(42)
        local_data_arrays = [np.random.randn(5) for _ in range(world_size)]

        init_mailboxes(world_size)

        # Before running
        self.assertEqual(ring_reduce_impl.bytes_sent, 0)
        self.assertEqual(ring_reduce_impl.bytes_received, 0)

        results = run_allreduce_across_threads(world_size, local_data_arrays)

        # After
        self.assertGreater(ring_reduce_impl.bytes_sent, 0, "No bytes sent? Unexpected.")
        self.assertGreater(ring_reduce_impl.bytes_received, 0, "No bytes received? Unexpected.")

        # Verify correctness too:
        expected = np.sum(local_data_arrays, axis=0)
        for r in range(world_size):
            self.assertTrue(np.allclose(results[r], expected))


if __name__ == "__main__":
    unittest.main()
