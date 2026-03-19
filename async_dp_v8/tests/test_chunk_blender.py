"""Test chunk blender."""
import numpy as np
from async_dp_v8.control.chunk_blender import ChunkBlender


def test_first_chunk_no_blend():
    blender = ChunkBlender(execute_horizon=4, alpha=0.6)
    chunk = np.ones((12, 6))
    result = blender.blend(chunk)
    np.testing.assert_array_equal(result, chunk)


def test_second_chunk_blends():
    blender = ChunkBlender(execute_horizon=4, alpha=0.6)
    chunk1 = np.ones((12, 6))
    blender.blend(chunk1)

    chunk2 = np.ones((12, 6)) * 2.0
    result = blender.blend(chunk2)
    # First 8 steps should be blended (overlap from prev tail)
    assert result[0, 0] != 2.0  # Should be blended


def test_reset():
    blender = ChunkBlender()
    blender.blend(np.ones((12, 6)))
    blender.reset()
    assert blender.prev_tail is None
