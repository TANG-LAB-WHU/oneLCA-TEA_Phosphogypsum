"""Tests for TEA Module"""
import pytest
from pgloop.tea.tea_engine import TEAEngine

def test_tea_engine_init():
    engine = TEAEngine()
    assert engine is not None
