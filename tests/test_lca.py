"""Tests for LCA Module"""
import pytest
from pgloop.lca.engine import LCAEngine
from pgloop.lca.inventory import LifeCycleInventory

def test_lca_engine_init():
    engine = LCAEngine()
    assert engine is not None

def test_inventory_creation():
    lci = LifeCycleInventory("Test", "1 kg", 1.0)
    assert lci.process_name == "Test"
