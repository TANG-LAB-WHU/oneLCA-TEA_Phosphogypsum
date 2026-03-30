"""Tests for LCA Module"""

from pgloop.lca.inventory import LifeCycleInventory
from pgloop.lca.lca_engine import LCAEngine


def test_lca_engine_init():
    engine = LCAEngine()
    assert engine is not None


def test_inventory_creation():
    lci = LifeCycleInventory("Test", "1 kg", 1.0)
    assert lci.process_name == "Test"
