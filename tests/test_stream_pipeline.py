"""
Tests for the Industrial Stream Pipeline (Edge Bridge + Stream Processor)
"""

import os
import json
import pytest
import sqlite3
from unittest.mock import patch
from pgloop.iodata import stream_processor

@pytest.fixture
def processor_fixture(tmp_path):
    with patch.object(stream_processor, 'IOT_AVAILABLE', True), \
         patch.object(stream_processor, 'mqtt'):
        db_path = tmp_path / "test_sensors_live.db"
        processor = stream_processor.StreamProcessor(db_path=str(db_path), mqtt_broker="localhost", mqtt_port=1883)
        return processor

def test_stream_processor_validation(processor_fixture):
    # Test valid temperature
    payload_valid_temp = {"node_id": "reactor_temp", "value": 100.0, "timestamp": "2023-01-01T12:00:00Z"}
    status = processor_fixture._validate_payload(payload_valid_temp)
    assert status == "OK"

    # Test invalid temperature
    payload_invalid_temp = {"node_id": "reactor_temp", "value": 200.0, "timestamp": "2023-01-01T12:00:00Z"}
    status = processor_fixture._validate_payload(payload_invalid_temp)
    assert status == "ALARM: T_OUT_OF_BOUNDS"

    # Test negative flow
    payload_neg_flow = {"node_id": "feed_flow", "value": -5.0, "timestamp": "2023-01-01T12:00:00Z"}
    status = processor_fixture._validate_payload(payload_neg_flow)
    assert status == "ALARM: NEGATIVE_FLOW"

def test_stream_processor_lca_tea(processor_fixture):
    payload = {"node_id": "acid_flow", "value": 10.0}
    co2, cost = processor_fixture._compute_live_lca_tea(payload)
    assert pytest.approx(co2) == 4.5  # 10 * 0.45
    assert pytest.approx(cost) == (10.0 ** 1.2) * 0.12

def test_stream_processor_db_write(processor_fixture):
    class MockMsg:
        def __init__(self, payload_dict):
            self.payload = json.dumps(payload_dict).encode('utf-8')
            
    payload = {"node_id": "acid_flow", "value": 10.0, "timestamp": "2023-01-01T12:00:00Z"}
    msg = MockMsg(payload)
    
    # Process message directly
    processor_fixture._on_message(None, None, msg)
    
    # Verify DB write
    conn = sqlite3.connect(processor_fixture.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT node_id, value, status, lca_co2_rate FROM telemetry")
    rows = cursor.fetchall()
    conn.close()
    
    assert len(rows) == 1
    assert rows[0][0] == "acid_flow"
    assert rows[0][1] == 10.0
    assert rows[0][2] == "OK"
    assert pytest.approx(rows[0][3]) == 4.5
