"""
Real-time Stream Processor (MQTT -> SQLite WAL)

Subscribes to raw telemetry data via MQTT, performs physics/conservation bounds checking,
computes instantaneous LCA/TEA KPIs, and persists to a SQLite database in WAL mode
for non-blocking dashboard reads.
"""

import os
import json
import sqlite3
import logging
from typing import Dict, Any, Tuple

try:
    import paho.mqtt.client as mqtt
    IOT_AVAILABLE = True
except ImportError:
    mqtt = None
    IOT_AVAILABLE = False


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("StreamProcessor")


class StreamProcessor:
    def __init__(
        self,
        db_path: str = "sensors_live.db",
        mqtt_broker: str = "localhost",
        mqtt_port: int = 1883,
        mqtt_topic: str = "phosphogypsum/reactor/raw",
        client_id: str = "pg_stream_processor"
    ):
        self.db_path = db_path
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.mqtt_topic = mqtt_topic
        self.client_id = client_id
        
        if not IOT_AVAILABLE:
            raise ImportError("Please install paho-mqtt: pip install paho-mqtt")

        self._setup_database()
        self.mqtt_client = self._setup_mqtt()

    def _setup_database(self):
        """Initialize SQLite database with WAL mode."""
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        # Enable Write-Ahead Logging to prevent lock contention between writer and Streamlit reader
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                node_id TEXT NOT NULL,
                value REAL,
                status TEXT,
                lca_co2_rate REAL,
                tea_cost_rate REAL
            )
        """)
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path} with WAL mode.")

    def _setup_mqtt(self):
        # We explicitly use paho-mqtt 1.x / 2.x compatible syntax
        client = mqtt.Client(client_id=self.client_id, clean_session=False)
        client.on_connect = self._on_connect
        client.on_message = self._on_message
        return client

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info(f"Connected to MQTT broker. Subscribing to {self.mqtt_topic} (QoS 1)")
            client.subscribe(self.mqtt_topic, qos=1)
        else:
            logger.error(f"Failed to connect to MQTT broker with code: {rc}")

    def _validate_payload(self, payload: Dict[str, Any]) -> str:
        """
        Validate incoming data against physics and conservation bounds.
        Example: Temperature should be between 20 and 150 C.
        """
        val = float(payload.get("value", 0.0))
        node_id = payload.get("node_id", "")
        
        # Simulated physical validation gates
        if "temp" in node_id.lower() and (val < 20 or val > 150):
            return "ALARM: T_OUT_OF_BOUNDS"
        if "flow" in node_id.lower() and val < 0:
            return "ALARM: NEGATIVE_FLOW"
        if "ph" in node_id.lower() and (val < 0 or val > 14):
            return "ALARM: INVALID_PH"
            
        return "OK"

    def _compute_live_lca_tea(self, payload: Dict[str, Any]) -> Tuple[float, float]:
        """
        Compute instantaneous LCA/TEA metrics based on the current flowsheet streams.
        (Mock logic representing calls to LCAEngine and TEAEngine)
        """
        val = float(payload.get("value", 0.0))
        # Simulated calculation: CO2 rate proportional to value, cost proportional to value^1.2
        co2_rate = val * 0.45
        cost_rate = (val ** 1.2) * 0.12 if val >= 0 else 0
        return co2_rate, cost_rate

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            status = self._validate_payload(payload)
            co2, cost = self._compute_live_lca_tea(payload)
            
            # Fast write to WAL database
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT INTO telemetry (timestamp, node_id, value, status, lca_co2_rate, tea_cost_rate) VALUES (?, ?, ?, ?, ?, ?)",
                (str(payload.get("timestamp", "")), str(payload.get("node_id", "")), float(payload.get("value", 0.0)), status, co2, cost)
            )
            conn.commit()
            conn.close()
            
            if "ALARM" in status:
                logger.warning(f"Constraint Violation: {status} for payload {payload}")
            else:
                logger.debug(f"Processed telemetry: {payload.get('node_id')}={payload.get('value')}")
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def run(self):
        """Start the MQTT processing loop."""
        logger.info(f"Connecting to MQTT broker at {self.mqtt_broker}:{self.mqtt_port}")
        try:
            self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, keepalive=60)
            self.mqtt_client.loop_forever()
        except KeyboardInterrupt:
            logger.info("Terminated by user.")
            self.mqtt_client.disconnect()
        except Exception as e:
            logger.error(f"Failed to run stream processor: {e}")


if __name__ == "__main__":
    processor = StreamProcessor()
    processor.run()
