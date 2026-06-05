"""
Industrial Edge Bridge (OPC UA -> MQTT)

Provides a production-grade asynchronous bridge to subscribe to OPC UA telemetry
and publish to an MQTT broker. Implements QoS 1, connection resilience, and LWT.
"""

import asyncio
import json
import logging
from typing import List

try:
    from asyncua import Client, Node
    import paho.mqtt.client as mqtt
    IOT_AVAILABLE = True
except ImportError:
    Client = None
    Node = None
    mqtt = None
    IOT_AVAILABLE = False


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("EdgeBridge")


class SubHandler:
    """Subscription Handler for asyncua to forward data to MQTT."""
    def __init__(self, mqtt_client, topic: str):
        self.mqtt_client = mqtt_client
        self.topic = topic

    def datachange_notification(self, node: "Node", val, data):
        """Callback when OPC UA node value changes."""
        node_id = str(node.nodeid)
        # Use server timestamp if available, else local
        if data.monitored_item.Value.ServerTimestamp:
            ts = data.monitored_item.Value.ServerTimestamp.isoformat()
        else:
            ts = data.monitored_item.Value.SourceTimestamp.isoformat() if data.monitored_item.Value.SourceTimestamp else None
            
        payload = {
            "node_id": node_id,
            "value": val,
            "timestamp": ts
        }
        json_payload = json.dumps(payload)
        # Publish to MQTT with QoS 1 (At least once)
        self.mqtt_client.publish(self.topic, json_payload, qos=1)
        logger.debug(f"Published to MQTT: {json_payload}")


class EdgeBridge:
    def __init__(
        self,
        opcua_url: str = "opc.tcp://localhost:4840",
        mqtt_broker: str = "localhost",
        mqtt_port: int = 1883,
        mqtt_topic: str = "phosphogypsum/reactor/raw",
        client_id: str = "pg_edge_bridge"
    ):
        self.opcua_url = opcua_url
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.mqtt_topic = mqtt_topic
        self.client_id = client_id

        if not IOT_AVAILABLE:
            raise ImportError("Please install IoT dependencies: pip install asyncua paho-mqtt")

        self.mqtt_client = self._setup_mqtt()

    def _setup_mqtt(self):
        client = mqtt.Client(client_id=self.client_id, clean_session=False)
        client.on_connect = self._on_mqtt_connect
        client.on_disconnect = self._on_mqtt_disconnect
        
        # Last Will and Testament
        lwt_topic = f"system/clients/{self.client_id}/status"
        client.will_set(lwt_topic, payload=json.dumps({"status": "offline"}), qos=1, retain=True)
        return client

    def _on_mqtt_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info("Connected to MQTT broker.")
            lwt_topic = f"system/clients/{self.client_id}/status"
            client.publish(lwt_topic, payload=json.dumps({"status": "online"}), qos=1, retain=True)
        else:
            logger.error(f"Failed to connect to MQTT broker with code: {rc}")

    def _on_mqtt_disconnect(self, client, userdata, rc):
        logger.warning(f"Disconnected from MQTT broker with code: {rc}")

    async def run(self, node_ids: List[str]):
        """Connects to OPC UA, sets up subscriptions, and runs indefinitely."""
        logger.info(f"Connecting to MQTT broker at {self.mqtt_broker}:{self.mqtt_port}")
        self.mqtt_client.connect_async(self.mqtt_broker, self.mqtt_port, keepalive=60)
        self.mqtt_client.loop_start()

        client = Client(url=self.opcua_url)
        while True:
            try:
                logger.info(f"Connecting to OPC UA server at {self.opcua_url}")
                async with client:
                    handler = SubHandler(self.mqtt_client, self.mqtt_topic)
                    subscription = await client.create_subscription(500, handler)
                    
                    nodes_to_subscribe = []
                    for nid in node_ids:
                        node = client.get_node(nid)
                        nodes_to_subscribe.append(node)
                    
                    if nodes_to_subscribe:
                        await subscription.subscribe_data_change(nodes_to_subscribe)
                        logger.info(f"Subscribed to {len(nodes_to_subscribe)} OPC UA nodes.")
                    
                    # Exponential Backoff reset on successful connect
                    backoff = 5
                    # Keep the connection alive
                    while True:
                        await asyncio.sleep(1)
            except asyncio.CancelledError:
                logger.info("Edge Bridge execution cancelled.")
                break
            except Exception as e:
                logger.error(f"OPC UA connection error: {e}. Reconnecting in {backoff} seconds...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
        
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()


if __name__ == "__main__":
    import sys
    # Example usage: python edge_bridge.py "ns=2;i=2" "ns=2;i=3"
    nodes = sys.argv[1:] if len(sys.argv) > 1 else ["ns=2;i=2"]
    bridge = EdgeBridge()
    try:
        asyncio.run(bridge.run(nodes))
    except KeyboardInterrupt:
        logger.info("Terminated by user.")
