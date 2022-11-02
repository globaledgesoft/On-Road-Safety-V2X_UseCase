import sys
import argparse
#from awscrt import io, http, auth
sys.path.append("/data/aws-iot/")

from awsiot import mqtt_connection_builder

class MQTT_builder:
    def __init__(self, description) -> None:
        self.parser = argparse.ArgumentParser(description="Send and receive messages through and MQTT connection.")
        self.commands = {}
        self.parsed_commands = None

    def build_mqtt_connection(self, on_connection_interrupted, on_connection_resumed, endpoint, port, cert_path, pri_key, ca_path, client_id):
        mqtt_connection = mqtt_connection_builder.mtls_from_path(
            endpoint=endpoint,
            port=port,
            cert_filepath=cert_path,
            pri_key_filepath=pri_key,
            ca_filepath=ca_path,
            on_connection_interrupted=on_connection_interrupted,
            on_connection_resumed=on_connection_resumed,
            client_id=client_id,
            clean_session=False,
            keep_alive_secs=30,
            http_proxy_options=None)
        return mqtt_connection

