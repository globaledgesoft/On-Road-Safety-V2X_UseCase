## aws_send file

import sys
import ssl
import aws_config as config
import json
sys.path.append("./libboto3")
import boto3

client= boto3.client('iot-data',aws_access_key_id =config.aws_access_key_id ,aws_secret_access_key =config.aws_secret_access_key , region_name

message = {"hello":"qcs"}

message_json = json.dumps(message)

response = client.publish(
                     topic = "qcs610/msg",
                     qos = 1,
                     payload = message_json
                     )



