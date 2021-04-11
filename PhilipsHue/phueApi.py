from phue import Bridge
import ipdb

BRIDGE_IP = "192.168.50.67"

b = Bridge(BRIDGE_IP)
# b.connect()

b.get_group()

ipdb.set_trace()