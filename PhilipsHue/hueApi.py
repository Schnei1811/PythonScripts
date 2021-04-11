import ipdb
import webcolors
import random
import colorsys

from hue_api import HueApi
api = HueApi()

def hsv2rgb(h,s,v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))

def closest_color(requested_color):
    # import ipdb;ipdb.set_trace()
    try:
        closest_color = webcolors.rgb_to_name(requested_color)
    except:
        min_colors = {}
        for hex_val, color in webcolors.CSS3_HEX_TO_NAMES.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(hex_val)
            rd = (r_c - requested_color[0]) ** 2
            gd = (g_c - requested_color[1]) ** 2
            bd = (b_c - requested_color[2]) ** 2
            min_colors[(rd + gd + bd)] = color
        closest_color = min_colors[min(min_colors.keys())]
    return closest_color

BRIDGE_IP = "192.168.50.67"

try:
    api.load_existing("user_cache")
except:
    api.create_new_user(BRIDGE_IP)
    api.save_api_key("user_cache")
    api.load_existing("user_cache")



# Kitchen
# Dining
# Studio
# Hallway
# Living room
# Office
# Game den
# Driveway
# Bedroom
# Go Light
# Bathroom
# TV
# Backyard
# Home
# Downstairs
# Main Floor
# Top floor
# Basement String Lights
# discoAGroup
# discoAGroupOdd
# discoAGroupEven
# Basement Sides Ceiling
# Basement Sides
# Backyard
# Basement Gradient
# Basement Gradient Sides
# Basement Gradient Sides Ceiling
# Office
# Studio / Kitchen
# Studio
# Studio / Porch
# Studio
# Studio / Kitchen
# Video

api.fetch_lights()
api.fetch_scenes()
api.fetch_groups()

# for group in api.groups:
#     print(group.name)

lightIdToName = {i : lightObject.name for (i, lightObject) in enumerate(api.lights)}
nameToLightId = {lightObject.name : i for (i, lightObject) in enumerate(api.lights)}

groupIdToName = {i: groupObject.name for (i, groupObject) in enumerate(api.groups)}
nameToGroupId = {groupObject.name: i for (i, groupObject) in enumerate(api.groups)}

nameToIndices = {groupObject.name: [light.id for light in groupObject.lights] for (i, groupObject) in enumerate(api.groups)}

# print("FETCH GROUPS", api.fetch_groups())                   #list of objects

# print("FETCH LIGHTS", api.fetch_lights())                     #instatiates api.lights

# print(api.lights)


# def changeColor(lights, color):
#     for light in range(lights):

# ids = api.groups[nameToGroupId["Downstairs"]].lights[]

# api.set_color("#00FF00", indices=nameToIndices["Downstairs"])

group = "Go Light"


api.turn_on(indices=nameToIndices[group])

hue = 0
prev_color = ""
while True:
    hue += 0.02
    if hue == 1:
        hue = 0

    #color = closest_color((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    rgb = hsv2rgb(hue, 1, 1)
    color = closest_color(rgb)

    if (prev_color != color):
        api.set_color(color, indices=nameToIndices[group])
        prev_color = color

    print(hue, rgb, color)
    # import ipdb;ipdb.set_trace()

  # colorName = webcolors.rgb_to_name(color)
    # print(color, colorName)
    # api.set_color(colorName, indices=nameToIndices["Downstairs"])



# print("FETCH SCENES", api.fetch_scenes())

# print("LIST GROUPS", api.list_groups())

# print("LIST LIGHTS", api.list_lights())                   #debug tool.    prints self.lights

# print("LIST SCENE GROUPS", api.list_scene_groups())

# print("LIST SCENES", api.list_scenes())

#print(api.set_brightness(brightness, indices=[]))          int:0-255 float: 0-1

# print(api.set_color(color, indices=[]))

# print(api.toggle_on, indices=[])

# print(api.turn_on, indicies=[])

# print(api.turn_off, indices=[])