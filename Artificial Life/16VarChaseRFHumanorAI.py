import pygame
import pandas as pd
import numpy as np
import random
import sys
from math import sqrt,fabs,atan2, degrees, pi
from timeit import default_timer as timer
from sklearn.preprocessing import OneHotEncoder
import time
sys.setrecursionlimit(10000)

oldfile = open("Data/16VarALData.txt","r")
olddata = oldfile.read()
oldfile.close()

file = open("Data/16VarALData.txt", "w")
file.write(olddata)
file.flush()

pygame.init()
pygame.mixer.music.load("Data/Corridors_of_Time_Piano_.wav")
pygame.mixer.music.play(-1)

display_width = 1200
display_height = 800

black = (0, 0, 0)
white = (255, 255, 255)
red = (200, 0, 0)
green = (0, 200, 0)
bright_red = (255, 0, 0)
bright_green = (0, 255, 0)

animal_width = 73
animal_height = 73

gameDisplay = pygame.display.set_mode((display_width, display_height))
pygame.display.set_caption('Artificial Life')
clock = pygame.time.Clock()

AnimalImg = pygame.image.load('Data/animalimg.png')
ShelterImg = pygame.image.load('Data/Shelter.png')
Icon = pygame.image.load('Data/animalimg.png')
pygame.display.set_icon(Icon)

def food_counter(count):
    font = pygame.font.SysFont("comicsansms", 25)
    text = font.render("Food Eaten: " + str(count), True, black)
    gameDisplay.blit(text, (0, 0))

def water_counter(count):
    font = pygame.font.SysFont("comicsansms", 25)
    text = font.render("Water Consumed: " + str(count), True, black)
    gameDisplay.blit(text, (0, display_height*0.04))

def energy_counter(count):
    font = pygame.font.SysFont("comicsansms", 25)
    text = font.render("Energy Restored: " + str(count), True, black)
    gameDisplay.blit(text, (0, display_height*0.08))

def hunger(hungertimer, initialtime, forage):
    font = pygame.font.SysFont("comicsansms", 25)
    text = font.render("Hunger: " + str(round(initialtime - hungertimer) + forage), True, black)
    gameDisplay.blit(text, (display_width-175, 0))
    if (initialtime - hungertimer+forage) < 0:
        edge_contact()
    return

def thirst(thirsttimer, initialtime, forage):
    font = pygame.font.SysFont("comicsansms", 25)
    text = font.render("Thirst: " + str(round(initialtime - thirsttimer)+forage), True, black)
    gameDisplay.blit(text, (display_width-175, display_height*0.04))
    if (initialtime - thirsttimer+forage) < 0:
        edge_contact()
    return

def energy(energytimer, initialtime, forage):
    font = pygame.font.SysFont("comicsansms", 25)
    text = font.render("Energy: " + str(round(initialtime - energytimer + forage)), True, black)
    gameDisplay.blit(text, (display_width-175, display_height*0.08))
    if (initialtime - energytimer+forage) < 0:
        edge_contact()
    return

def currenttime(timer):
    font = pygame.font.SysFont("comicsansms", 25)
    text = font.render("Time: " + str(round(timer)), True, black)
    gameDisplay.blit(text, (display_width/2-50, display_height*0.01))
    return

def edge_contact():

    pygame.mixer.music.stop()

    largeText = pygame.font.SysFont("comicsansms", 115)
    TextSurf, TextRect = text_objects("You Died", largeText)
    TextRect.center = ((display_width / 2), (display_height / 2))
    gameDisplay.blit(TextSurf, TextRect)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        button("Play Again", display_width*.3, display_height*(5/8), 100, 50, green, bright_green, game_loop)
        button("Quit", display_width*.6, display_height*(5/8), 100, 50, red, bright_red, quit)

        pygame.display.update()
        clock.tick(15)

def food(x, y, width, height, color):
    pygame.draw.rect(gameDisplay, color, [x, y, width, height])

def water(x, y, width, height, color):
    pygame.draw.rect(gameDisplay, color, [x, y, width, height])

def animal(x, y):
    gameDisplay.blit(AnimalImg, (x, y))

def text_objects(text, font):
    textSurface = font.render(text, True, black)
    return textSurface, textSurface.get_rect()

def button(msg, x, y, w, h, ic, ac, action=None):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    # print(click)
    if x + w > mouse[0] > x and y + h > mouse[1] > y:
        pygame.draw.rect(gameDisplay, ac, (x, y, w, h))
        if click[0] == 1 and action != None:
            action()
    else:
        pygame.draw.rect(gameDisplay, ic, (x, y, w, h))
    smallText = pygame.font.SysFont("comicsansms", 20)
    textSurf, textRect = text_objects(msg, smallText)
    textRect.center = ((x + (w / 2)), (y + (h / 2)))
    gameDisplay.blit(textSurf, textRect)

def quit():
    pygame.quit()
    quit()

def get_item(degitem,x,y):
    x_change = 0
    y_change = 0

    if 28 < degitem < 61:
        x_change = 5
        y_change = 5
        print(9)
    elif 60 < degitem < 121:
        x_change = 5
        print(6)
    elif 120 < degitem < 151:
        x_change = 5
        y_change = -5
        print(3)
    elif 150 < degitem < 211:
       y_change = -5
       print(2)
    elif 210 < degitem < 241:
        x_change = -5
        y_change = -5
        print(1)
    elif 240 < degitem < 301:
        x_change = -5
        print(4)
    elif 300 < degitem < 331:
        x_change = -5
        y_change = 5
        print(7)
    elif 330 < degitem < 361 or degitem < 29:
        y_change = 5
        print(8)

    x += x_change
    y += y_change
    return x,y

def food_move(food_startx,food_starty):

    foodmove = random.randrange(1,9,1)
    x_change = 0
    y_change = 0

    if food_startx < 50:
        x_change = 5
    elif food_startx > display_width-50:
        x_change = -5
    elif food_starty < 50:
        y_change = 5
    elif food_starty > display_height-50:
        y_change = -5
    elif foodmove == 4:
        x_change = -5
    elif foodmove == 6:
        x_change = 5
    elif foodmove == 8:
        y_change = -5
    elif foodmove == 2:
        y_change = 5
    elif foodmove == 1:
        x_change = -5
        y_change = 5
    elif foodmove == 5:
        x_change = 0
        y_change = 0
    elif foodmove == 3:
        x_change = 5
        y_change = 5
    elif foodmove == 7:
        x_change = -5
        y_change = -5
    elif foodmove == 9:
        x_change = 5
        y_change = -5
    food_startx += x_change
    food_starty += y_change
    return food_startx, food_starty

def game_intro():
    intro = True

    while intro:
        for event in pygame.event.get():
            # print(event)
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        gameDisplay.fill(white)
        largeText = pygame.font.SysFont("comicsansms", 115)
        TextSurf, TextRect = text_objects("Artificial Life", largeText)
        TextRect.center = ((display_width / 2), (display_height / 2))
        gameDisplay.blit(TextSurf, TextRect)

        button("GO!", display_width*.3, display_height*(5/8), 100, 50, green, bright_green, game_loop)
        button("Quit", display_width*.6, display_height*(5/8), 100, 50, red, bright_red, quit)

        pygame.display.update()
        clock.tick(15)

def game_loop():

    global animal_width
    global animal_height
    global file

    x = (display_width * 0.45)
    y = (display_height * 0.8)

    x_change = 0
    y_change = 0

    food_startx = random.randrange(0+display_width*0.1, display_width-display_width*.1)
    food_starty = random.randrange(0+display_height*0.1, display_height-display_height*.1)
    food_width = 50
    food_height = 50

    water_startx = random.randrange(0+display_width*0.1, display_width-display_width*.1)
    water_starty = random.randrange(0+display_height*0.1, display_height-display_height*.1)
    water_width = 50
    water_height = 50

    shelter_startx = (display_width * 0.8)
    shelter_starty = (display_height * 0.8)
    shelter_height = 120
    shelter_width = 120

    food_green = (0, 170, 0)
    water_blue = (53, 115, 255)
    morning_color = (242, 240, 142)
    noon_color = (64, 187, 247)
    evening_color = (23, 79, 162)
    night_color = (30, 12, 20)

    scorefood = 0
    scorewater = 0
    scoreenergy = 0
    foragefood = 0
    foragewater = 0
    forageenergy = 0
    foodtimemultiplier = 2
    watertimemultiplier = 2.5
    energytimemultiplier = 3
    daytimemultiplier = 0.2
    initialfoodtime = 100
    initialwatertime = 100
    initialenergytime = 100
    keypress = 5
    daytimercount = False
    daycount = 0
    starttime = time.time()
    movingfoodtimer = 0
    foodmove = True
    action = 1

    gameExit = False

    rfclf = pd.read_pickle('Data/RFpickle.pickle')

    while not gameExit:

        movingfoodtimer = (time.time()-starttime)
        if round(movingfoodtimer) % 2 == 0:
            food_startx, food_starty = food_move(food_startx,food_starty)

        daytimer = round(((time.time()-starttime)*daytimemultiplier),2)
        totaltimer = time.time()-starttime

        if daytimercount == True:
            daytimer = round((daytimer - 4*daycount),2)

        if daytimer < 1:
            gameDisplay.fill(morning_color)
            food_color = food_green
            water_color = water_blue
        elif daytimer < 2:
            gameDisplay.fill(noon_color)
            food_color = food_green
            water_color = water_blue
        elif daytimer < 3:
            gameDisplay.fill(evening_color)
            food_color = food_green
            water_color = water_blue
        elif daytimer < 4:
            gameDisplay.fill(night_color)
            food_color = black
            water_color = black
        elif daytimer < 5:
            daytimercount = True
            daycount = daycount+ 1
            food_color = food_green
            water_color = water_blue

        food(food_startx, food_starty, food_width, food_height, food_color)
        water(water_startx, water_starty, water_width, water_height, water_color)

        animal(x, y)
        food_counter(scorefood)
        water_counter(scorewater)
        energy_counter(scoreenergy)
        hunger(foodtimemultiplier*totaltimer, initialfoodtime, foragefood)
        thirst(watertimemultiplier*totaltimer, initialwatertime, foragewater)
        energy(energytimemultiplier*totaltimer, initialenergytime, forageenergy)
        gameDisplay.blit(ShelterImg, (shelter_startx, shelter_starty))

        distxfood = food_startx - x
        distyfood = food_starty - y
        distfood = round(sqrt(fabs(distxfood)+fabs(distyfood)),2)

        radfood = atan2(distxfood,distyfood)
        radfood %= 2*pi
        degfood = round(degrees(radfood),2)

        distxwater = water_startx - x
        distywater = water_starty - y
        distwater = round(sqrt(fabs(distxwater) + fabs(distywater)),2)

        radwater = atan2(distxwater, distywater)
        radwater %= 2 * pi
        degwater = round(degrees(radwater),2)

        distxshelter = shelter_startx - x
        distyshelter = shelter_starty - y
        distshelter = round(sqrt(fabs(distxshelter) + fabs(distyshelter)),2)

        radshelter = atan2(distxshelter, distyshelter)
        radshelter %= 2 * pi
        degshelter = round(degrees(radshelter),2)

        distcorner = round(sqrt(x+y),2)

        radcorner = atan2(y,x)
        radcorner %= 2 * pi
        degcorner = round(degrees(radcorner),2)

        distcornerfood = round(sqrt(food_startx+food_starty),2)

        radcornerfood = atan2(food_starty, food_startx)
        radcornerfood %= 2 * pi
        degcornerfood = round(degrees(radcornerfood),2)

        distcornerwater = round(sqrt(water_startx+water_starty),2)

        radcornerwater = atan2(water_starty, water_startx)
        radcornerwater %= 2 * pi
        degcornerwater = round(degrees(radcornerwater),2)

        distcornershelter = round(sqrt(shelter_starty+shelter_startx),2)

        radcornershelter = atan2(shelter_starty, shelter_startx)
        radcornershelter %= 2 * pi
        degcornershelter = round(degrees(radcornershelter), 2)

        foodtime = round(initialfoodtime - foodtimemultiplier*totaltimer) + foragefood
        watertime = round(initialwatertime - watertimemultiplier*totaltimer) + foragewater
        energytime = round(initialenergytime - energytimemultiplier*totaltimer) + forageenergy

        # file.write(str(distfood) + "," + str(degfood) + "," + str(distwater) + "," + str(degwater) + "," + str(distcorner) + "," +
        #            str(degcorner) + "," + str(distcornerfood) + "," + str(degcornerfood) + "," + str(distcornerwater) + "," +
        #            str(degcornerwater) + "," + str(distcornershelter) + "," + str(degcornershelter) + "," + str(foodtime) + "," +
        #            str(watertime) + "," + str(energytime) + "," + str(daytimer) + "," + str(action) + "\n")
        # file.flush()

        # for event in pygame.event.get():

        #     if event.type == pygame.KEYDOWN:
        #         if event.key == pygame.K_KP1:
        #             action = 1
        #         if event.key == pygame.K_KP2:
        #             action = 2
        #         if event.key == pygame.K_KP3:
        #             action = 3
        input_data = np.array([distfood,degfood,distwater,degwater,distcorner,degcorner,distcornerfood,degcornerfood,distcornerwater,degcornerwater,distcornershelter,degcornershelter,foodtime,watertime,energytime,daytimer])
        action = rfclf.predict(input_data)
        print(action)

        if action == 1:
            x,y = get_item(degfood,x,y)
        elif action == 2:
            x,y = get_item(degwater,x,y)
        elif action == 3:
            x,y = get_item(degshelter,x,y)

        if x-120 > display_width - animal_width or x < -105:               #-20   -5
            edge_contact()

        if y-150 > display_height - animal_height or y < -105:             #-1xdif-50   -5
            edge_contact()

        if daytimer > 3 and y > shelter_starty and y < shelter_starty + shelter_height and x > shelter_startx and x < shelter_startx + shelter_width and energytime < 100 or daytimer > 3 and ((y+animal_width/2) > shelter_starty and (y+animal_width/2) < shelter_starty + shelter_height and (x+animal_width/2) > shelter_startx and (x+animal_width/2) < shelter_startx + shelter_width):
            if energytime < 100:
                forageenergy += 0.3

        if (y > food_starty and y < food_starty + food_height and x > food_startx and x < food_startx + food_width) or (y+animal_width > food_starty and y+animal_width < food_starty + food_height and x+animal_width > food_startx and x+animal_width < food_startx + food_width)  \
            or ((y+animal_width/2) > food_starty and (y+animal_width/2) < food_starty + food_height and (x+animal_width/2) > food_startx and (x+animal_width/2) < food_startx + food_width):
                food_startx = random.randrange(0 + display_width * 0.1, display_width - display_width * .1)
                food_starty = random.randrange(0 + display_height * 0.1, display_height - display_height * .1)
                scorefood += 1
                foragefood += 5

        if (y > water_starty and y < water_starty + water_height and x > water_startx and x < water_startx + water_width) or (y+animal_width > water_starty and y+animal_width < water_starty + water_height and x+animal_width > water_startx and x+animal_width < water_startx + water_width)  \
            or ((y+animal_width/2) > water_starty and (y+animal_width/2) < water_starty + water_height and (x+animal_width/2) > water_startx and (x+animal_width/2) < water_startx + water_width):
                water_startx = random.randrange(0 + display_width * 0.1, display_width - display_width * .1)
                water_starty = random.randrange(0 + display_height*0.1, display_height - display_height*.1)
                scorewater += 1
                foragewater += 5

        pygame.display.update()
        clock.tick(60)

game_intro()
game_loop()
pygame.quit()
quit()
file.close()