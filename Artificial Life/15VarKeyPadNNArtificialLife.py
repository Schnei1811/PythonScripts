import numpy as np
import pygame
import random
import sys
import pandas as pd
from math import sqrt,fabs,atan2, degrees, pi
from timeit import default_timer as timer
sys.setrecursionlimit(10000)

#oldfile = open("Data/15VarAIData.txt","r")
#olddata = oldfile.read()
#oldfile.close()

#file = open("Data/15VarAIData.txt", "w")
#file.write(olddata + "\n")
#file.flush()

fmin = pd.read_pickle('Data/15VarALSimpleNN.pickle')

input_size = 15
hidden_size = 100
num_labels = 9

theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

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

food_color = (0,170,0)
water_color = (53, 115, 255)

animal_width = 73
animal_height = 73

gameDisplay = pygame.display.set_mode((display_width, display_height))
pygame.display.set_caption('Artificial Life')
clock = pygame.time.Clock()

AnimalImg = pygame.image.load('Data/animalimg.png')
ShelterImg = pygame.image.load('Data/Shelter.png')
Icon = pygame.image.load('Data/animalimg.png')
pygame.display.set_icon(Icon)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward_propagate(X, theta1, theta2):
    m = X.shape[0]
    a1 = np.insert(X, 0, values=np.ones(m), axis=1)
    z2 = a1 * theta1.T
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)
    z3 = a2 * theta2.T
    h = sigmoid(z3)
    return a1, z2, a2, z3, h

def food_counter(count):
    font = pygame.font.SysFont("comicsansms", 25)
    text = font.render("Food Eaten: " + str(count), True, black)
    gameDisplay.blit(text, (0, 0))

def water_counter(count):
    font = pygame.font.SysFont("comicsansms", 25)
    text = font.render("Water Consumed: " + str(count), True, black)
    gameDisplay.blit(text, (0, display_height*0.05))

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
    gameDisplay.blit(text, (display_width-175, display_height*0.05))
    if (initialtime - thirsttimer+forage) < 0:
        edge_contact()
    return

def edge_contact():

#    pygame.mixer.music.stop()
#    pygame.mixer.Sound.play(crash_sound)

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

    food_green = (0, 170, 0)
    water_blue = (53, 115, 255)
    morning_color = (242, 240, 142)
    noon_color = (64, 187, 247)
    evening_color = (23, 79, 162)
    night_color = (0, 0, 0)

    scorefood = 0
    scorewater = 0
    foragefood = 0
    foragewater = 0
    time = timer()
    foodtimemultiplier = 1.5
    watertimemultiplier = 2.5
    defaultinitialtime = 100
    daytimemultiplier = 0.2
    daycount = 0
    daytimercount = False
    gameExit = False

    while not gameExit:

        x += x_change
        y += y_change

        daytimer = round((daytimemultiplier*timer()),2)

        if daytimercount == True:
            daytimer = round((daytimer - 4*daycount),2)

        if daytimer < 1:
            gameDisplay.fill(morning_color)
            food_color = food_green
            water_color = water_blue
        elif daytimer < 2:
            gameDisplay.fill(noon_color)
        elif daytimer < 3:
            gameDisplay.fill(evening_color)
        elif daytimer < 4:
            gameDisplay.fill(night_color)
            food_color = black
            water_color = black
        elif daytimer < 5:
            daytimercount = True
            daycount = daycount+ 1

        food(food_startx, food_starty, food_width, food_height, food_color)
        water(water_startx, water_starty, water_width, water_height, water_color)

        initialfoodtime = defaultinitialtime + foodtimemultiplier*time
        initialwatertime = defaultinitialtime + watertimemultiplier*time

        animal(x, y)
        food_counter(scorefood)
        water_counter(scorewater)
        hunger(foodtimemultiplier*(timer()), initialfoodtime, foragefood)
        thirst(watertimemultiplier*(timer()), initialwatertime, foragewater)
        gameDisplay.blit(ShelterImg, (shelter_startx, shelter_starty))

        distxfood = food_startx - x
        distyfood = food_starty - y
        distfood = round(sqrt(fabs(distxfood) + fabs(distyfood)), 2)

        radfood = atan2(distxfood, distyfood)
        radfood %= 2 * pi
        degfood = round(degrees(radfood), 2)

        distxwater = water_startx - x
        distywater = water_starty - y
        distwater = round(sqrt(fabs(distxwater) + fabs(distywater)), 2)

        radwater = atan2(distxwater, distywater)
        radwater %= 2 * pi
        degwater = round(degrees(radwater), 2)

        distcorner = round(sqrt(x + y), 2)

        radcorner = atan2(y, x)
        radcorner %= 2 * pi
        degcorner = round(degrees(radcorner), 2)

        distcornerfood = round(sqrt(food_startx + food_starty), 2)

        radcornerfood = atan2(food_starty, food_startx)
        radcornerfood %= 2 * pi
        degcornerfood = round(degrees(radcornerfood), 2)

        distcornerwater = round(sqrt(water_startx + water_starty), 2)

        radcornerwater = atan2(water_starty, water_startx)
        radcornerwater %= 2 * pi
        degcornerwater = round(degrees(radcornerwater), 2)

        distcornershelter = round(sqrt(shelter_starty + shelter_startx), 2)

        radcornershelter = atan2(shelter_starty, shelter_startx)
        radcornershelter %= 2 * pi
        degcornershelter = round(degrees(radcornershelter), 2)

        foodtime = round(initialfoodtime - foodtimemultiplier * timer()) + foragefood
        watertime = round(initialwatertime - watertimemultiplier * timer()) + foragewater

        X = np.matrix([distfood,degfood,distwater,degwater,distcorner,degcorner,distcornerfood,degcornerfood,
                       distcornerwater,degcornerwater,distcornershelter,degcornershelter,foodtime,watertime,daytimer])
        a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
        AIPredict = np.array(np.argmax(h, axis=1) + 1)
        print(AIPredict)
        #file.write(str(distfood) + "," + str(degfood) + "," + str(foodtime) + "," + str(distwater) + "," + str(degwater) + "," + str(watertime) + "," + str(AIPredict) + "\n")
        #file.flush()

        if AIPredict == 1:
            x_change = -5
            y_change = 5
        if AIPredict == 2:
            y_change = 5
        if AIPredict == 3:
            x_change = 5
            y_change = 5
        if AIPredict == 4:
            x_change = -5
        if AIPredict == 5:
            y_change = 0
        if AIPredict == 6:
            x_change = 5
        if AIPredict == 7:
            x_change = -5
            y_change = -5
        if AIPredict == 8:
            y_change = -5
        if AIPredict == 9:
            x_change = 5
            y_change = -5

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        if x - 20 > display_width - animal_width:               #-20   -5
            x_change = -1
        if x < -5:       # -20   -5
            x_change = 1
        if y-50 > display_height - animal_height:              #-1xdif-50   -5
            y_change = -1
        if y < -5:
            y_change = 1


        if y > food_starty and y < food_starty + food_height or y + animal_height > food_starty and y + animal_height < food_starty + food_height:

            if x > food_startx and x < food_startx + food_width or x + animal_width > food_startx and x + animal_width < food_startx + food_width:
                food_startx = random.randrange(0 + display_width * 0.1, display_width - display_width * 0.1)
                food_starty = random.randrange(0 + display_height * 0.1, display_height - display_height * 0.1)
                scorefood += 1
                foragefood += 10

        if y > water_starty and y < water_starty + water_height or y + animal_height > water_starty and y + animal_height < water_starty + water_height:

            if x > water_startx and x < water_startx + water_width or x + animal_width > water_startx and x + animal_width < water_startx + water_width:
                water_startx = random.randrange(0 + display_width * 0.1, display_width - display_width * 0.1)
                water_starty = random.randrange(0 + display_height * 0.1, display_height - display_height * 0.1)
                scorewater += 1
                foragewater += 10

        pygame.display.update()
        clock.tick(60)

game_intro()
game_loop()
pygame.quit()
quit()
file.close()