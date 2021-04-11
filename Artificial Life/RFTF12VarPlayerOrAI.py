import pygame
import numpy as np
import pandas as pd
import random
import sys
from math import sqrt, fabs, atan2, degrees, pi
from timeit import default_timer as timer
from sklearn.preprocessing import OneHotEncoder
import time
# import tensorflow as tf
# from tensorflow.python.ops import rnn, rnn_cell

sys.setrecursionlimit(10000)


def recurrent_neural_network(x, n_classes, n_features, rnn_size, n_chunks, chunk_size):
    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
             'biases': tf.Variable(tf.random_normal([n_classes]))}
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(0, n_chunks, x)
    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size, state_is_tuple=True)
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']
    return output


def food_counter(count):
    font = pygame.font.SysFont("comicsansms", 25)
    text = font.render("Food Eaten: " + str(count), True, black)
    gameDisplay.blit(text, (0, 0))


def water_counter(count):
    font = pygame.font.SysFont("comicsansms", 25)
    text = font.render("Water Consumed: " + str(count), True, black)
    gameDisplay.blit(text, (0, display_height * 0.04))


def energy_counter(count):
    font = pygame.font.SysFont("comicsansms", 25)
    text = font.render("Energy Restored: " + str(count), True, black)
    gameDisplay.blit(text, (0, display_height * 0.08))


def hunger(hungertimer, initialtime, forage):
    font = pygame.font.SysFont("comicsansms", 25)
    text = font.render("Hunger: " + str(round(initialtime - hungertimer) + forage), True, black)
    gameDisplay.blit(text, (display_width - 175, 0))
    if (initialtime - hungertimer + forage) < 0:
        edge_contact()
    return


def thirst(thirsttimer, initialtime, forage):
    font = pygame.font.SysFont("comicsansms", 25)
    text = font.render("Thirst: " + str(round(initialtime - thirsttimer) + forage), True, black)
    gameDisplay.blit(text, (display_width - 175, display_height * 0.04))
    if (initialtime - thirsttimer + forage) < 0:
        edge_contact()
    return


def energy(energytimer, initialtime, forage):
    font = pygame.font.SysFont("comicsansms", 25)
    text = font.render("Energy: " + str(round(initialtime - energytimer + forage)), True, black)
    gameDisplay.blit(text, (display_width - 175, display_height * 0.08))
    if (initialtime - energytimer + forage) < 0:
        edge_contact()
    return


def currenttime(timer):
    font = pygame.font.SysFont("comicsansms", 25)
    text = font.render("Time: " + str(round(timer)), True, black)
    gameDisplay.blit(text, (display_width / 2 - 50, display_height * 0.01))
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

        button("Play Again", display_width * .3, display_height * (5 / 8), 100, 50, green, bright_green, game_loop)
        button("Quit", display_width * .6, display_height * (5 / 8), 100, 50, red, bright_red, quit)

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


def get_item(degitem, x, y):
    x_change = 0
    y_change = 0

    if 28 < degitem < 61:
        x_change = 5
        y_change = 5
    elif 60 < degitem < 121:
        x_change = 5
    elif 120 < degitem < 151:
        x_change = 5
        y_change = -5
    elif 150 < degitem < 211:
        y_change = -5
    elif 210 < degitem < 241:
        x_change = -5
        y_change = -5
    elif 240 < degitem < 301:
        x_change = -5
    elif 300 < degitem < 331:
        x_change = -5
        y_change = 5
    elif 330 < degitem < 361 or degitem < 29:
        y_change = 5

    x += x_change
    y += y_change
    return x, y


def food_move(food_startx, food_starty, foodmovedecision):
    foodmove = random.randrange(1, 9, 1)
    x_change = 0
    y_change = 0
    if foodmove == 4:
        x_change = -5
    if foodmove == 6:
        x_change = 5
    if foodmove == 8:
        y_change = -5
    if foodmove == 2:
        y_change = 5
    if foodmove == 1:
        x_change = -5
        y_change = 5
    if foodmove == 5:
        x_change = 0
        y_change = 0
    if foodmove == 3:
        x_change = 5
        y_change = 5
    if foodmove == 7:
        x_change = -5
        y_change = -5
    if foodmove == 9:
        x_change = 5
        y_change = -5
    food_startx += x_change
    food_starty += y_change
    foodmovedecision = 0
    return foodmovedecision


def game_intro():
    intro = True

    while intro:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        gameDisplay.fill(white)
        largeText = pygame.font.SysFont("comicsansms", 115)
        TextSurf, TextRect = text_objects("Artificial Life", largeText)
        TextRect.center = ((display_width / 2), (display_height / 2))
        gameDisplay.blit(TextSurf, TextRect)

        button("GO!", display_width * .3, display_height * (5 / 8), 100, 50, green, bright_green, game_loop)
        button("Quit", display_width * .6, display_height * (5 / 8), 100, 50, red, bright_red, quit)

        pygame.display.update()
        clock.tick(15)


def game_loop():
    animal_width = 73
    animal_height = 73

    global file

    x = (display_width * 0.45)
    y = (display_height * 0.8)

    x_change = 0
    y_change = 0

    food_startx = random.randrange(0 + display_width * 0.1, display_width - display_width * .1)
    food_starty = random.randrange(0 + display_height * 0.1, display_height - display_height * .1)
    food_width = 50
    food_height = 50

    water_startx = random.randrange(0 + display_width * 0.1, display_width - display_width * .1)
    water_starty = random.randrange(0 + display_height * 0.1, display_height - display_height * .1)
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
    energytimemultiplier = 2.5
    daytimemultiplier = 0.2
    initialfoodtime = 100
    initialwatertime = 100
    initialenergytime = 100
    keypress = 5
    daytimercount = False
    daycount = 0
    starttime = time.time()
    movingfoodtimer = 0
    foodmovedecision = 1

    gameExit = False
    decisiontime = 1
    action = 1

    while not gameExit:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        x += x_change
        y += y_change

        totaltimer = time.time() - starttime

        if foodmovedecision == 1:
            foodmovedecision = food_move(food_startx, food_starty, foodmovedecision)

        if totaltimer % 5 == 0:
            foodmovedecision = 1

        daytimer = round(((time.time() - starttime) * daytimemultiplier), 2)

        if daytimercount == True:
            daytimer = round((daytimer - 4 * daycount), 2)

        if daytimer < 1:
            gameDisplay.fill(morning_color)
            food_color = food_green
            water_color = water_blue
            nightonehot = 0
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
            nightonehot = 1
        elif daytimer < 5:
            daytimercount = True
            daycount = daycount + 1
            food_color = food_green
            water_color = water_blue

        food(food_startx, food_starty, food_width, food_height, food_color)
        water(water_startx, water_starty, water_width, water_height, water_color)

        animal(x, y)
        food_counter(scorefood)
        water_counter(scorewater)
        energy_counter(scoreenergy)
        hunger(foodtimemultiplier * totaltimer, initialfoodtime, foragefood)
        thirst(watertimemultiplier * totaltimer, initialwatertime, foragewater)
        energy(energytimemultiplier * totaltimer, initialenergytime, forageenergy)
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

        distxshelter = shelter_startx - x
        distyshelter = shelter_starty - y
        distshelter = round(sqrt(fabs(distxshelter) + fabs(distyshelter)), 2)

        radshelter = atan2(distxshelter, distyshelter)
        radshelter %= 2 * pi
        degshelter = round(degrees(radshelter), 2)

        foodtime = round(initialfoodtime - foodtimemultiplier * totaltimer) + foragefood
        watertime = round(initialwatertime - watertimemultiplier * totaltimer) + foragewater
        energytime = round(initialenergytime - energytimemultiplier * totaltimer) + forageenergy

        if userorai in ["p"]:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_KP1:
                        action = 1
                    if event.key == pygame.K_KP2:
                        action = 2
                    if event.key == pygame.K_KP3:
                        action = 3

        if userorai in ["a"]:
            input_data = np.array(
                [distfood, degfood, distwater, degwater, distshelter, degshelter, foodtime, watertime, energytime,
                 daytimer, nightonehot])
        # input_data = input_data.reshape((n_chunks,chunk_size))

        if decisiontime == 1:
            action = rfclf.predict(input_data)
            # with tf.Session() as sess:
            #	sess.run(tf.initialize_all_variables())
            #	saver.restore(sess,"ALData/BiNNmodel.ckpt")
            #	action = (sess.run(tf.argmax(prediction.eval(feed_dict={xtf:[input_data]}),1)))
            decisiontime = 0

        if action == 1:
            x, y = get_item(degfood, x, y)
        elif action == 2:
            x, y = get_item(degwater, x, y)
        elif action == 3:
            x, y = get_item(degshelter, x, y)

        print(action)

        # file.write(str(distfood) + "," + str(degfood) + "," + str(distwater) + "," + str(degwater) + "," + str(distshelter) + "," +
        #	str(degshelter) + "," + str(foodtime) + "," + str(watertime) + "," + str(energytime) + "," + str(daytimer) + "," +
        #	str(nightonehot) + "," + str(action) + "\n")
        # file.flush()

        if x - 120 > display_width - animal_width or x < -105:
            edge_contact()

        if y - 150 > display_height - animal_height or y < -105:
            edge_contact()

        if y > shelter_starty - 5 and y < shelter_starty + shelter_height and x > shelter_startx - 5 and x < shelter_startx + shelter_width and energytime < 100:
            if round(totaltimer, 2) % 1.5 == 0:
                decisiontime = 1
                forageenergy += 0.2
            if daytimer > 3:
                forageenergy += 0.5

        if (y > food_starty and y < food_starty + food_height and x > food_startx and x < food_startx + food_width) or \
                (y + animal_width > food_starty and y + animal_width < food_starty + food_height and x + animal_width > food_startx and x + animal_width < food_startx + food_width) or \
                ((y + animal_width / 2) > food_starty and (y + animal_width / 2) < food_starty + food_height and
                (x + animal_width / 2) > food_startx and (x + animal_width / 2) < food_startx + food_width):
            food_startx = random.randrange(0 + display_width * 0.1, display_width - display_width * .1)
            food_starty = random.randrange(0 + display_height * 0.1, display_height - display_height * .1)
            scorefood += 1
            foragefood += 5
            decisiontime = 1

        if (y > water_starty and y < water_starty + water_height and x > water_startx and x < water_startx + water_width) or \
                (y + animal_width > water_starty and y + animal_width < water_starty + water_height and x + animal_width > water_startx and x + animal_width < water_startx + water_width) or \
                ((y + animal_width / 2) > water_starty and (y + animal_width / 2) < water_starty + water_height and
                (x + animal_width / 2) > water_startx and (x + animal_width / 2) < water_startx + water_width):
            water_startx = random.randrange(0 + display_width * 0.1, display_width - display_width * .1)
            water_starty = random.randrange(0 + display_height * 0.1, display_height - display_height * .1)
            scorewater += 1
            foragewater += 5
            decisiontime = 1

        pygame.display.update()
        clock.tick(60)

oldfile = open("ALData/11VarALData.txt", "r")
olddata = oldfile.read()
oldfile.close()
file = open("ALData/11VarALData.txt", "w")
file.write(olddata)
file.flush()

pygame.init()
pygame.mixer.music.load("ALData/Corridors_of_Time_Piano_.wav")
pygame.mixer.music.play(-1)

AnimalImg = pygame.image.load('ALData/animalimg.png')
ShelterImg = pygame.image.load('ALData/Shelter.png')
Icon = pygame.image.load('ALData/animalimg.png')
pygame.display.set_icon(Icon)

display_width = 1200
display_height = 800

black = (0, 0, 0)
white = (255, 255, 255)
red = (200, 0, 0)
green = (0, 200, 0)
bright_red = (255, 0, 0)
bright_green = (0, 255, 0)

gameDisplay = pygame.display.set_mode((display_width, display_height))
pygame.display.set_caption('Artificial Life')
clock = pygame.time.Clock()

filefeatures = np.loadtxt("ALData/nclassesfeatures.txt", delimiter=",")
rnnmodelaccuracy = np.loadtxt("ALData/rnnmodelaccuracy.txt", delimiter=",")

n_classes = int(filefeatures[0]) + 1
n_features = int(filefeatures[1])
n_chunks = int(rnnmodelaccuracy[4])
chunk_size = int(rnnmodelaccuracy[5])
rnn_size = 256

# xtf = tf.placeholder('float', [None, n_chunks, chunk_size])
# prediction = recurrent_neural_network(xtf, n_classes, n_features, rnn_size, n_chunks, chunk_size)
# saver = tf.train.Saver()

userorai = input("Select Player or AI (p/a)")

rfclf = pd.read_pickle('ALData/RFpickle.pickle')

game_intro()
game_loop()
pygame.quit()
quit()
file.close()