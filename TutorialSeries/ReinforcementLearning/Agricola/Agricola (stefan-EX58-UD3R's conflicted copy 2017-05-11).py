import pygame
import random

class Players:

    def __init__(self, name, colour, turn, food, startingplayer, xmenulocation, fam1posx, fam1posy, fam2posx, fam2posy,
                 fam3posx, fam3posy, fam4posx, fam4posy, fam5posx, fam5posy, fieldx1, fieldy1, fieldx2, fieldy2):
        self.name = name
        self.colour = colour
        self.turn = turn
        self.fields = 0
        self.fences = 0
        self.pastures = 0
        self.emptyspace = 13
        self.stables = 0
        self.woodroom = 2
        self.clayroom = 0
        self.stoneroom = 0
        self.familysize = 2
        self.bonuspts = 0
        self.wood = 0
        self.clay = 0
        self.stone = 0
        self.reed = 0
        self.sheep = 0
        self.boar = 0
        self.cattle = 0
        self.grain = 0
        self.vegatable = 0
        self.reed = 0
        self.food = food
        self.familymember = 1
        self.startingplayer = startingplayer
        self.startingplayertoggle = 0

        self.fieldx1 = int(fieldx1)
        self.fieldy1 = int(fieldy1)
        self.fieldx2 = int(fieldx2)
        self.fieldy2 = int(fieldy2)

        self.xmenulocation = xmenulocation
        self.fam1posx = int(fam1posx)
        self.fam1posy = int(fam1posy)
        self.fam2posx = int(fam2posx)
        self.fam2posy = int(fam2posy)
        self.fam3posx = int(fam3posx)
        self.fam3posy = int(fam3posy)
        self.fam4posx = int(fam4posx)
        self.fam4posy = int(fam4posy)
        self.fam5posx = int(fam5posx)
        self.fam5posy = int(fam5posy)

pygame.init()
crash_sound = pygame.mixer.Sound("Crash.wav")
pygame.mixer.music.load('F:/PythonDataBackUp/Agricola/StardewValleyOST/01 - Stardew Valley Overture.mp3')

ingamemusicdict = {1: 'F:/PythonDataBackUp/Agricola/StardewValleyOST/02 - Cloud Country.mp3',
                   2: 'F:/PythonDataBackUp/Agricola/StardewValleyOST/04 - Settling In.mp3',
                   3: "F:/PythonDataBackUp/Agricola/StardewValleyOST/05 - Spring (It's A Big World Outside).mp3",
                   4: 'F:/PythonDataBackUp/Agricola/StardewValleyOST/06 - Spring (The Valley Comes Alive).mp3',
                   5: 'F:/PythonDataBackUp/Agricola/StardewValleyOST/14 - Summer (The Sun Can Bend An Orange Sky).mp3',
                   6: 'F:/PythonDataBackUp/Agricola/StardewValleyOST/29 - Winter (Ancient).mp3',
                   7: 'F:/PythonDataBackUp/Agricola/StardewValleyOST/56 - Mines (Crystal Bells).mp3',
                   8: "F:/PythonDataBackUp/Agricola/StardewValleyOST/22 - Fall (Raven's Descent).mp3",
                   9: 'F:/PythonDataBackUp/Agricola/StardewValleyOST/08 - Pelican Town.mp3',
                   10: 'F:/PythonDataBackUp/Agricola/StardewValleyOST/11 - Distant Banjo.mp3'}

ingamemusicorder = random.sample(range(1, 11), 10)

display_width = 1280
display_height = 800

black = (0, 0, 0)
white = (255, 255, 255)
blue = (232, 162, 0)
red = (36, 28, 237)
green = (76, 177, 34)
yellow = (0, 242, 255)
purple = (164, 73, 163)
bright_red = (255, 0, 0)
bright_green = (0, 255, 0)

boardwood = 3
boardclay = 1
boardreed = 1
boardfish = 1
boardsheep = 1
boardboar = 1
boardcow = 1
boardstone1 = 1
boardstone2 = 1
boardstartingplayerfood = 1

toggleBuildRooms = 0
toggleStartingPlayer = 0
toggleTakeOneGrain = 0
togglePlowField = 0
toggleBuildStable = 0
toggleDayLaborer = 0
toggleWood = 0
toggleClay = 0
toggleReed = 0
toggleFish = 0
toggleSheep = 0
toggleFences = 0
toggleMajorImprovement = 0
toggleSowAndBake = 0

gameDisplay = pygame.display.set_mode((display_width, display_height))
pygame.display.set_caption('Agricola')
clock = pygame.time.Clock()

gameIcon = pygame.image.load('Agricola_box.jpg')
carImg = pygame.image.load('Agricola_box.jpg')

pygame.display.set_icon(gameIcon)

action = 'start'
prevaction = None
stage = 1

buttonW = display_width / 10
smallbuttonH = display_height / 1.5 / 3 / 2
largebuttonH = smallbuttonH * 2
ItemGap = largebuttonH / 10
famx = buttonW / 12

p1fam1x, p1fam1y = int(4 * famx + buttonW), int(largebuttonH * 3 + ItemGap * 13)
p1fam2x, p1fam2y = int(6 * famx + buttonW), int(p1fam1y)
p1fam3x, p1fam3y = int(p1fam1x), int(largebuttonH * 3 + ItemGap * 14)
p1fam4x, p1fam4y = int(p1fam2x), int(p1fam3y)
p1fam5x, p1fam5y = int(8 * famx + buttonW), int(largebuttonH * 3 + ItemGap * (13 + 14) / 2)

p2fam1x, p2fam1y = int(2 * buttonW + p1fam1x), int(p1fam1y)
p2fam2x, p2fam2y = int(2 * buttonW + p1fam2x), int(p1fam2y)
p2fam3x, p2fam3y = int(2 * buttonW + p1fam3x), int(p1fam3y)
p2fam4x, p2fam4y = int(2 * buttonW + p1fam4x), int(p1fam4y)
p2fam5x, p2fam5y = int(2 * buttonW + p1fam5x), int(p1fam5y)

p3fam1x, p3fam1y = int(4 * buttonW + p1fam1x), int(p1fam1y)
p3fam2x, p3fam2y = int(4 * buttonW + p1fam2x), int(p1fam2y)
p3fam3x, p3fam3y = int(4 * buttonW + p1fam3x), int(p1fam3y)
p3fam4x, p3fam4y = int(4 * buttonW + p1fam4x), int(p1fam4y)
p3fam5x, p3fam5y = int(4 * buttonW + p1fam5x), int(p1fam5y)

p4fam1x, p4fam1y = int(6 * buttonW + p1fam1x), int(p1fam1y)
p4fam2x, p4fam2y = int(6 * buttonW + p1fam2x), int(p1fam2y)
p4fam3x, p4fam3y = int(6 * buttonW + p1fam3x), int(p1fam3y)
p4fam4x, p4fam4y = int(6 * buttonW + p1fam4x), int(p1fam4y)
p4fam5x, p4fam5y = int(6 * buttonW + p1fam5x), int(p1fam5y)

p5fam1x, p5fam1y = int(8 * buttonW + p1fam1x), int(p1fam1y)
p5fam2x, p5fam2y = int(8 * buttonW + p1fam2x), int(p1fam2y)
p5fam3x, p5fam3y = int(8 * buttonW + p1fam3x), int(p1fam3y)
p5fam4x, p5fam4y = int(8 * buttonW + p1fam4x), int(p1fam4y)
p5fam5x, p5fam5y = int(8 * buttonW + p1fam5x), int(p1fam5y)

fieldx1 = int(0.01 * display_width + buttonW)
fieldy1 = int(0.02 * display_height + largebuttonH * 3)
fieldx2 = int(2 * buttonW - 0.01 * display_width)
fieldy2 = int(0.02 * display_height + largebuttonH * 4)
buildroomx = int(2 * buttonW + 0.02 * display_width)
buildroomy = int(0.02 * display_height)
startingplayerx = int(2 * buttonW + 0.02 * display_width)
startingplayery = int(0.02 * display_height + smallbuttonH)
takeonegrainx = int(2 * buttonW + 0.02 * display_width)
takeonegrainy = int(0.02 * display_height + smallbuttonH * 2)
plowfieldx = int(2 * buttonW + 0.02 * display_width)
plowfieldy = int(0.02 * display_height + smallbuttonH * 3)
buildstablex = int(2 * buttonW + 0.02 * display_width)
buildstabley = int(0.02 * display_height + smallbuttonH * 4)
daylaborerx = int(2 * buttonW + 0.02 * display_width)
daylaborery = int(0.02 * display_height + smallbuttonH * 5)
stage1x = int(3 * buttonW + 0.02 * display_width)
stage1y = int(0.02 * display_height)
woodx = int(3 * buttonW + 0.02 * display_width)
woody = int(0.02 * display_height + smallbuttonH * 2)
clayx = int(3 * buttonW + 0.02 * display_width)
clayy = int(0.02 * display_height + smallbuttonH * 3)
reedx = int(3 * buttonW + 0.02 * display_width)
reedy = int(0.02 * display_height + smallbuttonH * 4)
fishx = int(3 * buttonW + 0.02 * display_width)
fishy = int(0.02 * display_height + smallbuttonH * 5)

Playertext = pygame.font.SysFont("comicsansms", 15)

pause = False

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

def quitgame():
    pygame.quit()
    quit()

def unpause():
    pygame.mixer.music.unpause()
    global pause
    pause = False

def paused():

    pygame.mixer.music.pause()
    largeText = pygame.font.SysFont("comicsansms", 115)
    TextSurf, TextRect = text_objects("Paused", largeText)
    TextRect.center = ((display_width / 2), (display_height / 2))
    gameDisplay.blit(TextSurf, TextRect)

    while pause:
        for event in pygame.event.get():
            # print(event)
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        button("Continue", 150, 450, 100, 50, green, bright_green, unpause)
        button("Quit", 550, 450, 100, 50, red, bright_red, quitgame)
        pygame.display.update()
        clock.tick(15)

def game_intro():
    intro = True
    pygame.mixer.music.play(-1)

    bg = pygame.image.load("Intro_background_1280x800.jpg")
    gameDisplay.blit(bg, (0, 0))

    while intro:
        for event in pygame.event.get():
            # print(event)
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        largeText = pygame.font.SysFont("comicsansms", 115)
        TextSurf, TextRect = text_objects("Agricola", largeText)
        TextRect.center = ((display_width / 2), (display_height / 7))
        gameDisplay.blit(TextSurf, TextRect)

        button("2 Player", display_width/3, display_height/1.5, 100, 50, green, bright_green, game_loop)
        #button("3 Player", display_width/3, display_height/1.25, 1xdif-100, 1xdif-50, green, bright_green, game_loop)
        #button("4 Player", display_width/1.75, display_height/1.5, 1xdif-100, 1xdif-50, green, bright_green, game_loop)
        #button("5 Player", display_width/1.75, display_height/1.25, 1xdif-100, 1xdif-50, green, bright_green, game_loop)
        button("Quit", display_width/2.2, display_height/1.37, 100, 50, red, bright_red, quitgame)

        pygame.display.update()
        clock.tick(15)

def play_music():
    pygame.mixer.music.load(ingamemusicdict[ingamemusicorder[0]])
    for i, value in ingamemusicdict.items():
        pygame.mixer.music.queue(value)
    pygame.mixer.music.play(1)

def draw_borders():
    gameDisplay.fill(white)
    # Borders
    pygame.draw.line(gameDisplay, black, (0, 0), (display_width, 0))
    pygame.draw.line(gameDisplay, black, (0, display_height), (display_width, display_height))
    pygame.draw.line(gameDisplay, black, (0, 0), (0, display_height))
    pygame.draw.line(gameDisplay, black, (display_width, 0), (display_width, display_height))
    # FullColoumns
    pygame.draw.line(gameDisplay, black, (display_width / 10, 0), (display_width / 10, display_height / 1.5))
    pygame.draw.line(gameDisplay, black, (2 * display_width / 10, 0), (2 * display_width / 10, display_height / 1.5))
    pygame.draw.line(gameDisplay, black, (3 * display_width / 10, 0), (3 * display_width / 10, display_height / 1.5))
    pygame.draw.line(gameDisplay, black, (4 * display_width / 10, 0), (4 * display_width / 10, display_height / 1.5))
    pygame.draw.line(gameDisplay, black, (5 * display_width / 10, 0), (5 * display_width / 10, display_height / 1.5))
    pygame.draw.line(gameDisplay, black, (6 * display_width / 10, 0), (6 * display_width / 10, display_height / 1.5))
    # FullRows
    pygame.draw.line(gameDisplay, black, (0, display_height / 1.5), (display_width, display_height / 1.5))
    pygame.draw.line(gameDisplay, black, (0, 5 + display_height / 1.5), (display_width, 5 + display_height / 1.5))
    pygame.draw.line(gameDisplay, black, (0, display_height / 1.5 / 3), (display_width, display_height / 1.5 / 3))
    pygame.draw.line(gameDisplay, black, (0, 2 * display_height / 1.5 / 3), (display_width, 2 * display_height / 1.5 / 3))
    # PartRows
    pygame.draw.line(gameDisplay, black, (2 * display_width / 10, 0), (2 * display_width / 10, display_height / 1.5))
    pygame.draw.line(gameDisplay, black, (2 * display_width / 10, display_height / 1.5 / 3 / 2), (3 * display_width / 10, display_height / 1.5 / 3 / 2))
    pygame.draw.line(gameDisplay, black, (2 * display_width / 10, 3 * display_height / 1.5 / 3 / 2), (4 * display_width / 10, 3 * display_height / 1.5 / 3 / 2))
    pygame.draw.line(gameDisplay, black, (2 * display_width / 10, 5 * display_height / 1.5 / 3 / 2), (4 * display_width / 10, 5 * display_height / 1.5 / 3 / 2))
    # PartColumns
    pygame.draw.line(gameDisplay, black, (7 * display_width / 10, display_height / 1.5 / 3), (7 * display_width / 10, display_height / 1.5))
    pygame.draw.line(gameDisplay, black, (8 * display_width / 10, display_height / 1.5 / 3), (8 * display_width / 10, display_height / 1.5))
    pygame.draw.line(gameDisplay, black, (9 * display_width / 10, display_height / 1.5 / 3), (9 * display_width / 10, display_height / 1.5))

def draw_player_inventory(Player):
    TextSurf, TextRect = text_objects("{}'s Farm".format(Player.name), Playertext)
    TextRect = (Player.xmenulocation, 0.01 * display_height + largebuttonH * 3)
    gameDisplay.blit(TextSurf, TextRect)
    TextSurf, TextRect = text_objects('{} Food'.format(Player.food), Playertext)
    TextRect = (Player.xmenulocation, largebuttonH * 3 + ItemGap * 2)
    gameDisplay.blit(TextSurf, TextRect)
    TextSurf, TextRect = text_objects('{} Wood'.format(Player.wood), Playertext)
    TextRect = (Player.xmenulocation, largebuttonH * 3 + ItemGap * 3)
    gameDisplay.blit(TextSurf, TextRect)
    TextSurf, TextRect = text_objects('{} Clay'.format(Player.clay), Playertext)
    TextRect = (Player.xmenulocation, largebuttonH * 3 + ItemGap * 4)
    gameDisplay.blit(TextSurf, TextRect)
    TextSurf, TextRect = text_objects('{} Stone'.format(Player.stone), Playertext)
    TextRect = (Player.xmenulocation, largebuttonH * 3 + ItemGap * 5)
    gameDisplay.blit(TextSurf, TextRect)
    TextSurf, TextRect = text_objects('{} Reed'.format(Player.reed), Playertext)
    TextRect = (Player.xmenulocation, largebuttonH * 3 + ItemGap * 6)
    gameDisplay.blit(TextSurf, TextRect)
    TextSurf, TextRect = text_objects('{} Grain'.format(Player.grain), Playertext)
    TextRect = (Player.xmenulocation, largebuttonH * 3 + ItemGap * 7)
    gameDisplay.blit(TextSurf, TextRect)
    TextSurf, TextRect = text_objects('{} Vegatable'.format(Player.vegatable), Playertext)
    TextRect = (Player.xmenulocation, largebuttonH * 3 + ItemGap * 8)
    gameDisplay.blit(TextSurf, TextRect)
    TextSurf, TextRect = text_objects('{} Sheep'.format(Player.sheep), Playertext)
    TextRect = (Player.xmenulocation, largebuttonH * 3 + ItemGap * 9)
    gameDisplay.blit(TextSurf, TextRect)
    TextSurf, TextRect = text_objects('{} Boar'.format(Player.boar), Playertext)
    TextRect = (Player.xmenulocation, largebuttonH * 3 + ItemGap * 10)
    gameDisplay.blit(TextSurf, TextRect)
    TextSurf, TextRect = text_objects('{} Cattle'.format(Player.cattle), Playertext)
    TextRect = (Player.xmenulocation, largebuttonH * 3 + ItemGap * 11)
    gameDisplay.blit(TextSurf, TextRect)
    TextSurf, TextRect = text_objects('{} Family'.format(Player.familysize), Playertext)
    TextRect = (Player.xmenulocation, largebuttonH * 3 + ItemGap * 12)
    gameDisplay.blit(TextSurf, TextRect)
    TextSurf, TextRect = text_objects('{} Bonus Pts'.format(Player.bonuspts), Playertext)
    TextRect = (Player.xmenulocation, largebuttonH * 3 + ItemGap * 13)
    gameDisplay.blit(TextSurf, TextRect)
    if Player.startingplayer == 1:
        TextSurf, TextRect = text_objects('SP', Playertext)
        TextRect = (Player.xmenulocation + buttonW/1.1, largebuttonH * 3 + ItemGap * 12.5)
        gameDisplay.blit(TextSurf, TextRect)

    #Player Icons
    pygame.draw.line(gameDisplay, black, (Player.fieldx1, Player.fieldy1), (Player.fieldx2, Player.fieldy1))
    pygame.draw.line(gameDisplay, black, (Player.fieldx1, Player.fieldy1), (Player.fieldx1, Player.fieldy2))
    pygame.draw.line(gameDisplay, black, (Player.fieldx1, Player.fieldy2), (Player.fieldx2, Player.fieldy2))
    pygame.draw.line(gameDisplay, black, (Player.fieldx2, Player.fieldy1), (Player.fieldx2, Player.fieldy2))

def draw_player_position(Player):
    pygame.draw.circle(gameDisplay, Player.colour, (Player.fam1posx, Player.fam1posy), 8)
    pygame.draw.circle(gameDisplay, Player.colour, (Player.fam2posx, Player.fam2posy), 8)
    if Player.familysize >= 3:
        pygame.draw.circle(gameDisplay, Player.colour, (Player.fam3posx, Player.fam3posy), 8)
    if Player.familysize >= 4:
        pygame.draw.circle(gameDisplay, Player.colour, (Player.fam4posx, Player.fam4posy), 8)
    if Player.familysize >= 5:
        pygame.draw.circle(gameDisplay, Player.colour, (Player.fam5posx, Player.fam5posy), 8)

def Build_Room():
    global action, toggleBuildRooms
    if toggleBuildRooms == 0:
        action = 'buildrooms'
        toggleBuildRooms = 1

def Starting_Player():
    global action, toggleStartingPlayer
    if toggleStartingPlayer == 0:
        action = 'startingplayer'
        toggleStartingPlayer = 1

def Take_1_Grain():
    global action, toggleTakeOneGrain
    if toggleTakeOneGrain == 0:
        action = 'grain'
        toggleTakeOneGrain = 1

def Plow_Field():
    global action, togglePlowField
    if togglePlowField == 0:
        action = 'plowfield'
        togglePlowField = 1

def Build_Stable():
    global action, toggleBuildStable
    if toggleBuildStable == 0:
        action = 'buildstable'
        toggleBuildStable = 1

def Day_Laborer():
    global action, toggleDayLaborer
    if toggleDayLaborer == 0:
        action = 'daylaborer'
        toggleDayLaborer = 1

def Wood():
    global action, toggleWood
    if toggleWood == 0:
        action = 'wood'
        toggleWood = 1

def Clay():
    global action, toggleClay
    if toggleClay == 0:
        action = 'clay'
        toggleClay = 1

def Reed():
    global action, toggleReed
    if toggleReed == 0:
        action = 'reed'
        toggleReed = 1

def Fish():
    global action, toggleFish
    if toggleFish == 0:
        action = "fish"
        toggleFish = 1

def Sheep():
    global action, toggleSheep
    if toggleSheep == 0:
        action = "sheep"
        toggleSheep = 1

def Fences():
    global action, toggleFences
    if toggleFences == 0:
        action = "fences"
        toggleFences = 1

def SowAndBake():
    global action, toggleSowAndBake
    if toggleSowAndBake == 0:
        action = "sowandbake"
        toggleSowAndBake = 1

def MajorImprovement():
    global action, toggleMajorImprovement
    if toggleMajorImprovement == 0:
        action = "majorimprovement"
        toggleMajorImprovement = 1

def FamilyGrowth(): pass

def Stone1(): pass

def Renovate(): pass

def Vegetable(): pass

def Boar(): pass

def Stone2(): pass

def Cattle(): pass

def FamilyGrowthWithoutRoom(): pass

def PlowAndSow(): pass

def perform_action(Player):
    global action, prevaction, boardwood, boardclay, boardreed, boardfish, boardsheep, boardboar, boardcow, \
        boardstone1, boardstone2, boardstartingplayerfood
    if Player.familymember == 1:
        if action == 'buildrooms':
            if Player.wood > 4 and Player.reed > 1:
                Player.wood -= 5
                Player.reed -= 2
                Player.woodroom += 1
                Player.fam1posx = buildroomx
                Player.fam1posy = buildroomy
            else:
                action = None
        if action == 'startingplayer':
            Player.food += boardstartingplayerfood
            boardstartingplayerfood = 0
            Player.startingplayertoggle = 1
            Player.fam1posx = startingplayerx
            Player.fam1posy = startingplayery
        if action == 'grain':
            Player.grain += 1
            Player.fam1posx = takeonegrainx
            Player.fam1posy = takeonegrainy
        if action == 'plowfield':
            Player.fields += 1
            Player.fam1posx = plowfieldx
            Player.fam1posy = plowfieldy
        if action == 'buildstable':
            Player.stables += 1
            Player.fam1posx = buildstablex
            Player.fam1posy = buildstabley
        if action == 'daylaborer':
            Player.food += 1
            Player.stone += 1
            Player.fam1posx = daylaborerx
            Player.fam1posy = daylaborery
        if action == 'wood':
            Player.wood += boardwood
            boardwood = 0
            Player.fam1posx = woodx
            Player.fam1posy = woody
        if action == 'clay':
            Player.clay += boardclay
            boardclay = 0
            Player.fam1posx = clayx
            Player.fam1posy = clayy
        if action == 'reed':
            Player.reed += boardreed
            boardreed = 0
            Player.fam1posx = reedx
            Player.fam1posy = reedy
        if action == 'fish':
            Player.food += boardfish
            boardfish = 0
            Player.fam1posx = fishx
            Player.fam1posy = fishy
        if action == 'sheep':
            Player.sheep += boardsheep
            boardsheep = 0
            Player.fam1posx = stage1x
            Player.fam1posy = stage1y
        if action == 'fences':
            print('fences')
            Player.fam1posx = stage1x
            Player.fam1posy = stage1y
        if action == 'majorimprovement':
            print('majorimprovement')
            Player.fam1posx = stage1x
            Player.fam1posy = stage1y
        if action == 'sowandbake':
            print('sowandbake')
            Player.fam1posx = stage1x
            Player.fam1posy = stage1y
    elif Player.familymember == 2:
        if action == 'buildrooms':
            if Player.wood > 4 and Player.reed > 1:
                Player.wood -= 5
                Player.reed -= 2
                Player.woodroom += 1
                Player.fam2posx = buildroomx
                Player.fam2posy = buildroomy
            else:
                action = None
        if action == 'startingplayer':
            Player.food += boardstartingplayerfood
            boardstartingplayerfood = 0
            Player.fam2posx = startingplayerx
            Player.fam2posy = startingplayery
        if action == 'grain':
            Player.grain += 1
            Player.fam2posx = takeonegrainx
            Player.fam2posy = takeonegrainy
        if action == 'plowfield':
            Player.fields += 1
            Player.fam2posx = plowfieldx
            Player.fam2posy = plowfieldy
        if action == 'buildstable':
            Player.stables += 1
            Player.fam2posx = buildstablex
            Player.fam2posy = buildstabley
        if action == 'daylaborer':
            Player.food += 1
            Player.stone += 1
            Player.fam2posx = daylaborerx
            Player.fam2posy = daylaborery
        if action == 'wood':
            Player.wood += boardwood
            boardwood = 0
            Player.fam2posx = woodx
            Player.fam2posy = woody
        if action == 'clay':
            Player.clay += boardclay
            boardclay = 0
            Player.fam2posx = clayx
            Player.fam2posy = clayy
        if action == 'reed':
            Player.reed += boardreed
            boardreed = 0
            Player.fam2posx = reedx
            Player.fam2posy = reedy
        if action == 'fish':
            Player.food += boardfish
            boardfish = 0
            Player.fam2posx = fishx
            Player.fam2posy = fishy
        if action == 'sheep':
            Player.sheep += boardsheep
            boardsheep = 0
            Player.fam1posx = stage1x
            Player.fam1posy = stage1y
        if action == 'fences':
            print('fences')
            Player.fam1posx = stage1x
            Player.fam1posy = stage1y
        if action == 'majorimprovement':
            print('majorimprovement')
            Player.fam1posx = stage1x
            Player.fam1posy = stage1y
        if action == 'sowandbake':
            print('sowandbake')
            Player.fam1posx = stage1x
            Player.fam1posy = stage1y
    prevaction = action

def Player_Reset(Player, xfactor):
    Player.fam1posx = int(xfactor * buttonW + p1fam1x)
    Player.fam1posy = p1fam1y
    Player.fam2posx = int(xfactor * buttonW + p1fam2x)
    Player.fam2posy = p1fam2y
    Player.familymember = 1

def Board_Reset():
    global boardwood, boardclay, boardreed, boardfish, boardsheep, boardboar, boardcow, boardstone1, boardstone2, \
        boardstartingplayerfood, toggleBuildRooms, toggleStartingPlayer, toggleTakeOneGrain, togglePlowField, \
        toggleBuildStable, toggleDayLaborer, toggleWood, toggleClay, toggleReed, toggleFish, toggleSheep, action, stage
    action = None
    stage += 1
    toggleBuildRooms = 0
    toggleStartingPlayer = 0
    toggleTakeOneGrain = 0
    togglePlowField = 0
    toggleBuildStable = 0
    toggleDayLaborer = 0
    toggleWood = 0
    toggleClay = 0
    toggleReed = 0
    toggleFish = 0
    toggleSheep = 0
    boardstartingplayerfood += 1
    boardwood += 3
    boardclay += 1
    boardreed += 1
    boardfish += 1

def game_loop():
    global pause, action, stage, prevaction, mousetrigger, boardwood, boardclay, boardreed, boardfish, boardsheep, boardboar,\
        boardcow, boardstone1, boardstone2, boardstartingplayerfood

    numplayers = 2

    Player1 = Players("Bob", blue, 1, 2, 1, 0.01 * display_width, p1fam1x, p1fam1y, p1fam2x, p1fam2y, p1fam3x, p1fam3y,
                      p1fam4x, p1fam4y, p1fam5x, p1fam5y, fieldx1, fieldy1, fieldx2, fieldy2)
    Player2 = Players("Sue", red, 2, 3, 0, 0.01 * display_width + buttonW * 2, p2fam1x, p2fam1y, p2fam2x, p2fam2y,
                      p2fam3x, p2fam3y, p2fam4x, p2fam4y, p2fam5x, p2fam5y, 2 * buttonW + fieldx1, fieldy1,
                      2 * buttonW + fieldx2, fieldy2)
    Player3 = Players("John", black, 3, 3, 0, 0.01 * display_width + buttonW * 4, p3fam1x, p3fam1y, p3fam2x, p3fam2y,
                      p3fam3x, p3fam3y, p3fam4x, p3fam4y, p3fam5x, p3fam5y, 4 * buttonW + fieldx1, fieldy1,
                      4 * buttonW + fieldx2, fieldy2)
    Player4 = Players("Fred", yellow, 4, 3, 0, 0.01 * display_width + buttonW * 6, p4fam1x, p4fam1y, p4fam2x, p4fam2y,
                      p4fam3x, p4fam3y, p4fam4x, p4fam4y, p4fam5x, p4fam5y, 6 * buttonW + fieldx1, fieldy1,
                      6 * buttonW + fieldx2, fieldy2)
    Player5 = Players("Sarah", purple, 5, 3, 0, 0.01 * display_width + buttonW * 8, p5fam1x, p5fam1y, p5fam2x, p5fam2y,
                      p5fam3x, p5fam3y, p5fam4x, p5fam4y, p5fam5x, p5fam5y, 8 * buttonW + fieldx1, fieldy1,
                      8 * buttonW + fieldx2, fieldy2)
    play_music()

    stage = 1
    turn = 1

    stage1order = random.sample(range(1, 5), 4)
    stage2order = random.sample(range(1, 4), 3)
    stage3order = random.sample(range(1, 3), 2)
    stage4order = random.sample(range(1, 3), 2)
    stage5order = random.sample(range(1, 3), 2)

    gameExit = False
    while not gameExit:
        stage1dict = {1: '{} Sheep'.format(boardsheep), 2: 'Fences', 3: 'Maj Improve', 4: 'Sow & Bake'}
        stage2dict = {1: 'Fam Growth', 2: '{} Stone'.format(boardstone1), 3: 'Renovate'}
        stage3dict = {1: '1 Veg', 2: '{} Board'.format(boardboar)}
        stage4dict = {1: '{} Stone'.format(boardstone2), 2: '{} Cattle'.format(boardcow)}
        stage5dict = {1: 'FG Wout Room', 2: 'Plow & Sow'}

        stage1functiondict = {1: Sheep, 2: Fences, 3: MajorImprovement, 4: SowAndBake}
        stage2functiondict = {1: FamilyGrowth, 2: Stone1, 3: Renovate}
        stage3functiondict = {1: Vegetable, 2: Boar}
        stage4functiondict = {1: Stone2, 2: Cattle}
        stage5functiondict = {1: FamilyGrowthWithoutRoom, 2: PlowAndSow}

        stage1card, stage1function = stage1dict[stage1order[0]], stage1functiondict[stage1order[0]]
        stage2card, stage2function = stage1dict[stage1order[1]], stage1functiondict[stage1order[1]]
        stage3card, stage3function = stage1dict[stage1order[2]], stage1functiondict[stage1order[2]]
        stage4card, stage4function = stage1dict[stage1order[3]], stage1functiondict[stage1order[3]]
        stage5card, stage5function = stage2dict[stage2order[0]], stage2functiondict[stage2order[0]]
        stage6card, stage6function = stage2dict[stage2order[1]], stage2functiondict[stage2order[1]]
        stage7card, stage7function = stage2dict[stage2order[2]], stage2functiondict[stage2order[2]]
        stage8card, stage8function = stage3dict[stage3order[0]], stage3functiondict[stage3order[0]]
        stage9card, stage9function = stage3dict[stage3order[1]], stage3functiondict[stage3order[1]]
        stage10card, stage10function = stage4dict[stage4order[0]], stage4functiondict[stage4order[0]]
        stage11card, stage11function = stage4dict[stage4order[1]], stage4functiondict[stage4order[1]]
        stage12card, stage12function = stage5dict[stage5order[0]], stage5functiondict[stage5order[0]]
        stage13card, stage13function = stage5dict[stage5order[1]], stage5functiondict[stage5order[1]]
        stage14card = 'Reno & Fences'

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    x_change = -5
                if event.key == pygame.K_RIGHT:
                    x_change = 5
                if event.key == pygame.K_p:
                    pause = True
                    paused()

        draw_borders()
        draw_player_inventory(Player1)
        draw_player_inventory(Player2)
        if numplayers >= 3: draw_player_inventory(Player3)
        if numplayers >= 4: draw_player_inventory(Player4)
        if numplayers >= 5: draw_player_inventory(Player5)

        if numplayers == 2: maxturn = Player1.familysize + Player2.familysize
        elif numplayers == 3: maxturn = Player1.familysize + Player2.familysize + Player3.familysize
        elif numplayers == 4: maxturn = Player1.familysize + Player2.familysize + Player3.familysize + Player4.familysize
        elif numplayers == 5: maxturn = Player1.familysize + Player2.familysize + Player3.familysize + Player4.familysize + Player5.familysize

        #Buttons
        button("Build Rooms", 2 * display_width / 10, 0, buttonW, smallbuttonH, green, bright_green, Build_Room)
        button("SP {} Food".format(boardstartingplayerfood), 2 * display_width / 10, smallbuttonH, buttonW, smallbuttonH, green, bright_green, Starting_Player)
        button("1 Grain", 2 * display_width / 10, smallbuttonH * 2, buttonW, smallbuttonH, green, bright_green, Take_1_Grain)
        button("Plow Field", 2 * display_width / 10, smallbuttonH * 3, buttonW, smallbuttonH, green, bright_green, Plow_Field)
        button("Stable BB", 2 * display_width / 10, smallbuttonH * 4, buttonW, smallbuttonH, green, bright_green, Build_Stable)
        button("Day Laborer", 2 * display_width / 10, smallbuttonH * 5, buttonW, smallbuttonH, green, bright_green, Day_Laborer)
        button(stage1card, 3 * display_width / 10, 0, buttonW, largebuttonH, green, bright_green, stage1function)
        button("{} Wood".format(boardwood), 3 * display_width / 10, smallbuttonH * 2, buttonW, smallbuttonH, green, bright_green, Wood)
        button("{} Clay".format(boardclay), 3 * display_width / 10, smallbuttonH * 3, buttonW, smallbuttonH, green, bright_green, Clay)
        button("{} Reed".format(boardreed), 3 * display_width / 10, smallbuttonH * 4, buttonW, smallbuttonH, green, bright_green, Reed)
        button("{} Fish".format(boardfish), 3 * display_width / 10, smallbuttonH * 5, buttonW, smallbuttonH, green, bright_green, Fish)
        if stage >= 2: button(stage2card, 4 * display_width / 10, 0, buttonW, largebuttonH, green, bright_green, stage2function)
        if stage >= 3: button(stage3card, 4 * display_width / 10, largebuttonH, buttonW, largebuttonH, green, bright_green, stage3function)
        if stage >= 4: button(stage4card, 4 * display_width / 10, largebuttonH * 2, buttonW, largebuttonH, green, bright_green, stage4function)
        if stage >= 5: button(stage5card, 5 * display_width / 10, 0, buttonW, largebuttonH, green, bright_green, stage5function)
        if stage >= 6: button(stage6card, 5 * display_width / 10, largebuttonH, buttonW, largebuttonH, green, bright_green, stage6function)
        if stage >= 7: button(stage7card, 5 * display_width / 10, largebuttonH * 2, buttonW, largebuttonH, green, bright_green, stage7function)
        if stage >= 8: button(stage8card, 6 * display_width / 10, largebuttonH, buttonW, largebuttonH, green, bright_green, stage8function)
        if stage >= 9: button(stage9card, 6 * display_width / 10, largebuttonH * 2, buttonW, largebuttonH, green, bright_green, stage9function)
        if stage >= 10: button(stage10card, 7 * display_width / 10, largebuttonH, buttonW, largebuttonH, green, bright_green, stage10function)
        if stage >= 11: button(stage11card, 7 * display_width / 10, largebuttonH * 2, buttonW, largebuttonH, green, bright_green, stage11function)
        if stage >= 12: button(stage12card, 8 * display_width / 10, largebuttonH, buttonW, largebuttonH, green, bright_green, stage12function)
        if stage >= 13: button(stage13card, 8 * display_width / 10, largebuttonH * 2, buttonW, largebuttonH, green, bright_green, stage13function)
        if stage >= 14: button(stage14card, 9 * display_width / 10, largebuttonH, buttonW, largebuttonH, green, bright_green, Build_Room)

        draw_player_position(Player1)
        draw_player_position(Player2)
        if numplayers >= 3: draw_player_position(Player3)
        if numplayers >= 4: draw_player_position(Player4)
        if numplayers >= 5: draw_player_position(Player5)

        if prevaction != action and action != 'start' and Player1.startingplayer == 1:
            if turn % 2 == 1:
                perform_action(Player1)
                turn += 1
            elif turn % 2 == 0:
                perform_action(Player2)
                turn += 1
                Player1.familymember += 1
                Player2.familymember += 1
        elif prevaction != action and action != 'start' and Player2.startingplayer == 1:
            if turn % 2 == 0:
                perform_action(Player2)
                turn += 1
            elif turn % 2 == 1:
                perform_action(Player1)
                turn += 1
                Player1.familymember += 1
                Player2.familymember += 1

        if turn > maxturn:
            Player_Reset(Player1, 0)
            Player_Reset(Player2, 2)
            turn = 1
            Board_Reset()
            action = None

        pygame.display.update()
        clock.tick(60)

game_intro()
game_loop()
pygame.quit()
quit()