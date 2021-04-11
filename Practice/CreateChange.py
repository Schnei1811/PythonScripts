valtoonie = 200
valloonie = 100
valquarter = 25
valdime = 10
valnickle = 5
valpenny = 1

amntgiven = 2000
cost = 1324

#changeamnt = 746

changeamnt = amntgiven - cost

changedict = {"twoonie":0, "loonie":0, "quarter":0, "dime":0, "nickle":0, "penny":0}


def determinecoinchange(changeamnt, valcoin, stringcoin):
    if changeamnt // valcoin > 0:
        changedict[stringcoin] = changeamnt // valcoin
        changeamnt -= (changeamnt // valcoin) * valcoin
    return changeamnt

changeamnt = determinecoinchange(changeamnt, valtoonie, "twoonie")
changeamnt = determinecoinchange(changeamnt, valloonie, "loonie")
changeamnt = determinecoinchange(changeamnt, valquarter, "quarter")
changeamnt = determinecoinchange(changeamnt, valdime, "dime")
changeamnt = determinecoinchange(changeamnt, valnickle, "nickle")
changeamnt = determinecoinchange(changeamnt, valpenny, "penny")

print(changedict)

