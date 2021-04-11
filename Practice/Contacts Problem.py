# Write a program that allows you to add names and tell you after you enter
# a string, how many contacts start with that string





def binarysearch(contactlist, val, count):
    first = 0
    last = len(contactlist) - 1

    while first <= last:
        mid = (first + last) // 2
        if contactlist[mid][:len(val)] == val:
            count += 1
            del contactlist[mid]
            return binarysearch(contactlist, val, count)
        elif contactlist[mid] > val: last = mid - 1
        else: first = mid + 1
    return count


def searchcontacts(contactlist):
    count = 0
    inputlet = input('Enter search letters: ').lower()
    if inputlet == '':
        print('Input Empty')
        searchcontacts(contactlist)
    contactlist.sort()
    contactlist = [contact.lower() for contact in contactlist]
    print(binarysearch(contactlist, inputlet, count))
    return


def addcontact(contactlist):
    inputname = input('Add new contact: ')
    if inputname == '':
        print('Input Empty')
        addcontact(contactlist)
    contactlist += inputname
    ans = input('Add another contact? (y/n) ')
    if ans == 'y': addcontact(contactlist)
    if ans == 'n':
        file = open('contactlist.txt', 'w')
        file.writelines(["{}".format(line) for line in contactlist])
        file.close()
    return


def main():
    contactlist = ['Anna', 'Anne', 'Bob', 'Bobby', 'George', 'Sarah']
    ans = input("Add or Search Contact List?: (add, search, quit) ")
    if ans == 'add': addcontact(contactlist)
    elif ans == 'search': searchcontacts(contactlist)
    elif ans == 'quit': quit()
    else:
        print('Invalid selection')
    main()



main()