#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import argparse
import datetime
import os
from random import randint

import pandas as pd
from langdetect import *
from lxml import etree

# parser = argparse.ArgumentParser()
# parser.add_argument("-ownName", dest='ownName', type=str,
#                     help="name of the owner of the chat logs, written as in the logs", required=True)
# parser.add_argument('-f', '-filePath', dest='filePath', help='Facebook chat log file (HTML file)',
#                     default='raw/messages')
# parser.add_argument("-max", "-maxExportedMessages", dest='maxExportedMessages', type=int, default=1000000,
#                     help="maximum number of messages to export")
# args = parser.parse_args()
#
# maxExportedMessages = args.maxExportedMessages
# ownName = args.ownName
# filePath = args.filePath

maxExportedMessages = 1000000
ownName = 'Stefan'
filePath = "D:PythonData/FacebookData/Tina"
filePath = "D:PythonData/FacebookData/Alina"

data = []
warnedNameChanges = []
nbInvalidSender = 0

for filename in os.listdir(filePath):

    if not filename.endswith('.html'):
        continue

    archive = etree.parse(filePath + "/" + filename)
    conversationId = filename.replace('.html', '')
    groupConversation = False
    timestamp = ''
    senderName = ''
    conversationWithName = None
    text = ''

    for element in archive.iter():
        tag = element.tag
        className = element.get('class')
        content = element.text

        print className

        if tag == 'p':
            text = content

            if conversationWithName is not "" and senderName is not "":

                # handles when the interlocutor's name changed at some point
                if (senderName != conversationWithName) and (senderName != ownName) and (senderName not in warnedNameChanges) and (not groupConversation):
                    if senderName not in warnedNameChanges:
                        print '\t', 'Assuming', senderName, 'is', conversationWithName
                        warnedNameChanges.append(senderName)

                    senderName = conversationWithName

                data += [[timestamp, conversationId, conversationWithName, senderName, text]]

            else:
                nbInvalidSender = nbInvalidSender + 1

        elif tag == 'span':
            if className == 'user':
                senderName = content
            elif className == 'meta':
                timestamp = pd.to_datetime(content[:-4], format='%A, %B %d, %Y at %H:%M%p').strftime("%S")

        elif tag == 'div' and className == 'thread':
            nbParticipants = str(element.xpath("text()")).count(', ') + 1
            if nbParticipants > 1:
                groupConversation = True

        elif tag == 'h3':
            if conversationWithName is not None:
                print 'Something is wrong. File format changed? (multiple conversation hearder in a single file)'
                exit(0)
            else:
                content = content.replace('Conversation with ', '')
                conversationWithName = content

            print conversationId, conversationWithName, "(group?", groupConversation, ")"

        else:
            print '{}'.format(tag)

        if len(data) >= maxExportedMessages:
            break

print len(data), 'messages parsed.'

if nbInvalidSender > 0:
    print nbInvalidSender, 'messages discarded because of bad ID.'

if len(data) < 1:
    print 'Nothing to save.'
    exit(0)

print 'Converting to DataFrame...'
df = pd.DataFrame(data)
df.columns = ['timestamp', 'conversationId', 'conversationWithName', 'senderName', 'text']
df['platform'] = 'messenger'

print 'Detecting languages...'
df['language'] = 'unknown'
for name, group in df.groupby(df.conversationWithName):
    sample = ''
    df2 = df[df.conversationWithName == name].dropna()

    if len(df2) > 10:
        for x in range(0, min(len(df2), 100)):
            sample = sample + df2.iloc[randint(0, len(df2) - 1)]['text']

        print '\t', name, detect(sample), "(", len(df2), "msgs)"
        df.loc[df.conversationWithName == name, 'language'] = detect(sample)

print 'Computing dates...'
df['datetime'] = df[df["timestamp"] != ""]['timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(float(x)).toordinal())

print df.head()

print 'Saving to pickle file...'
df.to_pickle('data/messenger.pkl')

print 'Done.'
