from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sentiment_mod as s
import sys

# consumer key, consumer secret, access token, access secret.
ckey = "tHgGqGcIKWa4zjtiYuhR64ELb"
csecret = "QHlmITwizIXfT8SRrwQ4sIXx2NAjTHK6bk3iaLYglWvk5tJtmP"
atoken = "1307496877-Y8VfQUAMG5RHwpn7qy2AHlDAjjT3kBwfR6xdoC8"
asecret = "0iR9HGmYTIQbslsQm3EHTowePXqNatL13OxZnLTCnrX60"

class listener(StreamListener):
    def on_data(self, data):
        all_data = json.loads(data)

        tweet = all_data["text"]
        sentiment_value, confidence = s.sentiment(tweet)
        print((tweet, sentiment_value, confidence))

        if confidence*100 >= 80:
            output = open('twitter-out.txt','a')
            output.write(sentiment_value)
            output.write('\n')
            output.close()
        return True

    def on_error(self, status):
        print(status)


auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["bernie"])