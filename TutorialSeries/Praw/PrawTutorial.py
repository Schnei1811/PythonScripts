import praw

reddit = praw.Reddit(client_id='lIYT126pXfeHQg',
                     client_secret='lIY9DpsRPiR_IYc0XVPZ_psRkDE',
                     username='Schnei1811',
                     password='Water1bottle',
                     user_agent='Schnei1811v1')

subreddit = reddit.subreddit('python')

hot_python = subreddit.hot(limit=5)

for submission in hot_python:
    if not submission.stickied:
        print('Title: {}, ups: {}, downs: {}, Have we visited: {}'.format(submission.title,  #.upvote() .downvote() .reply()
                                                                          submission.ups,
                                                                          submission.downs,
                                                                          submission.visited))
        submission.comments.replace_more(limit=0)
        for comment in submission.comments.list():
            print(20*'-')
            print('Parent ID:', comment.parent())
            print('Comment ID:', comment.id)
            print(comment.body)

#subreddit.subscribe()


