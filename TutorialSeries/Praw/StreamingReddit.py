import praw

reddit = praw.Reddit(client_id='lIYT126pXfeHQg',
                     client_secret='lIY9DpsRPiR_IYc0XVPZ_psRkDE',
                     username='Schnei1811',
                     password='Water1bottle',
                     user_agent='Schnei1811v1')

subreddit = reddit.subreddit('politics')

# for submission in subreddit.stream.submissions():
#     try:
#         print(submission.title)
#     except Exception as e:
#         print(str(e))

for comment in subreddit.stream.comments():
    try:
        print(comment.body)
        # parent_id = str(comment.parent())
        # original = reddit.comment(parent_id)
        # print('Parent:')
        # print(original.body)
        # print('Reply:')
        # print(comment.body)

    except praw.exceptions.PRAWException as e: pass
