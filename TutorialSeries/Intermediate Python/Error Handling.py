import sys
import logging

try: a+b
except Exception as e:
    print(sys.exc_info())
    print(sys.exc_info()[0])
    print(sys.exc_info()[1])
    print(sys.exc_info()[2].tb_lineno)

    print('Error: {}. {}, line: {}'.format(sys.exc_info()[0],sys.exc_info()[1],sys.exc_info()[2]))


def error_handling(): return 'Error: {}. {}, line: {}'.format(sys.exc_info()[0],sys.exc_info()[1],sys.exc_info()[2])

try: a+b
except Exception as e: logging.error(error_handling())
