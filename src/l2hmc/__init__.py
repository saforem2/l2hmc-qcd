"""
l2hmc/__init__.py 
"""
from __future__ import absolute_import, annotations, division, print_function
import logging

# from colorlog import ColoredFormatter

# # formatter = ColoredFormatter(
# # 	"%(log_color)s%(levelname)-8s%(reset)s %(message_log_color)s%(message)s",
# # 	secondary_log_colors={
# # 		'message': {
# # 			'ERROR':    'red',
# # 			'CRITICAL': 'red'
# # 		}
# # 	}
# # )

# formatter = ColoredFormatter(
# 	"%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
# 	datefmt=None,
# 	reset='%X',
# 	log_colors={
# 		'DEBUG':    'cyan',
# 		'INFO':     'green',
# 		'WARNING':  'yellow',
# 		'ERROR':    'red',
# 		'CRITICAL': 'red,bg_white',
# 	},
# 	secondary_log_colors={},
# 	style='%'
# )

# handler = colorlog.StreamHandler()
# handler.setFormatter(colorlog.ColoredFormatter(
# 	'%(log_color)s%(levelname)s:%(name)s:%(message)s'))

# logger = colorlog.getLogger('example')
# logger.addHandler(handler)

# the handler determines where the logs go: stdout/file
# shell_handler = RichHandler()
# file_handler = logging.FileHandler("debug.log")

# logger.setLevel(logging.DEBUG)
# shell_handler.setLevel(logging.DEBUG)
# file_handler.setLevel(logging.DEBUG)

# # the formatter determines what our logs will look like
# fmt_shell = '%(message)s'
# fmt_file = (
#     '%(levelname)s %(asctime)s '
#     '[%(filename)s:%(funcName)s:%(lineno)d] '
#     '%(message)s'
# )

# shell_formatter = logging.Formatter(fmt_shell)
# file_formatter = logging.Formatter(fmt_file)

# # here we hook everything together
# shell_handler.setFormatter(shell_formatter)
# file_handler.setFormatter(file_formatter)

# logger.addHandler(shell_handler)
# logger.addHandler(file_handler)
