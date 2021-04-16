#!/usr/bin/env python                                            
#
# fetal_brain_assessment ds ChRIS plugin app
#
# (c) 2021 Fetal-Neonatal Neuroimaging & Developmental Science Center
#                   Boston Children's Hospital
#
#              http://childrenshospital.org/FNNDSC/
#                        dev@babyMRI.org
#

from chrisapp.base import ChrisApp
from argparse import ArgumentDefaultsHelpFormatter
import colorlog
import logging
import os
from os import path
from glob import glob
import tensorflow as tf
from fetal_brain_assessment.predict_resnet import Predictor
from fetal_brain_assessment.volume import Volume

Gstr_title = r"""
 _____             _ _ _            ___                                             _   
|  _  |           | (_) |          / _ \                                           | |  
| | | |_   _  __ _| |_| |_ _   _  / /_\ \___ ___  ___  ___ ___ _ __ ___   ___ _ __ | |_ 
| | | | | | |/ _` | | | __| | | | |  _  / __/ __|/ _ \/ __/ __| '_ ` _ \ / _ \ '_ \| __|
\ \/' / |_| | (_| | | | |_| |_| | | | | \__ \__ \  __/\__ \__ \ | | | | |  __/ | | | |_ 
  \_/\_\__,_|\__,_|_|_|\__|\__, | \_| |_/___/___/\___||___/___/_| |_| |_|\___|_| |_|\__|
                            __/ |                                                       
                           |___/                                                        
"""

handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(thin_green)s[%(asctime)s]%(reset)s  '
    '%(log_color)s%(levelname)-8s'
    '%(cyan)s%(filename)s:%(funcName)s:%(lineno)-4s | '
    '%(log_color)s%(message)s%(reset)s'
))

logger = colorlog.getLogger()
logger.addHandler(handler)


class Fetal_brain_assessment(ChrisApp):
    """
    The aim of this project was to develop a Quality Assessment tool for fetal brain MRIs,
    which is able to score each volume through a deep learning regression model.
    Developed using Python3 and Keras/Tensorflow framework.
    """
    PACKAGE                 = __package__
    TITLE                   = 'Fetal Brain Quality Assessment'
    CATEGORY                = 'Statistical Operation'
    TYPE                    = 'ds'
    ICON                    = '' # url of an icon image
    MAX_NUMBER_OF_WORKERS   = 1  # Override with integer value
    MIN_NUMBER_OF_WORKERS   = 1  # Override with integer value
    MAX_CPU_LIMIT           = '' # Override with millicore value as string, e.g. '2000m'
    MIN_CPU_LIMIT           = '' # Override with millicore value as string, e.g. '2000m'
    MAX_MEMORY_LIMIT        = '' # Override with string, e.g. '1Gi', '2000Mi'
    MIN_MEMORY_LIMIT        = '' # Override with string, e.g. '1Gi', '2000Mi'
    MIN_GPU_LIMIT           = 1  # Override with the minimum number of GPUs, as an integer, for your plugin
    MAX_GPU_LIMIT           = 1  # Override with the maximum number of GPUs, as an integer, for your plugin

    # Use this dictionary structure to provide key-value output descriptive information
    # that may be useful for the next downstream plugin. For example:
    #
    # {
    #   "finalOutputFile":  "final/file.out",
    #   "viewer":           "genericTextViewer",
    # }
    #
    # The above dictionary is saved when plugin is called with a ``--saveoutputmeta``
    # flag. Note also that all file paths are relative to the system specified
    # output directory.
    OUTPUT_META_DICT = {}

    def __init__(self):
        super().__init__()
        self.formatter_class = ArgumentDefaultsHelpFormatter

    def define_parameters(self):
        # this is a future spec
        # https://github.com/FNNDSC/chrisapp/issues/6
        self.add_argument(
            '-p', '--inputPathFilter',
            dest='inputPathFilter',
            help='Selection for which files to evaluate.',
            default='*.nii',
            type=str,
            optional=True
        )
        self.add_argument(
            '-o', '--output-file',
            dest='output_filename',
            help='Name of output CSV file.',
            default='quality_assessment.csv',
            type=str,
            optional=True
        )
        self.add_argument(
            '-c', '--crop-destination',
            dest='crop_destination',
            help='Name of directory of where to save cropped images, before selection.',
            default='',
            type=str,
            optional=True
        )
        self.add_argument(
            '-d', '--destination',
            dest='destination_folder',
            help='Name of directory of where to save images that meet the threshold.',
            default='Best_Images_crop',
            type=str,
            optional=True
        )
        self.add_argument(
            '-t', '--threshold',
            dest='threshold',
            help='Score threshold for images to keep.',
            type=float,
            default=0.4,
            optional=True
        )

    def run(self, options):
        # legacy thing in chrisapp==2.1.0
        verbosity = int(options.verbosity)
        if verbosity > 0:
            logger.setLevel(logging.DEBUG if verbosity > 1 else logging.INFO)
            logging.debug('\n' + Gstr_title)

        tf.get_logger().setLevel(logging.getLevelName(logger.level))

        input_pattern = path.join(options.inputdir, options.inputPathFilter)
        input_files = glob(input_pattern)
        output_file = path.join(options.outputdir, options.output_filename)

        if not input_files:
            logger.warning('No input files found in "%s"', input_pattern)
            return

        # load data from files and crop
        volumes = [Volume(f) for f in input_files]

        # save cropped volumes if desired
        if options.crop_destination:
            crop_folder = path.join(options.outputdir, options.crop_destination)
            os.mkdir(crop_folder)
            for volume in volumes:
                volume.save_cropped(crop_folder)

        df = Predictor().predict([v.padded_data for v in volumes], input_files)

        df['slice_thickness'] = [v.slice_thickness for v in volumes]

        logger.debug('Saving results to %s', output_file)
        df.to_csv(output_file, index=False, header=True)

        logger.debug('threshold=%s', str(options.threshold))
        if options.threshold > 1 or not options.destination_folder:
            return

        selected_dir = path.join(options.outputdir, options.destination_folder)
        os.mkdir(selected_dir)
        for row, volume in zip(df.itertuples(index=False), volumes):
            to_keep = row.quality >= options.threshold
            logger.debug('%-60s  %-1.6f  %s', row.filename, row.quality,
                         'SELECTED' if to_keep else 'REJECTED')
            if to_keep:
                volume.save_cropped(selected_dir)

    def show_man_page(self):
        self.print_help()
