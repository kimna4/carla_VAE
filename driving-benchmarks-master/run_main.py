#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# import sys
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import argparse
import logging

from version084.benchmark_tools import run_driving_benchmark
from version084.driving_benchmarks import CoRL2017, CARLA100
from version084.benchmark_tools.experiment_suites.basic_experiment_suite import BasicExperimentSuite
from version084.benchmark_tools.agent import ForwardAgent

from agents.starlab.modules.imitation_learning import ImitationLearning

if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='verbose',
        help='print some extra status information')
    argparser.add_argument(
        '-db', '--debug',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2006,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-c', '--city-name',
        metavar='C',
        # default='Town01',
        default='Town02',
        help='The town that is going to be used on benchmark'
             + '(needs to match active town in server, options: Town01 or Town02)')
    argparser.add_argument(
        '-n', '--log_name',
        metavar='T',
        default='starlab_210621',
        help='The name of the log file to be created by the benchmark'
    )
    argparser.add_argument(
        '--corl-2017',
        action='store_true',
        help='If you want to benchmark the corl-2017 instead of the Basic one'
    )
    argparser.add_argument(
        '--carla100',
        action='store_true',
        help='If you want to use the carla100 benchmark instead of the Basic one'
    )
    argparser.add_argument(
        '--continue-experiment',
        action='store_true',
        help='If you want to continue the experiment with the same name'
    )

    ### from old eval source code
    argparser.add_argument(
        '--model-path',
        metavar='P',
        # default='/home/kimna/PytorchWorkspace/CARLA_starlab/starlab_2022/save_models/training_best.pth',
        default='/home/kimna/PytorchWorkspace/CARLA_starlab/starlab_2022/save_models/training_210621_best.pth',
        # default='/home/kimna/PytorchWorkspace/CARLA_starlab/starlab_2022/save_models/training_210621/5_training_210621.pth',
        type=str,
        help='torch imitation learning model path (relative in model dir)'
    )
    argparser.add_argument(
        '--visualize',
        default=False,
        action='store_true',
        help='visualize the image and transfered image through tensorflow'
    )
    argparser.add_argument(
        '--avoid-stopping',
        default=True,
        action='store_false',
        help=' Uses the speed prediction branch to avoid unwanted agent stops'
    )

    argparser.add_argument('--class-latent-size', default=128, type=int)
    argparser.add_argument('--content-latent-size', default=128, type=int)
    argparser.add_argument('--vae-model-dir',
                        default="/home/kimna/PytorchWorkspace/CARLA_starlab/Disentanglement_VAE/save_models/training_v2/48_training_v2.pth",
                        type=str, metavar='PATH')

    args = argparser.parse_args()
    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING

    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)

    # We instantiate a forward agent, a simple policy that just set
    # acceleration as 0.9 and steering as zero
    # agent = ForwardAgent()
    agent = ImitationLearning(args.city_name,
                              args.avoid_stopping,
                              args.model_path,
                              args.visualize,
                              args.log_name,
                              args.class_latent_size,
                              args.content_latent_size,
                              vae_model_path = args.vae_model_dir
                              )

    # We instantiate an experiment suite. Basically a set of experiments
    # that are going to be evaluated on this benchmark.
    if args.corl_2017:
        experiment_suite = CoRL2017(args.city_name)
    elif args.carla100:
        experiment_suite = CARLA100(args.city_name)
    else:
        print (' WARNING: running the basic driving benchmark, to run for CoRL 2017'
               ' experiment suites, you should run'
               ' python driving_benchmark_example.py --corl-2017')
        experiment_suite = BasicExperimentSuite(args.city_name)

    # Now actually run the driving_benchmark
    run_driving_benchmark(agent, experiment_suite, args.city_name,
                          args.log_name, args.continue_experiment,
                          args.host, args.port)
